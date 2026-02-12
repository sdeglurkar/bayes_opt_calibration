import GPy
import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
from scipy.stats import norm

from bolevelset import BOLevelSet
from find_bound import find_epsLU


########################### REACHABILITY SETTINGS ###########################
THETA_VALUE = 0.0
DUBINS_VELOCITY = 10.0
DT = 0.01
FINAL_TIME = -1.0
TRAJ_TIME_STEPS = int(np.abs(FINAL_TIME)/DT)
goal_R = 5

########################### GAUSSIAN PROCESS SETTINGS ###########################
CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NUM_BO_INIT_ITERS = 40 #10   # Amount of initial random samples
# NUM_BO_ITERS = 10 #30  # Set to 0 if only random sampling is desired

########################### RANDOM SEED SETTINGS ###########################
RANDOM_SEED = 0 #100  # If only a single seed is being run
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = False 
MULTIPLE_SEED_LIST = [0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DELTA = 1e-4
WORLD_DISCRETIZATION = 0.5  # Dictates the size of D if gridding the space
DESIRED_N = 3600 #500 #1000 #3600

########################### OTHER SETTINGS ###########################
VALIDATION_DISCRETIZATION = 0.5
PLOT_DURING_ACQUISITION = False
PLOT_D = False
PLOT_VALIDATION_DATA = False

########################### GET OPTIMAL VALUE FUNCTION -- BRT ###########################
class DubinsCar(hj.ControlAndDisturbanceAffineDynamics):

    '''
    xdot = v*cos(theta)
    ydot = v*sin(theta) 
    vdot = u1
    
    state: [x, y, theta]
    '''

    def __init__(self,
                velocity,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        
        self.velocity = velocity

        if control_space is None:
            control_space = hj.sets.Box(
                lo=jnp.array([-np.pi/2]),
                hi=jnp.array([np.pi/2])
            )
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(
                lo=jnp.array([0.0]),
                hi=jnp.array([0.0])
            )
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
    
    def open_loop_dynamics(self, state, time):
        """Implements the open loop dynamics `f(x, t)`."""
        _, _, theta = state
        return jnp.array([self.velocity * jnp.cos(theta), self.velocity * jnp.sin(theta), 0.])

    def optimal_control(self, state, time, grad_value):
        best_u = self.control_space.extreme_point(jnp.array([grad_value[2]]))
        if self.control_mode == 'max':
            return best_u
        elif self.control_mode == 'min':
            return -best_u
    
    def optimal_disturbance(self, state, time, grad_value):
        return None 

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [0.]
        ]) 

dynamics = DubinsCar(DUBINS_VELOCITY)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-15., -15., 0.]),
                                                                           np.array([15., 15., 2 * np.pi])),
                                                               (51, 40, 50),
                                                               periodic_dims=2)
index = grid.nearest_index([0., 0., THETA_VALUE])
THETA_INDEX = index[2]

range_x = [[-15, 15], [-15, 15]]

values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - goal_R
print("Running HJR Solver")
solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

times = np.linspace(0, FINAL_TIME, 100) 
initial_values = values
all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values)
print("Done!")

########################### HELPER FUNCTIONS ###########################
def batched_rollouts_generator(value_fn, time_steps=TRAJ_TIME_STEPS, dt=DT):
    value_gradients = grid.grad_values(value_fn[-1, :, :, :])
    def rollout(state, plot_traj=False):
        orig_state = state
        state_traj = [state]
        for _ in range(time_steps):
            index = grid.nearest_index(state)
            grad_val = value_gradients[index[0], index[1], index[2]]
            control = dynamics.optimal_control(None, None, grad_val)
            state = state + DT*dynamics(state, control, disturbance=np.array([0.]), time=None)
            state_traj.append(state)
        state_traj = np.array(state_traj)

        cost = jnp.min(jnp.linalg.norm(state_traj, axis=-1) - goal_R, axis=-1)
        print("Ran one rollout", orig_state, cost)
        return cost
    
    def batched_rollouts(states):
        costs = []
        for state in states:
            state = np.append(state, THETA_VALUE)
            cost = np.array(rollout(state))
            costs.append(cost)
        print("Done running rollouts!")
        return np.expand_dims(np.array(costs), -1)
    
    return batched_rollouts

def plot_main_gp(bols, grid, candidates, oned_x, fig_name, fig_name_colorbar, mile_name):
    plt.figure()
    # bols.m.plot()
    plt.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                all_values[-1, :, :, THETA_INDEX].T,
                levels=[0.0],
                colors="black",
                linewidths=3)
    mu, var = bols.m.predict(candidates, full_cov=False)
    criterion = mu + BETA * np.sqrt(var)  
    criterion = criterion.reshape(len(oned_x), len(oned_x))
    # plt.contour(oned_x,
    #             oned_x,
    #             criterion,
    #             levels=[0.0],
    #             colors="lightblue",
    #             linewidths=2)
    criterion = mu - BETA * np.sqrt(var)  
    criterion = criterion.reshape(len(oned_x), len(oned_x))
    plt.contour(oned_x,
                oned_x,
                criterion,
                levels=[0.0],
                colors="blue",
                linewidths=2)
    if len(bols.acq_cache) > 0:
        acq_points = np.array(bols.acq_cache).squeeze()
        plt.scatter(acq_points[:, 0], acq_points[:, 1], color='g')
    plt.savefig(fig_name)
    plt.figure()
    plt.contourf(oned_x,
                oned_x,
                criterion)
    plt.colorbar()
    plt.savefig(fig_name_colorbar)
    plt.figure()
    x = bols.candidates[:, 0].flatten()
    y = bols.candidates[:, 1].flatten()
    original_cand_len = int(np.sqrt(len(x)))
    original_x = x.reshape((original_cand_len, original_cand_len))[0, :]
    original_y = y.reshape((original_cand_len, original_cand_len))[:, 0]
    z = bols.acq.reshape((original_cand_len, original_cand_len))
    plt.contourf(original_x, original_y, BOLevelSet.normalize(z))
    plt.colorbar()
    plt.savefig(mile_name)

    return original_cand_len

def get_ground_truths_for_a_grid(f, discretization):
    print("\nRunning get_ground_truths_for_a_grid!") 
    oned_x = np.arange(-15, 15, discretization) 
    xv, yv = np.meshgrid(oned_x, oned_x)
    candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
    costs = f(candidates)
    print("Done!")
    return candidates, costs

def get_ground_truths_for_random_points(range_x, rng, f, size_set):
    print("\nRunning get_ground_truths_for_random_points!") 
    X_init = []
    for i in range(2):
        X_i_init = rng.uniform(
            range_x[i][0],
            range_x[i][1],
            (size_set,)
        )
        X_init.append(X_i_init)
    random_points = np.stack(X_init, axis=1)
    costs_at_random_points = f(random_points)
    print("Done!")
    return random_points, costs_at_random_points

def validate_final_level_set(bols, candidates, true_costs):
    print("Running validation of final GP level set")
    mu, var = bols.m.predict(candidates, full_cov=False)
    criterion = mu - BETA * np.sqrt(var) # Conservative bc more likely to say state is in BRT
    
    assert len(criterion) == len(true_costs)

    gp_below_zero = np.where(criterion <= 0)
    gp_above_zero = np.where(criterion > 0)
    true_below_zero = np.where(true_costs <= 0)
    true_above_zero = np.where(true_costs > 0)

    true_pos = np.intersect1d(gp_below_zero, true_below_zero)
    false_pos = np.intersect1d(gp_below_zero, true_above_zero)
    true_neg = np.intersect1d(gp_above_zero, true_above_zero)
    false_neg = np.intersect1d(gp_above_zero, true_below_zero)

    tpr = len(true_pos) / (len(true_pos) + len(false_neg))
    fpr = len(false_pos) / (len(false_pos) + len(true_neg))
    tnr = len(true_neg) / (len(true_neg) + len(false_pos))
    fnr = len(false_neg) / (len(false_neg) + len(true_pos))

    print("Done running validation!")

    return tpr, fpr, tnr, fnr

F = batched_rollouts_generator(all_values)

########################### GET D ALONG WITH GROUND TRUTH COSTS ###########################
if os.path.isfile(f"D_{DESIRED_N}.pkl"):
    with open(f"D_{DESIRED_N}.pkl", "rb") as f:
        D_candidates, D_true_costs = pickle.load(f)
else:
    # D_candidates, D_true_costs = get_ground_truths_for_a_grid(F, discretization=WORLD_DISCRETIZATION)
    D_candidates, D_true_costs = get_ground_truths_for_random_points(range_x, RNG, F, DESIRED_N)
    with open(f'D_{DESIRED_N}.pkl', 'wb') as f:
        pickle.dump([D_candidates, D_true_costs], f)
N = len(D_candidates)
assert N == DESIRED_N
if PLOT_D:
    plt.scatter(D_candidates[:, 0], D_candidates[:, 1])
    plt.show()

########################### GET VALIDATION DATASET ###########################
# if os.path.isfile("validation_data.pkl"):
#     with open("validation_data.pkl", "rb") as f:
#         validation_candidates, validation_true_costs = pickle.load(f)
# else:
#     validation_candidates, validation_true_costs = get_ground_truths_for_a_grid(F, discretization=VALIDATION_DISCRETIZATION)
#     with open('validation_data.pkl', 'wb') as f:
#         pickle.dump([validation_candidates, validation_true_costs], f)
# if PLOT_VALIDATION_DATA:
#     plt.scatter(validation_candidates[:, 0], validation_candidates[:, 1])
#     plt.show()

########################### FIT INITIAL GAUSSIAN PROCESS ###########################
mean_function = GPy.core.Mapping(2,1)
mean_function.f = lambda x: np.expand_dims(x[:, 0]**2 + x[:, 1]**2 - goal_R, -1)
mean_function.update_gradients = lambda a,b: 0
mean_function.gradients_X = lambda a,b: 0
input_dim = 2
oned_x = np.arange(-15, 15, 1.0)
xv, yv = np.meshgrid(oned_x, oned_x)
candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
noise_var = 0.001 
cost_thres = 0.0 
length_scale = 0.25
logdir = 'picktolearn_model_dir'
bo_init_iters = NUM_BO_INIT_ITERS
if MULTIPLE_SEEDS:
    bols_list = []
    for i in range(len(MULTIPLE_RNG_LIST)):
        rng = MULTIPLE_RNG_LIST[i]
        seed_val = MULTIPLE_SEED_LIST[i]
        np.random.seed(seed_val)
        bols = BOLevelSet(F, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, 
                        CONF_THRES, length_scale, logdir)
        bols_list.append(bols)
        bols.initial_setup(bo_init_iters, rng)
    print("\nCompleted BOLevelSet initial setup")

    # Plot the initial GP overlaid with its 0-level set and the true BRT
    for i in range(len(bols_list)):
        bols = bols_list[i]
        seed = MULTIPLE_SEED_LIST[i]
        fig_name = logdir + f'/gp_init{seed}.png'
        fig_name_colorbar = logdir + f'/gp_init_colorbar{seed}.png'
        mile_name = logdir + f'/mile_init{seed}.png'
        original_cand_len = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                                            fig_name_colorbar, mile_name)
else:
    np.random.seed(RANDOM_SEED)
    bols = BOLevelSet(F, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, 
                        CONF_THRES, length_scale, logdir)
    bols.initial_setup(bo_init_iters, RNG)
    print("\nCompleted BOLevelSet initial setup")

    # Plot the initial GP overlaid with its 0-level set and the true BRT
    fig_name = bols.logdir + f'/gp_init.png'
    fig_name_colorbar = bols.logdir + f'/gp_init_colorbar.png'
    mile_name = bols.logdir + f'/mile_init.png'
    original_cand_len = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                                        fig_name_colorbar, mile_name)

########################### RUN PICK-TO-LEARN ###########################
if MULTIPLE_SEEDS:
    size_Ts = []
    epsUs = []
    for i in range(len(bols_list)):
        bols = bols_list[i]
        seed = MULTIPLE_SEED_LIST[i]
        rng = MULTIPLE_RNG_LIST[i]
        bols.optimize_loop_picktolearn(D_candidates, D_true_costs, BETA, 
                                                rng, to_plot=PLOT_DURING_ACQUISITION)
        plt.figure()
        plt.plot(range(len(bols.num_counterexamples_list)), bols.num_counterexamples_list)
        plt.savefig(bols.logdir + f'/counterexs{seed}.png')

        acq_cache = np.array(bols.acq_cache).squeeze()
        unique_values, counts = np.unique(acq_cache, axis=0, return_counts=True)
        size_T = len(unique_values)
        size_Ts.append(size_T)
        epsL, epsU = find_epsLU(size_T, N, DELTA)
        epsUs.append(epsU)

        # Plot the final GP overlaid with its 0-level set and the true BRT
        fig_name = bols.logdir + f'/gp_final{seed}.png'
        fig_name_colorbar = bols.logdir + f'/gp_final_colorbar{seed}.png'
        mile_name = bols.logdir + f'/mile_final{seed}.png'
        _ = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                            fig_name_colorbar, mile_name)
    print("SIZE Ts and EPSUs:", size_Ts, epsUs)
else:
    bols.optimize_loop_picktolearn(D_candidates, D_true_costs, BETA, 
                                                RNG, to_plot=PLOT_DURING_ACQUISITION)
    plt.figure()
    plt.plot(range(len(bols.num_counterexamples_list)), bols.num_counterexamples_list)
    plt.xlabel("Iteration")
    plt.ylabel("Number of Counterexamples")
    plt.savefig(bols.logdir + f'/counterexs.png')

    acq_cache = np.array(bols.acq_cache).squeeze()
    unique_values, counts = np.unique(acq_cache, axis=0, return_counts=True)
    size_T = len(unique_values)
    epsL, epsU = find_epsLU(size_T, N, DELTA)
    print("SIZE_T", size_T)
    print("EPS_U", epsU, size_T/N)

    # Plot the final GP overlaid with its 0-level set and the true BRT
    fig_name = bols.logdir + f'/gp_final.png'
    fig_name_colorbar = bols.logdir + f'/gp_final_colorbar.png'
    mile_name = bols.logdir + f'/mile_final.png'
    _ = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                        fig_name_colorbar, mile_name)

########################### VALIDATION OF FINAL GP ###########################
if MULTIPLE_SEEDS:
    results_dict = {}
    for i in range(len(bols_list)):
        bols = bols_list[i]
        seed = MULTIPLE_SEED_LIST[i]
        tpr, fpr, tnr, fnr = validate_final_level_set(bols, validation_candidates, validation_true_costs)
        results_dict[seed] = [tpr, fpr, tnr, fnr]
    keys = list(results_dict.keys())
    all_tprs = [results_dict[key][0] for key in keys]
    all_fnrs = [results_dict[key][3] for key in keys]
    all_fprs = [results_dict[key][1] for key in keys]
    all_tnrs = [results_dict[key][2] for key in keys]
    for key in keys:
        # TPR + FNR = 1
        # FPR + TNR = 1
        print("\nSeed: ", key)
        print("True Positive Rate: ", results_dict[key][0])
        print("False Negative Rate: ", results_dict[key][3])
        print("False Positive Rate: ", results_dict[key][1])
        print("True Negative Rate: ", results_dict[key][2])
    
    print("\nAverage TPR Over All Seeds: ", np.mean(all_tprs))
    print("Average FNR Over All Seeds: ", np.mean(all_fnrs))
    print("Average FPR Over All Seeds: ", np.mean(all_fprs))
    print("Average TNR Over All Seeds: ", np.mean(all_tnrs))
else:
    tpr, fpr, tnr, fnr = validate_final_level_set(bols, validation_candidates, validation_true_costs)
    # TPR + FNR = 1
    # FPR + TNR = 1
    print("True Positive Rate: ", tpr)
    print("False Negative Rate: ", fnr)
    print("False Positive Rate: ", fpr)
    print("True Negative Rate: ", tnr)
    
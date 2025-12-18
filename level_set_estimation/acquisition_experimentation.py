import GPy
import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
from scipy.stats import norm

from bolevelset import BOLevelSet

THETA_VALUE = 0.0
DUBINS_VELOCITY = 10.0
DT = 0.01
FINAL_TIME = -1.0
TRAJ_TIME_STEPS = int(np.abs(FINAL_TIME)/DT)
goal_R = 5
CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NUM_BO_INIT_ITERS = 10 
NUM_BO_ITERS = 30
USE_MILE = False
SIZE_CALIBRATION_SET = 100
RANDOM_SEED = 100 
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = True
MULTIPLE_SEED_LIST = [0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0
if os.path.isfile("calibration_data.pkl"):
    with open("calibration_data.pkl", "rb") as f:
        calibration_data = pickle.load(f)
        [calibration_points, costs_at_calibration_points] = calibration_data
        TO_PICKLE_CALIBRATION = False 
else:
    TO_PICKLE_CALIBRATION = True 
if os.path.isfile("counterexample_search_data.pkl"):
    with open("counterexample_search_data.pkl", "rb") as f:
        counterexample_search_data = pickle.load(f)
        [true_costs, search_candidates] = counterexample_search_data
        TO_PICKLE_COUNTEREXAMPLE = False 
else:
    TO_PICKLE_COUNTEREXAMPLE = True 
if os.path.isfile("validation_data.pkl"):
    with open("validation_data.pkl", "rb") as f:
        validation_data = pickle.load(f)
        TO_PICKLE_VALIDATION = False 
else:
    TO_PICKLE_VALIDATION = True 

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

########################### GET OPTIMAL VALUE FUNCTION -- BRT ###########################
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

def evaluate_costs_for_calibration_set(range_x, rng, f, size_calibration_set): 
    X_init = []
    for i in range(2):
        # X_i_init = np.random.uniform(
        #     range_x[i][0],
        #     range_x[i][1],
        #     (SIZE_CALIBRATION_SET,)
        # )
        X_i_init = rng.uniform(
            range_x[i][0],
            range_x[i][1],
            (size_calibration_set,)
        )
        X_init.append(X_i_init)
    calibration_points = np.stack(X_init, axis=1)
    print("Evaluating costs for calibration set")
    costs_at_calibration_points = f(calibration_points)
    print("Done evaluating costs for calibration set!")

    return calibration_points, costs_at_calibration_points

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
    plt.contour(oned_x,
                oned_x,
                criterion,
                levels=[0.0],
                colors="lightblue",
                linewidths=2)
    criterion = mu - BETA * np.sqrt(var)  
    criterion = criterion.reshape(len(oned_x), len(oned_x))
    plt.contour(oned_x,
                oned_x,
                criterion,
                levels=[0.0],
                colors="blue",
                linewidths=2)
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

def plot_error_gp(error_gp, candidates, oned_x, fig_name):
    mu, var = error_gp.m.predict(candidates, full_cov=False)
    criterion = mu 
    criterion = criterion.reshape(len(oned_x), len(oned_x))
    plt.figure()
    plt.contourf(oned_x,
                oned_x,
                criterion)
    plt.colorbar()
    plt.savefig(fig_name)

def get_ground_truths_for_a_grid(f, discretization):
    print("Adversary searching for failures")
    oned_x = np.arange(-15, 15, discretization) 
    xv, yv = np.meshgrid(oned_x, oned_x)
    search_candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
    true_costs = f(search_candidates)
    print("Done!")
    return search_candidates, true_costs

def validation_get_ground_truths(f, discretization=0.5):
    print("Running validation: ground truth values")
    oned_x = np.arange(-15, 15, discretization) # Should be a much finer grid
    xv, yv = np.meshgrid(oned_x, oned_x)
    candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
    true_costs = f(candidates)
    print("Done!")
    return true_costs, discretization

def validate_final_level_set(bols, true_costs, discretization):
    print("Running validation of final GP level set")
    oned_x = np.arange(-15, 15, discretization) # Should be a much finer grid
    xv, yv = np.meshgrid(oned_x, oned_x)
    candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))

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

########################### EVALUATE COSTS FOR CALIBRATION SET ###########################
range_x = [[-15, 15], [-15, 15]]
if TO_PICKLE_CALIBRATION:
    f = batched_rollouts_generator(all_values)
    calibration_points, costs_at_calibration_points = \
        evaluate_costs_for_calibration_set(range_x, RNG, f, SIZE_CALIBRATION_SET)
    calibration_data = [calibration_points, costs_at_calibration_points]
    with open('calibration_data.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)

########################### FIT GAUSSIAN PROCESS ###########################
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
logdir = 'acq_exp_model_dir'
bo_init_iters = NUM_BO_INIT_ITERS
f = batched_rollouts_generator(all_values)
if MULTIPLE_SEEDS:
    bols_list = []
    for rng in MULTIPLE_RNG_LIST:
        bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, 
                        CONF_THRES, length_scale, logdir)
        bols_list.append(bols)
        bols.initial_setup(bo_init_iters, rng)
    print("\nCompleted BOLevelSet initial setup")
else:
    bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, 
                        CONF_THRES, length_scale, logdir)
    bols.initial_setup(bo_init_iters, RNG)
    print("\nCompleted BOLevelSet initial setup")

if MULTIPLE_SEEDS:
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
    # Plot the initial GP overlaid with its 0-level set and the true BRT
    fig_name = bols.logdir + f'/gp_init.png'
    fig_name_colorbar = bols.logdir + f'/gp_init_colorbar.png'
    mile_name = bols.logdir + f'/mile_init.png'
    original_cand_len = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                                        fig_name_colorbar, mile_name)

########################### FIT THE ERROR GP ###########################
if not USE_MILE:
    if MULTIPLE_SEEDS:
        mean_function = None 
        logdir = 'error_gp_model_dir'
        error_gps_list = []
        for i in range(len(MULTIPLE_SEED_LIST)):
            bols = bols_list[i]
            seed = MULTIPLE_SEED_LIST[i]
            mu, var = bols.m.predict(calibration_points, full_cov=False)
            criterion = mu - BETA * np.sqrt(var) # Conservative bc more likely to say state is in BRT
            # errors = np.abs(criterion - costs_at_calibration_points)
            errors = []
            for i in range(len(criterion)):
                if (costs_at_calibration_points[i] == 0 and criterion[i] <= 0): # Product is 0
                    errors.append(0)
                elif criterion[i] * costs_at_calibration_points[i] > 0:  # They have the same sign
                    errors.append(0)
                else: # Even if costs_at_calibration_points[i] < 0 and criterion[i] == 0, say it's wrong
                    errors.append(1) 
            errors = np.expand_dims(np.array(errors), -1)
            error_gp = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, CONF_THRES, length_scale, logdir)
            error_gp.initial_setup_given_data(calibration_points, errors, to_plot=False)
            error_gps_list.append(error_gp)
            # Plot the initial error GP
            plot_error_gp(error_gp, candidates, oned_x, error_gp.logdir + f'/gp_init_colorbar{seed}.png')
    else:
        mu, var = bols.m.predict(calibration_points, full_cov=False)
        criterion = mu - BETA * np.sqrt(var) # Conservative bc more likely to say state is in BRT
        # errors = np.abs(criterion - costs_at_calibration_points)
        errors = []
        for i in range(len(criterion)):
            if (costs_at_calibration_points[i] == 0 and criterion[i] <= 0): # Product is 0
                errors.append(0)
            elif criterion[i] * costs_at_calibration_points[i] > 0:  # They have the same sign
                errors.append(0)
            else: # Even if costs_at_calibration_points[i] < 0 and criterion[i] == 0, say it's wrong
                errors.append(1) 
        errors = np.expand_dims(np.array(errors), -1)

        mean_function = None 
        logdir = 'error_gp_model_dir'
        error_gp = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, CONF_THRES, length_scale, logdir)
        error_gp.initial_setup_given_data(calibration_points, errors)

        # Plot the initial error GP
        plot_error_gp(error_gp, candidates, oned_x, error_gp.logdir + f'/gp_init_colorbar.png')

########################### RUN ACQUISITION FUNCTION ###########################
shaped_candidates = candidates.copy()
shaped_candidates = shaped_candidates.reshape((original_cand_len, original_cand_len, 2))
f = batched_rollouts_generator(all_values)
if TO_PICKLE_COUNTEREXAMPLE:
    search_candidates, true_costs = get_ground_truths_for_a_grid(f, discretization=1.0)
    with open('counterexample_search_data.pkl', 'wb') as f:
        pickle.dump([true_costs, search_candidates], f)
bo_iters = NUM_BO_ITERS
if MULTIPLE_SEEDS:
    assert len(bols_list) == len(error_gps_list)
    for i in range(len(bols_list)):
        bols = bols_list[i]
        error_gp = error_gps_list[i]
        seed = MULTIPLE_SEED_LIST[i]
        rng = MULTIPLE_RNG_LIST[i]
        if bo_iters != 0:
            if USE_MILE:
                bols.optimize_loop(original_cand_len, shaped_candidates, bo_iters, to_plot=False)
            else:
                # bols.optimize_loop_counterexamples(search_candidates, true_costs, BETA, 
                #                         rng, bo_iters, to_plot=False)
                # plt.figure()
                # plt.plot(range(bo_iters), bols.num_counterexamples_list)
                # plt.savefig(bols.logdir + f'/counterexs{seed}.png')
                bols.optimize_loop_error_gp(error_gp, original_cand_len, candidates, shaped_candidates, 
                                            calibration_points, costs_at_calibration_points,  
                                            BETA, rng, bo_iters, to_plot=False)
                # Plot the final error GP
                plot_error_gp(error_gp, candidates, oned_x, error_gp.logdir + f'/gp_final_colorbar{seed}.png')

        # Plot the final GP overlaid with its 0-level set and the true BRT
        fig_name = bols.logdir + f'/gp_final{seed}.png'
        fig_name_colorbar = bols.logdir + f'/gp_final_colorbar{seed}.png'
        mile_name = bols.logdir + f'/mile_final{seed}.png'
        _ = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                            fig_name_colorbar, mile_name)
else:
    if bo_iters != 0:
        if USE_MILE:
            bols.optimize_loop(original_cand_len, shaped_candidates, bo_iters, to_plot=False)
        else:
            # bols.optimize_loop_counterexamples(search_candidates, true_costs, BETA, 
            #                             RNG, bo_iters, to_plot=False)
            # plt.figure()
            # plt.plot(range(bo_iters), bols.num_counterexamples_list)
            # plt.savefig(bols.logdir + f'/counterexs.png')
            bols.optimize_loop_error_gp(error_gp, original_cand_len, candidates, shaped_candidates, 
                                        calibration_points, costs_at_calibration_points, 
                                        BETA, RNG, bo_iters, to_plot=False)
            # Plot the final error GP
            plot_error_gp(error_gp, candidates, oned_x, error_gp.logdir + f'/gp_final_colorbar.png')

    # Plot the final GP overlaid with its 0-level set and the true BRT
    fig_name = bols.logdir + f'/gp_final.png'
    fig_name_colorbar = bols.logdir + f'/gp_final_colorbar.png'
    mile_name = bols.logdir + f'/mile_final.png'
    _ = plot_main_gp(bols, grid, candidates, oned_x, fig_name, 
                        fig_name_colorbar, mile_name)

########################### VALIDATION OF FINAL GP ###########################
f = batched_rollouts_generator(all_values)
if TO_PICKLE_VALIDATION:
    true_costs, discretization = validation_get_ground_truths(f)
    with open('validation_data.pkl', 'wb') as f:
        pickle.dump([true_costs, discretization], f)
else:
    true_costs, discretization = validation_data

if MULTIPLE_SEEDS:
    results_dict = {}
    for i in range(len(bols_list)):
        bols = bols_list[i]
        seed = MULTIPLE_SEED_LIST[i]
        tpr, fpr, tnr, fnr = validate_final_level_set(bols, true_costs, discretization)
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
    tpr, fpr, tnr, fnr = validate_final_level_set(bols, true_costs, discretization)
    # TPR + FNR = 1
    # FPR + TNR = 1
    print("True Positive Rate: ", tpr)
    print("False Negative Rate: ", fnr)
    print("False Positive Rate: ", fpr)
    print("True Negative Rate: ", tnr)

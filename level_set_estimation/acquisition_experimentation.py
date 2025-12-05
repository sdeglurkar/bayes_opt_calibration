import GPy
import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from bolevelset import BOLevelSet

THETA_VALUE = 0.0
DUBINS_VELOCITY = 10.0
DT = 0.01
FINAL_TIME = -1.0
TRAJ_TIME_STEPS = int(np.abs(FINAL_TIME)/DT)
goal_R = 5
NUM_BO_INIT_ITERS = 10 
NUM_BO_ITERS = 30
SIZE_CALIBRATION_SET = 100

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

######### GET OPTIMAL VALUE FUNCTION -- BRT #########
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - goal_R

solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)


times = np.linspace(0, FINAL_TIME, 100) 
initial_values = values
all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values)


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

######### EVALUATE COSTS FOR CALIBRATION SET #########
f = batched_rollouts_generator(all_values)
range_x = [[-15, 15], [-15, 15]]
X_init = []
for i in range(2):
    X_i_init = np.random.uniform(
        range_x[i][0],
        range_x[i][1],
        (SIZE_CALIBRATION_SET,)
    )
    X_init.append(X_i_init)
calibration_points = np.stack(X_init, axis=1)
print("Evaluating costs for calibration set")
costs_at_calibration_points = f(calibration_points)
print("Done evaluating costs for calibration set!")

######### FIT GAUSSIAN PROCESS #########
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
conf_thres = 0.9
beta = norm.ppf(conf_thres)
length_scale = 0.25
logdir = 'acq_exp_model_dir'
bo_init_iters = NUM_BO_INIT_ITERS
bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, 
                    conf_thres, length_scale, logdir)
bols.initial_setup(bo_init_iters)
print("\nCompleted BOLevelSet initial setup")
# bo_iters = NUM_BO_ITERS
# if bo_iters != 0:
#     bols.optimize_loop(bo_iters)

# Plot the GP overlaid with its 0-level set and the true BRT
plt.figure()
bols.m.plot()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            all_values[-1, :, :, THETA_INDEX].T,
            levels=[0.0],
            colors="black",
            linewidths=3)
mu, var = bols.m.predict(candidates, full_cov=False)
criterion = mu + beta * np.sqrt(var)  
criterion = criterion.reshape(len(oned_x), len(oned_x))
plt.contour(oned_x,
            oned_x,
            criterion,
            levels=[0.0],
            colors="lightblue",
            linewidths=2)
criterion = mu - beta * np.sqrt(var)  
criterion = criterion.reshape(len(oned_x), len(oned_x))
plt.contour(oned_x,
            oned_x,
            criterion,
            levels=[0.0],
            colors="blue",
            linewidths=2)
plt.savefig(bols.logdir + f'/gp_final.png')
plt.figure()
x = bols.candidates[:, 0].flatten()
y = bols.candidates[:, 1].flatten()
length_of_candidates = len(x)
original_cand_len = int(np.sqrt(len(x)))
original_x = x.reshape((original_cand_len, original_cand_len))[0, :]
original_y = y.reshape((original_cand_len, original_cand_len))[:, 0]
z = bols.acq.reshape((original_cand_len, original_cand_len))
plt.contourf(original_x, original_y, BOLevelSet.normalize(z))
plt.colorbar()
plt.savefig(bols.logdir + f'/mile_final.png')


######### FIT THE ERROR GP #########
mu, var = bols.m.predict(calibration_points, full_cov=False)
criterion = mu - beta * np.sqrt(var) # Conservative bc more likely to say state is in BRT
errors = np.abs(criterion - costs_at_calibration_points)

mean_function = None 
logdir = 'error_gp_model_dir'
error_gp = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, length_scale, logdir)
error_gp.initial_setup_given_data(calibration_points, errors)

shaped_candidates = candidates.copy()
shaped_candidates = shaped_candidates.reshape((original_cand_len, original_cand_len, 2))

bo_iters = NUM_BO_ITERS
if bo_iters != 0:
    bols.optimize_loop_error_gp(error_gp, original_cand_len, candidates, shaped_candidates, 
                                calibration_points, costs_at_calibration_points, beta, bo_iters)

'''
# Error GP
error_mu, error_var = error_gp.m.predict(candidates, full_cov=False)
# max_acq_idx = np.unravel_index(np.argmax(error_mu), candidates.shape)
# x_next = candidates[max_acq_idx[0], :][np.newaxis, :]
# print(x_next)
sorted_indices = np.argsort(np.squeeze(error_mu))[::-1]
acq_idx = np.unravel_index(sorted_indices[0], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print("ERROR GP", x_next, error_mu[sorted_indices[0]])
acq_idx = np.unravel_index(sorted_indices[1], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print(x_next, error_mu[sorted_indices[1]])
acq_idx = np.unravel_index(sorted_indices[2], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print(x_next, error_mu[sorted_indices[2]])
acq_idx = np.unravel_index(sorted_indices[int(length_of_candidates/2)], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print(x_next, error_mu[sorted_indices[int(length_of_candidates/2)]])

# MILE
# max_acq_idx = np.unravel_index(np.argmax(bols.acq, axis=None), candidates.shape)
# x_next = candidates[max_acq_idx[0], :][np.newaxis, :]
# print(x_next)
sorted_indices = np.argsort(np.squeeze(bols.acq))[::-1]
acq_idx = np.unravel_index(sorted_indices[0], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print("MILE", x_next, bols.acq[sorted_indices[0]])
acq_idx = np.unravel_index(sorted_indices[1], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print(x_next, bols.acq[sorted_indices[1]])
acq_idx = np.unravel_index(sorted_indices[2], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print(x_next, bols.acq[sorted_indices[2]])
acq_idx = np.unravel_index(sorted_indices[int(length_of_candidates/2)], (original_cand_len, original_cand_len))
x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
print(x_next, bols.acq[sorted_indices[int(length_of_candidates/2)]])
'''



# Plot the GP overlaid with its 0-level set and the true BRT
plt.figure()
bols.m.plot()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            all_values[-1, :, :, THETA_INDEX].T,
            levels=[0.0],
            colors="black",
            linewidths=3)
mu, var = bols.m.predict(candidates, full_cov=False)
criterion = mu + beta * np.sqrt(var)  
criterion = criterion.reshape(len(oned_x), len(oned_x))
plt.contour(oned_x,
            oned_x,
            criterion,
            levels=[0.0],
            colors="lightblue",
            linewidths=2)
criterion = mu - beta * np.sqrt(var)  
criterion = criterion.reshape(len(oned_x), len(oned_x))
plt.contour(oned_x,
            oned_x,
            criterion,
            levels=[0.0],
            colors="blue",
            linewidths=2)
plt.savefig(bols.logdir + f'/gp_final.png')
plt.figure()
x = bols.candidates[:, 0].flatten()
y = bols.candidates[:, 1].flatten()
length_of_candidates = len(x)
original_cand_len = int(np.sqrt(len(x)))
original_x = x.reshape((original_cand_len, original_cand_len))[0, :]
original_y = y.reshape((original_cand_len, original_cand_len))[:, 0]
z = bols.acq.reshape((original_cand_len, original_cand_len))
plt.contourf(original_x, original_y, BOLevelSet.normalize(z))
plt.colorbar()
plt.savefig(bols.logdir + f'/mile_final.png')
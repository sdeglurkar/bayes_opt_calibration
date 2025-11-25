import GPy
import hj_reachability as hj
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

from bolevelset import BOLevelSet

THETA_INDEX = 0 #30
THETA_VALUE = 0.0
DUBINS_VELOCITY = 10.0
DT = 0.01
FINAL_TIME = -1.0
TRAJ_TIME_STEPS = int(np.abs(FINAL_TIME)/DT)
goal_R = 5
NUM_BO_INIT_ITERS = 100 #30
NUM_BO_ITERS = 100 #50

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


######### GET OPTIMAL VALUE FUNCTION -- BRT #########
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - goal_R

solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)


times = np.linspace(0, FINAL_TIME, 100) 
initial_values = values
all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values)

######### FIT GAUSSIAN PROCESS #########
mean_function = GPy.core.Mapping(2,1)
mean_function.f = lambda x: np.expand_dims(x[:, 0]**2 + x[:, 1]**2 - goal_R, -1)
mean_function.update_gradients = lambda a,b: 0
mean_function.gradients_X = lambda a,b: 0

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

f = batched_rollouts_generator(all_values)
input_dim = 2
oned_x = np.arange(-15, 15, 1.0)
xv, yv = np.meshgrid(oned_x, oned_x)
candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
range_x = [[-15, 15], [-15, 15]]
noise_var = 0.001 
cost_thres = 0.0 
conf_thres = 0.9
length_scale = 0.25
logdir = 'dubins_hj_model_dir'
bo_init_iters = NUM_BO_INIT_ITERS
bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, length_scale, logdir)
bols.initial_setup(bo_init_iters)
print("\nCompleted BOLevelSet initial setup")
bo_iters = NUM_BO_ITERS
if bo_iters != 0:
    bols.optimize_loop(bo_iters)
# level_set = bols.extract_levelset(candidates)
# print("Level Set\n", level_set)
# plt.figure()
# plt.scatter([elem[0] for elem in level_set], [0*elem[0] for elem in level_set])
# plt.savefig(logdir + f'/level_set.png')


# Plot the GP overlaid with the 0-level set and the true BRT
plt.figure()
bols.m.plot()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            all_values[-1, :, :, THETA_INDEX].T,
            levels=[0.0],
            colors="black",
            linewidths=3)
beta = norm.ppf(bols.conf_thres)
mu, var = bols.m.predict(candidates, full_cov=False)
criterion = mu - beta * np.sqrt(var)
criterion = criterion.reshape(len(oned_x), len(oned_x))
plt.contour(oned_x,
            oned_x,
            criterion,
            levels=[0.0],
            colors="blue",
            linewidths=3)
plt.savefig(bols.logdir + f'/gp_final.png')
plt.figure()
x = bols.candidates[:, 0].flatten()
y = bols.candidates[:, 1].flatten()
original_cand_len = int(np.sqrt(len(x))) 
original_x = x.reshape((original_cand_len, original_cand_len))[0, :]
original_y = y.reshape((original_cand_len, original_cand_len))[:, 0]
z = bols.acq.reshape((original_cand_len, original_cand_len))
plt.contourf(original_x, original_y, BOLevelSet.normalize(z))
plt.colorbar()
plt.savefig(bols.logdir + f'/mile_final.png')



######### PLOTTING - BRT #########
if False:
    vmin, vmax = all_values.min(), all_values.max()
    levels = np.linspace(round(vmin), round(vmax), round(vmax) - round(vmin) + 1)

    def render_frame(i, colorbar=False, traj=None):
        fig = plt.figure(figsize=(13, 8))
        plt.contourf(grid.coordinate_vectors[0],
                    grid.coordinate_vectors[1],
                    all_values[i, :, :, THETA_INDEX].T,
                    vmin=vmin,
                    vmax=vmax,
                    levels=levels)
        if colorbar:
            plt.colorbar()
        # plt.contour(grid.coordinate_vectors[0],
        #             grid.coordinate_vectors[1],
        #             target_values[:, :, THETA_INDEX].T,
        #             levels=0,
        #             colors="black",
        #             linewidths=3)
        plt.contour(grid.coordinate_vectors[0],
                    grid.coordinate_vectors[1],
                    all_values[0, :, :, THETA_INDEX].T,
                    levels=0,
                    colors="black",
                    linewidths=3)
        plt.contour(grid.coordinate_vectors[0],
                    grid.coordinate_vectors[1],
                    all_values[i, :, :, THETA_INDEX].T,
                    levels=0,
                    colors="black",
                    linewidths=3)
        
        if traj is not None:
            plt.scatter(traj[:, 0], traj[:, 1], c='red')


    render_frame(-1, True)
    render_frame(70, True)
    render_frame(50, True)
    render_frame(20, True)
    render_frame(10, True)
    render_frame(0, True)
    plt.show()











# time = 0.
# target_time = -10 #-2.8
# target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)

# plt.jet()
# plt.figure(figsize=(13, 8))
# plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values[:, :, THETA_INDEX].T)
# plt.colorbar()
# plt.contour(grid.coordinate_vectors[0],
#             grid.coordinate_vectors[1],
#             target_values[:, :, THETA_INDEX].T,
#             levels=0,
#             colors="black",
#             linewidths=3)
# plt.show()

# fig = go.Figure(data=go.Isosurface(x=grid.states[..., 0].ravel(),
#                              y=grid.states[..., 1].ravel(),
#                              z=grid.states[..., 2].ravel(),
#                              value=target_values.ravel(),
#                              colorscale="jet",
#                              isomin=0,
#                              surface_count=1,
#                              isomax=0))
# fig.show()




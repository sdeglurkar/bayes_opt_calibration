# export PYTHONPATH=/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration:$PYTHONPATH

import GPy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from bolevelset import BOLevelSet
from dynamics import dynamics
from scipy.io import loadmat

# mat_contents = loadmat('runs/dubins3dexpertavoidtube/ground_truth.mat')
# print(mat_contents['data'].shape)
# # print(mat_contents['g'].shape)
# print(mat_contents['tau'])
# mat_contents = loadmat('runs/dubins3dexpertreachtube/ground_truth.mat')
# print(mat_contents['data'].shape)
# # print(mat_contents['g'].shape)
# print(mat_contents['tau'])

DT = 0.001
DELTA = 0.01
TIME_STEPS = 10000
SYSTEM_TYPE = 'DOUBLE_INT'


goalR = 1.0
velocity = 1.0
omega_max = 1.0
angle_alpha_factor = 1.0
set_mode = 'avoid'  
diff_model = True  # Doesn't matter
freeze_model = False # Doesn't matter - to avoid a NotImplementedError
max_accel = 1.0
# system = dynamics.Dubins3D(goalR, velocity, omega_max, angle_alpha_factor, set_mode, diff_model, freeze_model)
system = dynamics.DoubleIntegrator(goalR, max_accel, set_mode, diff_model, freeze_model)


def get_dvds(value_fn, state, delta=DELTA):
    state = np.array([state])
    grad = []
    for i in range(state.shape[1]):
        direction = np.zeros(state.shape[1])
        direction[i] = 1.0
        delta_state = state + direction*delta
        partial_deriv = (value_fn(delta_state) - value_fn(state))/delta 
        grad.append(partial_deriv)
    grad = torch.squeeze(torch.Tensor(grad), -1)
    grad = torch.transpose(grad, 0, 1)
    return grad

def batched_rollouts_generator(value_fn, system=system, time_steps=TIME_STEPS, dt=DT):
    def rollout(state, plot_traj=True):
        state = torch.Tensor(state) 
        state_traj = [state]
        curr_state = state
        for t in range(time_steps):
            # dvds = get_dvds(value_fn, curr_state)
            # ctrl = system.optimal_control(curr_state, dvds)
            ctrl = system.random_control()
            dsdt = system.dsdt(curr_state, ctrl, disturbance=None)
            next_state = system.equivalent_wrapped_state(curr_state + dt*dsdt)
            state_traj.append(next_state)
            curr_state = next_state
        state_traj = torch.Tensor(np.array(state_traj))

        cost = system.cost_fn(state_traj)
        print("State: ", state, " Cost: ", cost)
        print("\n")
        length_of_traj = state_traj.shape[0]
        print("State traj: ", state_traj[2, :], state_traj[int(length_of_traj/3), :], \
                state_traj[int(length_of_traj/2), :], state_traj[int(2*length_of_traj/3), :], \
                state_traj[-1, :])
        print("\n")

        if plot_traj:
            if SYSTEM_TYPE == 'DOUBLE_INT':
                fig, ax = plt.subplots()
                ax.add_patch(matplotlib.patches.Rectangle((-1, -3), 2, 6, fill=False))
                ax.scatter(state_traj[:, 0], state_traj[:, 1])
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel("Position")
                ax.set_ylabel("Velocity")
                plt.show()
            elif SYSTEM_TYPE == 'DUBINS':
                fig, ax = plt.subplots()
                circle = plt.Circle((0, 0), goalR, color='r', fill=False)
                ax.scatter(state_traj[:, 0], state_traj[:, 1])
                ax.add_patch(circle)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                plt.show()
            else:
                raise NotImplementedError

        return cost
    
    def batched_rollouts(states):
        costs = []
        for state in states:
            if SYSTEM_TYPE == 'DOUBLE_INT':
                state = np.append(state, 1.0)
            cost = np.array(rollout(state))
            costs.append(cost)
        print("Done running rollouts!")
        return np.expand_dims(np.array(costs), -1)
    
    return batched_rollouts


# mean_function = GPy.core.Mapping(3,1)
# mean_function = GPy.core.Mapping(2,1)
mean_function = GPy.core.Mapping(1,1)
# mean_function.f = lambda x: np.expand_dims(x[:, 0]**2 + x[:, 1]**2 - 1, -1) # lambda x: np.expand_dims(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2 - 1, -1)
mean_function.f = lambda x: np.expand_dims(x[:, 0]**2 - 1, -1)
mean_function.update_gradients = lambda a,b: 0
mean_function.gradients_X = lambda a,b: 0
value_fn = mean_function.f


f = batched_rollouts_generator(value_fn) 
input_dim = 1 #3
# oned_x = np.arange(-5, 5, 1)
oned_x = np.arange(-5, 5, 0.1)
theta_grid = np.arange(-np.pi, np.pi, 1.0)
# xv, yv, thetav = np.meshgrid(oned_x, oned_x, theta_grid)
# candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1), thetav.reshape(-1, 1)))
# xv, yv = np.meshgrid(oned_x, oned_x)
xv = np.meshgrid(oned_x)[0]
# candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
candidates = xv.reshape(-1, 1)

# range_x = [[-5, 5], [-5, 5], [-np.pi, np.pi]]
# range_x = [[-5, 5], [-5, 5]]
range_x = [[-5, 5]]
noise_var = 0.001 #1e-15
cost_thres = 0.0 
conf_thres = 0.9
length_scale = 0.25
logdir = 'dblint_model_dir'
bo_init_iters = 20
bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, length_scale, logdir)
bols.initial_setup(bo_init_iters)
print("\nCompleted BOLevelSet initial setup")
bo_iters = 150
bols.optimize_loop(bo_iters)

# set_theta_grid = theta_grid * 0
# xv, yv, thetav = np.meshgrid(oned_x, oned_x, set_theta_grid)
# candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1), thetav.reshape(-1, 1)))
level_set = bols.extract_levelset(candidates)
# print("Level Set\n", bols.extract_levelset(candidates))
print("Level Set\n", level_set)
plt.figure()
plt.scatter([elem[0] for elem in level_set], [0*elem[0] for elem in level_set])
plt.savefig(logdir + f'/level_set.png')

# def new_value_fn_generator(bols):
#     def new_value_fn(state):
#         mu, cov = bols.m.predict(state, full_cov=True, include_likelihood=True)
#         return mu
#     return new_value_fn

# f = batched_rollouts_generator(new_value_fn_generator(bols)) 
# bo_init_iters = 10
# bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, length_scale, logdir)
# bols.initial_setup(bo_init_iters)
# print("\nCompleted BOLevelSet initial setup")


print("\nYAY!")
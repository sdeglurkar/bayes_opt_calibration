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

dt = 0.001


goalR = 1.0
velocity = 1.0
omega_max = 1.0
angle_alpha_factor = 1.0
set_mode = 'avoid'
diff_model = True  # Doesn't matter
freeze_model = False # Doesn't matter - to avoid a NotImplementedError
system = dynamics.Dubins3D(goalR, velocity, omega_max, angle_alpha_factor, set_mode, diff_model, freeze_model)

# print("\n")
# state = torch.Tensor([-1.0, 0.5, 0.0])
# dvds = torch.Tensor([1.0, 1.0, 1.0])
# ctrl = system.optimal_control(state, dvds)
# print(ctrl)
# dsdt = system.dsdt(state, ctrl, disturbance=None)
# print(dsdt)

# state_traj = [state]
# curr_state = state
# time_steps = 1000
# for t in range(time_steps):
#     # dvds = torch.Tensor([1.0, 1.0, 1.0])
#     ctrl = system.optimal_control(curr_state, dvds)
#     dsdt = system.dsdt(curr_state, ctrl, disturbance=None)
#     next_state = system.equivalent_wrapped_state(curr_state + dt*dsdt)
#     state_traj.append(next_state)
#     curr_state = next_state
# state_traj = torch.Tensor(np.array(state_traj))

# fig, ax = plt.subplots()
# circle = plt.Circle((0, 0), goalR, color='r', fill=False)
# ax.scatter(state_traj[:, 0], state_traj[:, 1])
# ax.add_patch(circle)
# ax.set_aspect('equal', adjustable='box')
# plt.show()

# print(system.cost_fn(state_traj))


def get_dvds(value_fn, state, delta=0.01):
    state = np.array([state[:2]])
    grad = []
    for i in range(len(state[0])):
        direction = np.zeros(len(state[0]))
        direction[i] = 1.0
        delta_state = state + direction*delta
        partial_deriv = (value_fn(delta_state) - value_fn(state))/delta 
        grad.append(partial_deriv)
    # For theta
    grad.append(0.0)
    return torch.Tensor(grad)

def batched_rollouts_generator(value_fn, system=system, theta=0.0, time_steps=10000, dt=dt):
    def rollout(state, plot_traj=True):
        state = list(state)
        state.append(theta)
        state = torch.Tensor(state) 
        state_traj = [state]
        curr_state = state
        for t in range(time_steps):
            # dvds = torch.Tensor([1.0, 1.0, 1.0])
            dvds = get_dvds(value_fn, state)
            ctrl = system.optimal_control(curr_state, dvds)
            dsdt = system.dsdt(curr_state, ctrl, disturbance=None)
            next_state = system.equivalent_wrapped_state(curr_state + dt*dsdt)
            state_traj.append(next_state)
            curr_state = next_state
        state_traj = torch.Tensor(np.array(state_traj))

        if plot_traj:
            fig, ax = plt.subplots()
            circle = plt.Circle((0, 0), goalR, color='r', fill=False)
            ax.scatter(state_traj[:, 0], state_traj[:, 1])
            ax.add_patch(circle)
            ax.set_aspect('equal', adjustable='box')
            plt.show()

        cost = system.cost_fn(state_traj)
        print(state, cost)
        return cost
    
    def batched_rollouts(states):
        costs = []
        for state in states:
            cost = np.array(rollout(state))
            costs.append(cost)
        print("Done running rollouts!")
        return np.expand_dims(np.array(costs), -1)
    
    return batched_rollouts


mean_function = GPy.core.Mapping(2,1)
mean_function.f = lambda x: x[0][0]**2 + x[0][1]**2 - 1
mean_function.update_gradients = lambda a,b: 0
mean_function.gradients_X = lambda a,b: 0
value_fn = mean_function.f

f = batched_rollouts_generator(value_fn) 
input_dim = 2
oned_x = np.arange(-5, 5, 1)
xv, yv = np.meshgrid(oned_x, oned_x)
candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
range_x = [[-5, 5], [-5, 5]]
noise_var = 1e-15
cost_thres = 0.0 
conf_thres = 0.9
length_scale = 0.25
logdir = 'model_dir'
bo_init_iters = 10
bols = BOLevelSet(f, mean_function, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, length_scale, logdir)
bols.initial_setup(bo_init_iters)
print("\nCompleted BOLevelSet initial setup")
# bols.optimize_loop(bo_iters)

print(bols.extract_levelset(candidates))

mu, cov = bols.m.predict(candidates, full_cov=True, include_likelihood=True)
value_fn = mu



print("\nYAY!")
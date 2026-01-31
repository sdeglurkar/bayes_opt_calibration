import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script import env_utils
from env_utils import NoResetSyncVectorEnv, find_a_batch

from main_model import MainGP

import gymnasium as gym
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def batched_rollouts_generator(horizon, policy, args):
    def batched_rollouts(states):
        """
        Measure the reach-avoid performance of the system given the initial 
        states. This is a vectorized version for efficiency.
        """
        num_samples = states.shape[0]
        reach_avoid_measures = np.zeros(num_samples)

        envs = NoResetSyncVectorEnv([lambda: gym.make(args.task) for _ in range(num_samples)])
        n_dim = envs.envs[0].observation_space.shape[0]

        for i, state in enumerate(states):
            envs.envs[i].reset(options={'initial_state': state})

        rewards = np.zeros((num_samples, horizon))
        constraints = np.zeros((num_samples, horizon))
        env_states = np.array([env.state for env in envs.envs])
        state_trajs = np.zeros((num_samples, n_dim, horizon+1))
        state_trajs[:, :, 0] = env_states

        for t in range(horizon):
            current_states = np.array([env.state for env in envs.envs])
            acts = find_a_batch(current_states, policy)
            actions = np.concatenate((acts[:, :3], np.zeros((num_samples, 3))), axis=1)  # assuming no noise in action for now
            st, rew, done, _, info = envs.step(actions)
            state_trajs[:, :, t+1] = st
            rewards[:, t] = rew
            constraints[:, t] = info['constraint']

        min_constraints = np.minimum.accumulate(constraints, axis=1)
        reach_avoid_measures = np.max(np.minimum(rewards, min_constraints), axis=1)

        print("Rollouts", state_trajs)

        return reach_avoid_measures, state_trajs

    return batched_rollouts

def plot_main_gp(model, grid, candidates, oned_x, oned_y, value_fn, theta_index, 
                    beta, fig_name, fig_name_colorbar, mile_name):
    plt.figure()
    # model.m.plot()
    plt.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                value_fn[-1, :, :, theta_index].T,
                levels=[0.0],
                colors="black",
                linewidths=3)
    mu, var = model.m.predict(candidates, full_cov=False)
    criterion = mu + beta * np.sqrt(var)  
    criterion = criterion.reshape(len(oned_x), len(oned_y))
    plt.contour(oned_x,
                oned_x,
                criterion,
                levels=[0.0],
                colors="lightblue",
                linewidths=2)
    criterion = mu - beta * np.sqrt(var)  
    criterion = criterion.reshape(len(oned_x), len(oned_y))
    plt.contour(oned_x,
                oned_x,
                criterion,
                levels=[0.0],
                colors="blue",
                linewidths=2)
    if model.acq_cache.size > 0:
        acq_points = np.array(model.acq_cache)
        plt.scatter(acq_points[:, 0], acq_points[:, 1], color='g')
    plt.savefig(fig_name)
    plt.figure()
    plt.contourf(oned_x,
                oned_y,
                criterion)
    plt.colorbar()
    plt.savefig(fig_name_colorbar)
    if model.do_MILE:
        plt.figure()
        x = model.candidates[:, 0].flatten()
        y = model.candidates[:, 1].flatten()
        original_cand_len = int(np.sqrt(len(x)))
        original_x = x.reshape((original_cand_len, original_cand_len))[0, :]
        original_y = y.reshape((original_cand_len, original_cand_len))[:, 0]
        z = model.acq.reshape((original_cand_len, original_cand_len))
        plt.contourf(original_x, original_y, MainGP.normalize(z))
        plt.colorbar()
        plt.savefig(mile_name)

def get_ground_truths_for_a_grid(f, range_x, discretization):
    print("\nRunning get_ground_truths_for_a_grid!") 
    oned_x = np.arange(range_x[0][0], range_x[0][1], discretization) 
    oned_y = np.arange(range_x[1][0], range_x[1][1], discretization) 
    xv, yv = np.meshgrid(oned_x, oned_y)
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

def validate_final_level_set(model, candidates, true_costs, beta):
    print("Running validation of final GP level set")
    mu, var = model.m.predict(candidates, full_cov=False)
    criterion = mu - beta * np.sqrt(var) # Conservative bc more likely to say state is in BRT
    
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

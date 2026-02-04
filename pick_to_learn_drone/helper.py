import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
# from experiment_script import env_utils
from experiment_script.env_utils import NoResetSyncVectorEnv, find_a_batch, evaluate_V

from main_model import MainGP

import gymnasium as gym
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def batched_rollouts_generator(horizon, policy, args):
    def batched_rollouts(states):
        """
        Measure the reach-avoid performance of the system given the initial 
        states. This is a vectorized version for efficiency.
        """
        print("\nRunning batched_rollouts")
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

        return np.expand_dims(reach_avoid_measures, -1)

    return batched_rollouts

def expand_state_based_on_model_dim(ego_setting, adversary_setting,
                                        model_dim):
    def state_expander(states):
        ego_vx, ego_vy, ego_z, ego_vz = ego_setting
        ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz = adversary_setting

        # Assumes states are each dim (1, n)
        expanded_states = []
        for state in states:
            expanded_state = list(state)

            if model_dim == 2:
                expanded_state.insert(1, ego_vx)
                expanded_state.extend([ego_vy, ego_z, ego_vz, ad_x, ad_vx, ad_y, \
                                        ad_vy, ad_z, ad_vz])
            elif model_dim == 3:
                expanded_state.insert(1, ego_vx)
                expanded_state.insert(3, ego_vy)
                expanded_state.extend([ego_vz, ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz])
            elif model_dim == 4:
                expanded_state.insert(3, ego_vy)
                expanded_state.extend([ego_vz, ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz])
            elif model_dim == 6:
                expanded_state.extend([ad_vx, ad_y, ad_vy, ad_z, ad_vz])
            elif model_dim == 12:
                pass 
            else:
                raise NotImplementedError

            expanded_states.append(expanded_state)

        return np.array(expanded_states)
    
    return state_expander

def get_ground_truths_for_random_points(range_x, ego_setting, adversary_setting, 
                                        rng, f, size_set, model_dim):
    print("\nRunning get_ground_truths_for_random_points!") 
    ego_vx, ego_vy, ego_z, ego_vz = ego_setting
    ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz = adversary_setting

    # If model dim is 2
    x01 = rng.uniform(range_x[0][0], range_x[0][1], size=(size_set, 1))
    ego_vx1 = np.full((size_set, 1), ego_vx)
    y01 = rng.uniform(range_x[2][0], range_x[2][1], size=(size_set, 1))
    ego_vy1 = np.full((size_set, 1), ego_vy)
    z01 = np.full((size_set, 1), ego_z)
    ego_vz1 = np.full((size_set, 1), ego_vz)

    ad_x1 = np.full((size_set, 1), ad_x)
    ad_vx1 = np.full((size_set, 1), ad_vx)
    ad_y1 = np.full((size_set, 1), ad_y)
    ad_vy1 = np.full((size_set, 1), ad_vy)
    ad_z1 = np.full((size_set, 1), ad_z)
    ad_vz1 = np.full((size_set, 1), ad_vz)

    inds = [0, 2]

    if model_dim == 2: pass    
    elif model_dim == 3:
        z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(size_set, 1)) 
        inds = [0, 2, 4]   
    elif model_dim == 4:
        ego_vx1 = rng.uniform(range_x[1][0], range_x[1][1], size=(size_set, 1))
        z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(size_set, 1))
        inds = [0, 1, 2, 4]
    elif model_dim == 6:
        ego_vx1 = rng.uniform(range_x[1][0], range_x[1][1], size=(size_set, 1))
        ego_vy1 = rng.uniform(range_x[3][0], range_x[3][1], size=(size_set, 1))
        z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(size_set, 1))
        ego_vz1 = rng.uniform(range_x[5][0], range_x[5][1], size=(size_set, 1))
        inds = [0, 1, 2, 3, 4, 5]
    elif model_dim == 12:
        ego_vx1 = rng.uniform(range_x[1][0], range_x[1][1], size=(size_set, 1))
        ego_vy1 = rng.uniform(range_x[3][0], range_x[3][1], size=(size_set, 1))
        z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(size_set, 1))
        ego_vz1 = rng.uniform(range_x[5][0], range_x[5][1], size=(size_set, 1))
        ad_x1 = rng.uniform(range_x[6][0], range_x[6][1], size=(size_set, 1))
        ad_vx1 = rng.uniform(range_x[7][0], range_x[7][1], size=(size_set, 1))
        ad_y1 = rng.uniform(range_x[8][0], range_x[8][1], size=(size_set, 1))
        ad_vy1 = rng.uniform(range_x[9][0], range_x[9][1], size=(size_set, 1))
        ad_z1 = rng.uniform(range_x[10][0], range_x[10][1], size=(size_set, 1))
        ad_vz1 = rng.uniform(range_x[11][0], range_x[11][1], size=(size_set, 1))
        inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    else:
        raise NotImplementedError
    
    random_points = np.hstack((x01, ego_vx1,
                            y01, ego_vy1,
                            z01, ego_vz1,
                            ad_x1, ad_vx1,
                            ad_y1, ad_vy1,
                            ad_z1, ad_vz1))

    costs_at_random_points = f(random_points)
    print("Done!")
    return random_points[:, inds], costs_at_random_points, random_points

def get_ground_truths_for_a_grid(range_x, ego_setting, adversary_setting, 
                                    f, discretization, model_dim, policy,
                                    get_costs=True,
                                    get_learned_V=False):
    print("\nRunning get_ground_truths_for_a_grid!")
    ego_vx, ego_vy, ego_z, ego_vz = ego_setting
    ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz = adversary_setting

    learned_V = None 

    if get_learned_V: assert model_dim == 2

    if model_dim == 2:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        xv_orig, yv_orig = np.meshgrid(oned_x, oned_y)
        learned_V = np.zeros(xv_orig.shape)
        xv = xv_orig.reshape(-1, 1)
        yv = yv_orig.reshape(-1, 1)

        size_set = len(xv)
        ego_vx1 = np.full((size_set, 1), ego_vx)
        ego_vy1 = np.full((size_set, 1), ego_vy)
        z01 = np.full((size_set, 1), ego_z)
        ego_vz1 = np.full((size_set, 1), ego_vz)
        
        ad_x1 = np.full((size_set, 1), ad_x)
        ad_vx1 = np.full((size_set, 1), ad_vx)
        ad_y1 = np.full((size_set, 1), ad_y)
        ad_vy1 = np.full((size_set, 1), ad_vy)
        ad_z1 = np.full((size_set, 1), ad_z)
        ad_vz1 = np.full((size_set, 1), ad_vz)

        inds = [0, 2]

        for ii in tqdm(range(learned_V.shape[0])):
            for jj in range(learned_V.shape[1]):
                tmp_point = [
                    xv_orig[ii,jj], ego_vx,
                    yv_orig[ii,jj], ego_vy,
                    ego_z, ego_vz,
                    ad_x,  ad_vx,
                    ad_y, ad_vy,
                    ad_z, ad_vz
                ]
                learned_V[ii,jj] = evaluate_V(tmp_point, policy)
    elif model_dim == 3:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        oned_z = np.arange(range_x[4][0], range_x[4][1], discretization)
        xv, yv, zv = np.meshgrid(oned_x, oned_y, oned_z)
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        z01 = zv.reshape(-1, 1)

        size_set = len(xv)
        ego_vx1 = np.full((size_set, 1), ego_vx)
        ego_vy1 = np.full((size_set, 1), ego_vy)
        ego_vz1 = np.full((size_set, 1), ego_vz)
        
        ad_x1 = np.full((size_set, 1), ad_x)
        ad_vx1 = np.full((size_set, 1), ad_vx)
        ad_y1 = np.full((size_set, 1), ad_y)
        ad_vy1 = np.full((size_set, 1), ad_vy)
        ad_z1 = np.full((size_set, 1), ad_z)
        ad_vz1 = np.full((size_set, 1), ad_vz) 

        inds = [0, 2, 4]  
    elif model_dim == 4:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        oned_z = np.arange(range_x[4][0], range_x[4][1], discretization)
        oned_vx = np.arange(range_x[1][0], range_x[1][1], discretization)
        xv, yv, zv, vxv = np.meshgrid(oned_x, oned_y, oned_z, oned_vx)
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        z01 = zv.reshape(-1, 1)
        ego_vx1 = vxv.reshape(-1, 1)

        size_set = len(xv)
        ego_vy1 = np.full((size_set, 1), ego_vy)
        ego_vz1 = np.full((size_set, 1), ego_vz)
        
        ad_x1 = np.full((size_set, 1), ad_x)
        ad_vx1 = np.full((size_set, 1), ad_vx)
        ad_y1 = np.full((size_set, 1), ad_y)
        ad_vy1 = np.full((size_set, 1), ad_vy)
        ad_z1 = np.full((size_set, 1), ad_z)
        ad_vz1 = np.full((size_set, 1), ad_vz)

        inds = [0, 1, 2, 4]
    elif model_dim == 6:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        oned_z = np.arange(range_x[4][0], range_x[4][1], discretization)
        oned_vx = np.arange(range_x[1][0], range_x[1][1], discretization)
        oned_vy = np.arange(range_x[3][0], range_x[3][1], discretization)
        oned_vz = np.arange(range_x[5][0], range_x[5][1], discretization)
        xv, yv, zv, vxv, vyv, vzv = np.meshgrid(oned_x, oned_y, oned_z, oned_vx,
                                                oned_vy, oned_vz)
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        z01 = zv.reshape(-1, 1)
        ego_vx1 = vxv.reshape(-1, 1)
        ego_vy1 = vyv.reshape(-1, 1)
        ego_vz1 = vzv.reshape(-1, 1)

        size_set = len(xv)

        ad_x1 = np.full((size_set, 1), ad_x)
        ad_vx1 = np.full((size_set, 1), ad_vx)
        ad_y1 = np.full((size_set, 1), ad_y)
        ad_vy1 = np.full((size_set, 1), ad_vy)
        ad_z1 = np.full((size_set, 1), ad_z)
        ad_vz1 = np.full((size_set, 1), ad_vz)

        inds = [0, 1, 2, 3, 4, 5]
    elif model_dim == 12:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        oned_z = np.arange(range_x[4][0], range_x[4][1], discretization)
        oned_vx = np.arange(range_x[1][0], range_x[1][1], discretization)
        oned_vy = np.arange(range_x[3][0], range_x[3][1], discretization)
        oned_vz = np.arange(range_x[5][0], range_x[5][1], discretization)

        oned_adx = np.arange(range_x[6][0], range_x[6][1], discretization)
        oned_ady = np.arange(range_x[8][0], range_x[8][1], discretization)
        oned_adz = np.arange(range_x[10][0], range_x[10][1], discretization)
        oned_advx = np.arange(range_x[7][0], range_x[7][1], discretization)
        oned_advy = np.arange(range_x[9][0], range_x[9][1], discretization)
        oned_advz = np.arange(range_x[11][0], range_x[11][1], discretization)
        
        xv, yv, zv, vxv, vyv, vzv, adxv, adyv, adzv, advxv, advyv, advzv = \
            np.meshgrid(oned_x, oned_y, oned_z, oned_vx, oned_vy, oned_vz,
                        oned_adx, oned_ady, oned_adz, oned_advx, oned_advy, oned_advz)
        
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        z01 = zv.reshape(-1, 1)
        ego_vx1 = vxv.reshape(-1, 1)
        ego_vy1 = vyv.reshape(-1, 1)
        ego_vz1 = vzv.reshape(-1, 1)
        ad_x1 = adxv.reshape(-1, 1)
        ad_y1 = adyv.reshape(-1, 1)
        ad_z1 = adzv.reshape(-1, 1)
        ad_vx1 = advxv.reshape(-1, 1)
        ad_vy1 = advyv.reshape(-1, 1)
        ad_xz1 = advzv.reshape(-1, 1)

        inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    else:
        raise NotImplementedError

    candidates = np.hstack((xv, ego_vx1,
                            yv, ego_vy1,
                            z01, ego_vz1,
                            ad_x1, ad_vx1,
                            ad_y1, ad_vy1,
                            ad_z1, ad_vz1))
    if get_costs: 
        costs = f(candidates)
    else:
        costs = None
    print("Done!")
    return candidates[:, inds], costs, oned_x, oned_y, candidates, learned_V

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

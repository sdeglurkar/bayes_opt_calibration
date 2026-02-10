import sys

from Lipschitz_Continuous_Reachability_Learning.experiment_script.env_utils import evaluate_V_batch 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import NoResetSyncVectorEnv, find_a_batch, evaluate_V, \
                                        evaluate_V_batch

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

        # print("Rollouts", state_trajs)

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
                expanded_state.extend([ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz])
            elif model_dim == 12:
                pass 
            else:
                raise NotImplementedError

            expanded_states.append(expanded_state)

        return np.array(expanded_states)
    
    return state_expander

def get_ground_truths_for_random_points(range_x, ego_setting, adversary_setting, 
                                        rng, f, size_set, model_dim, get_costs=True):
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

    if get_costs:
        # costs_at_random_points = np.expand_dims(f(random_points), -1)
        costs_at_random_points = f(random_points)
    else:
        costs_at_random_points = None
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

    if model_dim == 2:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        xv_orig, yv_orig = np.meshgrid(oned_x, oned_y)
        if get_learned_V: learned_V = np.zeros(xv_orig.shape)
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

        if get_learned_V:
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
        xv_orig, yv_orig, zv_orig = np.meshgrid(oned_x, oned_y, oned_z)
        if get_learned_V: learned_V = np.zeros(xv_orig.shape)
        xv = xv_orig.reshape(-1, 1)
        yv = yv_orig.reshape(-1, 1)
        z01 = zv_orig.reshape(-1, 1)

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

        if get_learned_V:
            for ii in tqdm(range(learned_V.shape[0])):
                for jj in range(learned_V.shape[1]):
                    for kk in range(learned_V.shape[2]):
                        tmp_point = [
                            xv_orig[ii,jj,kk], ego_vx,
                            yv_orig[ii,jj,kk], ego_vy,
                            zv_orig[ii,jj,kk], ego_vz,
                            ad_x,  ad_vx,
                            ad_y, ad_vy,
                            ad_z, ad_vz
                        ]
                        learned_V[ii,jj,kk] = evaluate_V(tmp_point, policy) 
    elif model_dim == 4:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        oned_z = np.arange(range_x[4][0], range_x[4][1], discretization)
        oned_vx = np.arange(range_x[1][0], range_x[1][1], discretization)
        xv_orig, yv_orig, zv_orig, vxv_orig = np.meshgrid(oned_x, oned_y, oned_z, oned_vx)
        if get_learned_V: learned_V = np.zeros(xv_orig.shape)
        xv = xv_orig.reshape(-1, 1)
        yv = yv_orig.reshape(-1, 1)
        z01 = zv_orig.reshape(-1, 1)
        ego_vx1 = vxv_orig.reshape(-1, 1)

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

        if get_learned_V:
            for ii in tqdm(range(learned_V.shape[0])):
                for jj in range(learned_V.shape[1]):
                    for kk in range(learned_V.shape[2]):
                        for ll in range(learned_V.shape[3]):
                            tmp_point = [
                                xv_orig[ii,jj,kk,ll], vxv_orig[ii,jj,kk,ll],
                                yv_orig[ii,jj,kk,ll], ego_vy,
                                zv_orig[ii,jj,kk,ll], ego_vz,
                                ad_x,  ad_vx,
                                ad_y, ad_vy,
                                ad_z, ad_vz
                            ]
                            learned_V[ii,jj,kk,ll] = evaluate_V(tmp_point, policy) 
    elif model_dim == 6:
        oned_x = np.arange(range_x[0][0], range_x[0][1], discretization)
        oned_y = np.arange(range_x[2][0], range_x[2][1], discretization)
        oned_z = np.arange(range_x[4][0], range_x[4][1], discretization)
        oned_vx = np.arange(range_x[1][0], range_x[1][1], discretization)
        oned_vy = np.arange(range_x[3][0], range_x[3][1], discretization)
        oned_vz = np.arange(range_x[5][0], range_x[5][1], discretization)
        xv_orig, yv_orig, zv_orig, vxv_orig, vyv_orig, vzv_orig = \
            np.meshgrid(oned_x, oned_y, oned_z, oned_vx, oned_vy, oned_vz)
        if get_learned_V: learned_V = np.zeros(xv_orig.shape)
        xv = xv_orig.reshape(-1, 1)
        yv = yv_orig.reshape(-1, 1)
        z01 = zv_orig.reshape(-1, 1)
        ego_vx1 = vxv_orig.reshape(-1, 1)
        ego_vy1 = vyv_orig.reshape(-1, 1)
        ego_vz1 = vzv_orig.reshape(-1, 1)

        size_set = len(xv)

        ad_x1 = np.full((size_set, 1), ad_x)
        ad_vx1 = np.full((size_set, 1), ad_vx)
        ad_y1 = np.full((size_set, 1), ad_y)
        ad_vy1 = np.full((size_set, 1), ad_vy)
        ad_z1 = np.full((size_set, 1), ad_z)
        ad_vz1 = np.full((size_set, 1), ad_vz)

        inds = [0, 1, 2, 3, 4, 5]

        if get_learned_V:
            for ii in tqdm(range(learned_V.shape[0])):
                for jj in range(learned_V.shape[1]):
                    for kk in range(learned_V.shape[2]):
                        for ll in range(learned_V.shape[3]):
                            for mm in range(learned_V.shape[4]):
                                for nn in range(learned_V.shape[5]):
                                    tmp_point = [
                                        xv_orig[ii,jj,kk,ll,mm,nn], vxv_orig[ii,jj,kk,ll,mm,nn],
                                        yv_orig[ii,jj,kk,ll,mm,nn], vyv_orig[ii,jj,kk,ll,mm,nn],
                                        zv_orig[ii,jj,kk,ll,mm,nn], vzv_orig[ii,jj,kk,ll,mm,nn],
                                        ad_x,  ad_vx,
                                        ad_y, ad_vy,
                                        ad_z, ad_vz
                                    ]
                                    learned_V[ii,jj,kk,ll,mm,nn] = evaluate_V(tmp_point, policy) 
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
        
        xv_orig, yv_orig, zv_orig, vxv_orig, vyv_orig, vzv_orig, adxv_orig, adyv_orig, \
            adzv_orig, advxv_orig, advyv_orig, advzv_orig = \
            np.meshgrid(oned_x, oned_y, oned_z, oned_vx, oned_vy, oned_vz,
                        oned_adx, oned_ady, oned_adz, oned_advx, oned_advy, oned_advz)
        if get_learned_V: learned_V = np.zeros(xv_orig.shape)

        xv = xv_orig.reshape(-1, 1)
        yv = yv_orig.reshape(-1, 1)
        z01 = zv_orig.reshape(-1, 1)
        ego_vx1 = vxv_orig.reshape(-1, 1)
        ego_vy1 = vyv_orig.reshape(-1, 1)
        ego_vz1 = vzv_orig.reshape(-1, 1)
        ad_x1 = adxv_orig.reshape(-1, 1)
        ad_y1 = adyv_orig.reshape(-1, 1)
        ad_z1 = adzv_orig.reshape(-1, 1)
        ad_vx1 = advxv_orig.reshape(-1, 1)
        ad_vy1 = advyv_orig.reshape(-1, 1)
        ad_vz1 = advzv_orig.reshape(-1, 1)

        inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        if get_learned_V:
            for ii in tqdm(range(learned_V.shape[0])):
                for jj in range(learned_V.shape[1]):
                    for kk in range(learned_V.shape[2]):
                        for ll in range(learned_V.shape[3]):
                            for mm in range(learned_V.shape[4]):
                                for nn in range(learned_V.shape[5]):
                                    for oo in range(learned_V.shape[6]):
                                        for pp in range(learned_V.shape[7]):
                                            for qq in range(learned_V.shape[8]):
                                                for rr in range(learned_V.shape[9]):
                                                    for ss in range(learned_V.shape[10]):
                                                        for tt in range(learned_V.shape[11]): 
                                                            tmp_point = [
                                                                xv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt], 
                                                                vxv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt],
                                                                yv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt], 
                                                                vyv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt],
                                                                zv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt], 
                                                                vzv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt],
                                                                adxv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt],  
                                                                advxv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt],
                                                                adyv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt], 
                                                                advyv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt],
                                                                adzv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt], 
                                                                advzv_orig[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt]
                                                            ]
                                                            learned_V[ii,jj,kk,ll,mm,nn, oo, pp, qq, rr, ss, tt] = \
                                                                evaluate_V(tmp_point, policy) 
    else:
        raise NotImplementedError

    candidates = np.hstack((xv, ego_vx1,
                            yv, ego_vy1,
                            z01, ego_vz1,
                            ad_x1, ad_vx1,
                            ad_y1, ad_vy1,
                            ad_z1, ad_vz1))
    if get_costs: 
        # costs = np.expand_dims(f(candidates), -1)
        costs = f(candidates)
    else:
        costs = None
    print("Done!")
    return candidates[:, inds], costs, oned_x, oned_y, candidates, learned_V

def plot_main_gp(learned_V, beta, oned_x, oned_y, 
                albert_alpha, model, dim, ego_setting, adversary_setting,
                range_x, V_discretization, model_discretization,
                state_expander, fig_name, fig_name_colorbar):
    
    ego_vx, ego_vy, ego_z, ego_vz = ego_setting
    ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz = adversary_setting

    # Take a slice of V if more than 2D
    if dim == 2:
        inds = [0, 2]
    elif dim == 3:
        inds = [0, 2, 4]
        oned_z = np.arange(range_x[4][0], range_x[4][1], V_discretization)
        index_of_closest_value = np.argmin(np.abs(oned_z - ego_z))
        learned_V = learned_V[:, :, index_of_closest_value]
    elif dim == 4:
        inds = [0, 1, 2, 4]
        oned_z = np.arange(range_x[4][0], range_x[4][1], V_discretization)
        index_of_closest_valuez = np.argmin(np.abs(oned_z - ego_z))
        oned_vx = np.arange(range_x[1][0], range_x[1][1], V_discretization)
        index_of_closest_valuevx = np.argmin(np.abs(oned_vx - ego_vx))
        learned_V = learned_V[:, :, index_of_closest_valuez, index_of_closest_valuevx]
    elif dim == 6:
        inds = [0, 1, 2, 3, 4, 5]
        oned_z = np.arange(range_x[4][0], range_x[4][1], V_discretization)
        index_of_closest_valuez = np.argmin(np.abs(oned_z - ego_z))
        oned_vx = np.arange(range_x[1][0], range_x[1][1], V_discretization)
        index_of_closest_valuevx = np.argmin(np.abs(oned_vx - ego_vx))
        oned_vy = np.arange(range_x[3][0], range_x[3][1], V_discretization)
        index_of_closest_valuevy = np.argmin(np.abs(oned_vy - ego_vy))
        oned_vz = np.arange(range_x[5][0], range_x[5][1], V_discretization)
        index_of_closest_valuevz = np.argmin(np.abs(oned_vz - ego_vz))
        learned_V = learned_V[:, :, index_of_closest_valuez, index_of_closest_valuevx, \
                            index_of_closest_valuevy, index_of_closest_valuevz]
    elif dim == 12:
        inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        oned_z = np.arange(range_x[4][0], range_x[4][1], V_discretization)
        index_of_closest_valuez = np.argmin(np.abs(oned_z - ego_z))
        oned_vx = np.arange(range_x[1][0], range_x[1][1], V_discretization)
        index_of_closest_valuevx = np.argmin(np.abs(oned_vx - ego_vx))
        oned_vy = np.arange(range_x[3][0], range_x[3][1], V_discretization)
        index_of_closest_valuevy = np.argmin(np.abs(oned_vy - ego_vy))
        oned_vz = np.arange(range_x[5][0], range_x[5][1], V_discretization)
        index_of_closest_valuevz = np.argmin(np.abs(oned_vz - ego_vz))
        oned_adx = np.arange(range_x[6][0], range_x[6][1], V_discretization)
        index_of_closest_valueadx = np.argmin(np.abs(oned_adx - ad_x))
        oned_ady = np.arange(range_x[8][0], range_x[8][1], V_discretization)
        index_of_closest_valueady = np.argmin(np.abs(oned_ady - ad_y))
        oned_adz = np.arange(range_x[10][0], range_x[10][1], V_discretization)
        index_of_closest_valueadz = np.argmin(np.abs(oned_adz - ad_z))
        oned_advx = np.arange(range_x[7][0], range_x[7][1], V_discretization)
        index_of_closest_valueadvx = np.argmin(np.abs(oned_advx - ad_vx))
        oned_advy = np.arange(range_x[9][0], range_x[9][1], V_discretization)
        index_of_closest_valueadvy = np.argmin(np.abs(oned_advy - ad_vy))
        oned_advz = np.arange(range_x[11][0], range_x[11][1], V_discretization)
        index_of_closest_valueadvz = np.argmin(np.abs(oned_advz - ad_vz))
        learned_V = learned_V[:, :, index_of_closest_valuez, index_of_closest_valuevx, \
                            index_of_closest_valuevy, index_of_closest_valuevz,
                            index_of_closest_valueadx, index_of_closest_valueady,
                            index_of_closest_valueadz, index_of_closest_valueadvx,
                            index_of_closest_valueadvy, index_of_closest_valueadvz]
    else:
        raise NotImplementedError

    # Find a slice of model candidates 
    candidates_for_plotting, _, _, _, full_candidates_for_plotting, _ = \
                                    get_ground_truths_for_a_grid(range_x, ego_setting, 
                                    adversary_setting, 
                                    None, model_discretization, 2, None,
                                    get_costs=False,
                                    get_learned_V=False)
    mu, var = model.m.predict(full_candidates_for_plotting[:, inds], full_cov=False)

    learnedV_xs = np.arange(range_x[0][0], range_x[0][1], V_discretization)
    learnedV_ys = np.arange(range_x[2][0], range_x[2][1], V_discretization)

    plt.figure()
    plt.contour(learnedV_xs,
            learnedV_ys,
            learned_V,
            levels=[0.0],
            colors="black",
            linewidths=3)
    plt.contour(learnedV_xs,
            learnedV_ys,
            learned_V,
            levels=[albert_alpha],
            colors="gray",
            linewidths=2) 
    criterion = mu + beta * np.sqrt(var) 
    criterion = criterion.reshape(len(oned_y), len(oned_x))
    plt.contour(oned_x,
                oned_y,
                criterion,
                levels=[0.0],
                colors="lightblue",
                linewidths=2)
    criterion = mu - beta * np.sqrt(var)  
    criterion = criterion.reshape(len(oned_y), len(oned_x))
    plt.contour(oned_x,
                oned_y,
                criterion,
                levels=[0.0],
                colors="blue",
                linewidths=2)
    if model.acq_cache.size > 0:
        acq_points = np.array(model.acq_cache)
        if dim == 2 or dim == 3:
            plt.scatter(acq_points[:, 0], acq_points[:, 1], color='g')
        elif dim == 4 or dim == 6 or dim == 12:
            plt.scatter(acq_points[:, 0], acq_points[:, 2], color='g')
        else:
            raise NotImplementedError
    plt.savefig(fig_name)
    plt.figure()
    plt.contourf(oned_x,
                oned_y,
                criterion)
    plt.colorbar()
    plt.savefig(fig_name_colorbar)

def validate_final_level_set(model, candidates, true_costs, beta):
    print("Running validation of final GP level set")
    mu, var = model.m.predict(candidates, full_cov=False)
    criterion = mu - beta * np.sqrt(var) # Conservative bc less likely to say state is in BRT
    
    assert len(criterion) == len(true_costs)

    gp_below_zero = np.where(criterion <= 0)
    gp_above_zero = np.where(criterion > 0)
    true_below_zero = np.where(true_costs <= 0)
    true_above_zero = np.where(true_costs > 0)

    true_neg = np.intersect1d(gp_below_zero, true_below_zero)
    false_neg = np.intersect1d(gp_below_zero, true_above_zero)
    true_pos = np.intersect1d(gp_above_zero, true_above_zero)
    false_pos = np.intersect1d(gp_above_zero, true_below_zero)

    tpr = len(true_pos) / (len(true_pos) + len(false_neg))
    fpr = len(false_pos) / (len(false_pos) + len(true_neg))
    tnr = len(true_neg) / (len(true_neg) + len(false_pos))
    fnr = len(false_neg) / (len(false_neg) + len(true_pos))

    print("Done running validation!")

    return tpr, fpr, tnr, fnr

def validate_albert(policy, alpha, candidates, true_costs, state_expander):
    print("Running validation of Albert's final level set")
    criterion = evaluate_V_batch(state_expander(candidates), policy)
    criterion = criterion - alpha
    
    assert len(criterion) == len(true_costs)

    pred_below_zero = np.where(criterion <= 0)
    pred_above_zero = np.where(criterion > 0)
    true_below_zero = np.where(true_costs <= 0)
    true_above_zero = np.where(true_costs > 0)

    true_neg = np.intersect1d(pred_below_zero, true_below_zero)
    false_neg = np.intersect1d(pred_below_zero, true_above_zero)
    true_pos = np.intersect1d(pred_above_zero, true_above_zero)
    false_pos = np.intersect1d(pred_above_zero, true_below_zero)

    tpr = len(true_pos) / (len(true_pos) + len(false_neg))
    fpr = len(false_pos) / (len(false_pos) + len(true_neg))
    tnr = len(true_neg) / (len(true_neg) + len(false_pos))
    fnr = len(false_neg) / (len(false_neg) + len(true_pos))

    print("Done running validation!")

    return tpr, fpr, tnr, fnr

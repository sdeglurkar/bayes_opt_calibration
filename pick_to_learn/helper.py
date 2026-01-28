import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from main_model import MainGP

def batched_rollouts_generator(value_fn, grid, dynamics, time_steps, dt, goal_R, theta_value):
    value_gradients = grid.grad_values(value_fn[-1, :, :, :])
    def rollout(state, plot_traj=False):
        orig_state = state
        state_traj = [state]
        for _ in range(time_steps):
            index = grid.nearest_index(state)
            grad_val = value_gradients[index[0], index[1], index[2]]
            control = dynamics.optimal_control(None, None, grad_val)
            state = state + dt*dynamics(state, control, disturbance=np.array([0.]), time=None)
            state_traj.append(state)
        state_traj = np.array(state_traj)

        cost = jnp.min(jnp.linalg.norm(state_traj, axis=-1) - goal_R, axis=-1)
        print("Ran one rollout", orig_state, cost)
        return cost
    
    def batched_rollouts(states):
        costs = []
        for state in states:
            state = np.append(state, theta_value)
            cost = np.array(rollout(state))
            costs.append(cost)
        print("Done running rollouts!")
        return np.expand_dims(np.array(costs), -1)
    
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

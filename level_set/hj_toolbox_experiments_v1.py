import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

from dynamics import dynamics


grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([bottom_left[0], bottom_left[1], x0[2]-v_grid_real_size, x0[3]-np.pi]),
                                                                   np.array([top_right[0], top_right[1], x0[2]+v_grid_real_size, x0[3]+np.pi])),
                                                                   (num_x_grids, num_y_grids, num_v_grids, num_theta_grids),
                                                                   periodic_dims=3)
# control_space = hj.sets.Box(
#     lo=jnp.array([-15, -np.pi/4]),
#     hi=jnp.array([15, np.pi/4])
# )
# disturbance_space = hj.sets.Box(
#     lo=jnp.array([-0.1,-0.1]),
#     hi=jnp.array([0.1,0.1])
# )

frt = lambda x: jnp.maximum(x,0)
goalR = 1.0
velocity = 1.0
omega_max = 1.0
angle_alpha_factor = 1.0
set_mode = 'avoid'  
diff_model = True  # Doesn't matter
freeze_model = False # Doesn't matter - to avoid a NotImplementedError
max_accel = 1.0
dynam = dynamics.Dubins3D(goalR, velocity, omega_max, angle_alpha_factor, set_mode, diff_model, freeze_model)
brt = lambda x: jnp.maximum(x,0)
solver_settings = hj.SolverSettings.with_accuracy(accuracy="very_high",
                                                    hamiltonian_postprocessor=brt
                                                  )


values = hj.shapes.shape_ellipse(grid=grid, center=x0, radii=0.5*np.array([range_x, range_y, range_v, range_t]))

time = 0.
target_time = 10.
times = np.linspace(time, target_time, num=30)
times, all_values = hj.solve(solver_settings, dynam, grid, times, values)
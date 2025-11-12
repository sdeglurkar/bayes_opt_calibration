import jax
import jax.numpy as jnp
import numpy as np

from IPython.display import HTML
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import hj_reachability as hj


THETA_INDEX = 0 #30
DUBINS_VELOCITY = 10.0

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


# dynamics = hj.systems.Air3d()
dynamics = DubinsCar(DUBINS_VELOCITY)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-6., -10., 0.]),
                                                                           np.array([20., 10., 2 * np.pi])),
                                                               (51, 40, 50),
                                                               periodic_dims=2)
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 5

solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)



time = 0.
target_time = -2.8
target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)

plt.jet()
plt.figure(figsize=(13, 8))
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values[:, :, THETA_INDEX].T)
plt.colorbar()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            target_values[:, :, THETA_INDEX].T,
            levels=0,
            colors="black",
            linewidths=3)
plt.show()

# fig = go.Figure(data=go.Isosurface(x=grid.states[..., 0].ravel(),
#                              y=grid.states[..., 1].ravel(),
#                              z=grid.states[..., 2].ravel(),
#                              value=target_values.ravel(),
#                              colorscale="jet",
#                              isomin=0,
#                              surface_count=1,
#                              isomax=0))
# fig.show()


# times = np.linspace(0, -2.8, 57)
# initial_values = values
# all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values)

# vmin, vmax = all_values.min(), all_values.max()
# levels = np.linspace(round(vmin), round(vmax), round(vmax) - round(vmin) + 1)
# fig = plt.figure(figsize=(13, 8))

# def render_frame(i, colorbar=False):
#     plt.contourf(grid.coordinate_vectors[0],
#                  grid.coordinate_vectors[1],
#                  all_values[i, :, :, THETA_INDEX].T,
#                  vmin=vmin,
#                  vmax=vmax,
#                  levels=levels)
#     if colorbar:
#         plt.colorbar()
#     plt.contour(grid.coordinate_vectors[0],
#                 grid.coordinate_vectors[1],
#                 target_values[:, :, THETA_INDEX].T,
#                 levels=0,
#                 colors="black",
#                 linewidths=3)

# render_frame(0, True)
# plt.show()








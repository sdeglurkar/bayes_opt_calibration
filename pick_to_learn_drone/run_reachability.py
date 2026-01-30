import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

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

def solve_brt(dubins_velocity, theta_value, goal_R, final_time, range_x, time_disc=100):
    dynamics = DubinsCar(dubins_velocity)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(
                                                np.array([range_x[0][0], range_x[1][0], 0.]),
                                                np.array([range_x[0][1], range_x[1][1], 2 * np.pi])),
                                                (51, 40, 50),
                                                periodic_dims=2)
    index = grid.nearest_index([0., 0., theta_value])
    theta_index = index[2]

    values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - goal_R
    # print("Running HJR Solver")
    solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

    times = np.linspace(0, final_time, time_disc) 
    initial_values = values
    all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values)
    # print("Done!")

    return all_values, theta_index, grid, dynamics
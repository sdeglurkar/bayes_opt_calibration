import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from dynamics import dynamics


dt = 0.0001


goalR = 1.0
velocity = 1.0
omega_max = 1.0
angle_alpha_factor = 1.0
set_mode = 'reach'
diff_model = True  # Doesn't matter
freeze_model = False # Doesn't matter
system = dynamics.Dubins3D(goalR, velocity, omega_max, angle_alpha_factor, set_mode, diff_model, freeze_model)

print("\n")
ctrl = system.optimal_control([0.0, 0.0, 0.0], torch.Tensor([1.0, 1.0, 1.0]))
print(ctrl)
print(system.dsdt(torch.Tensor([0.0, 0.0, 0.0]), ctrl, disturbance=None))


# system.equivalent_wrapped_state(state_trajs[:, k] + dt*system.dsdt(state_trajs[:, k], ctrl_trajs[:, k], None))


print("\nYAY!")
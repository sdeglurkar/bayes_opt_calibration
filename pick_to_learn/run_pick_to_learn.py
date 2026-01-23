import matplotlib.pyplot as plt
import numpy as np

from pick_to_learn import PickToLearn

obj = PickToLearn()
obj.setup()

for i in range(len(obj.model_list)):
    model = obj.model_list[i]
    error_gp = obj.error_gp_list[i]
    rng_instance = obj.rng_list[i]
    obj.picktolearn_alg(model, error_gp, rng_instance, i)

print(obj.T_x)
print(obj.T_y)
obj.run_validation()


import matplotlib.pyplot as plt
import numpy as np

from find_picktolearn_bound import *
from pick_to_learn import PickToLearn
from pick_to_learn_settings import *


obj = PickToLearn()
obj.setup()

results = {'T': [], 'epsU': []}
for i in range(len(obj.model_list)):
    model = obj.model_list[i]
    error_gp = obj.error_gp_list[i]
    rng_instance = obj.rng_list[i]
    obj.picktolearn_alg(model, error_gp, rng_instance, i)
    epsL, epsU = find_epsLU(len(obj.T_x[i]), DESIRED_N, DELTA)
    results['T'].append(len(obj.T_x[i]))
    results['epsU'].append(epsU)

obj.plot_multiple_models(obj.model_list, obj.seed_list, obj.candidates, obj.oned_x, 
                        obj.oned_y, stage='final')
obj.run_validation()
print(results)


import matplotlib.pyplot as plt
import numpy as np
import pickle

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

obj.plot_multiple_models(obj.learned_V, obj.model_list, obj.seed_list,  
                                    obj.oned_x, obj.oned_y, obj.albert_alphas,
                                    stage='final')
out1 = obj.run_validation()
print(results)
print("Size of C + Number of Initial GP Samples:", NUM_CALIBRATION_POINTS + \
                                                        NUM_MODEL_INIT_ITERS)
out2 = obj.validate_albert_method()

final_dict = out1 | out2
final_dict = final_dict | results
final_dict['size_C'] = NUM_CALIBRATION_POINTS
final_dict['num_model_init_iters'] = NUM_MODEL_INIT_ITERS
with open(f"{EXPERIMENT_PICKLE_NAME}.pkl", 'wb') as f:
    pickle.dump(final_dict, f)


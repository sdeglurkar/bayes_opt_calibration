import matplotlib.pyplot as plt
import numpy as np
import pickle

from pick_to_learn_settings import *


with open(f"{EXPERIMENT_PICKLE_NAME}.pkl", "rb") as f:
    results = pickle.load(f)

print("\n---------------\nExperiment name: ", EXPERIMENT_PICKLE_NAME)
# print(results.keys())
# print(results['our_method'].keys())
# print(results['albert_itr_method'].keys())
# print(results['albert_robust_method'].keys())

if 'num_functional_seeds' in results:
    print("Number of valid seeds:", results['num_functional_seeds'], "/", \
            len(MULTIPLE_SEED_LIST), "\n")
else:
    print("All seeds successful\n")

robust_albert = results['albert_robust_method']
avg_num_samples = np.mean(np.array(results['T']) + results['size_C'] + \
                    results['num_model_init_iters'])
print("Our Method/Albert's Methods, Average Number of Samples", \
        avg_num_samples, np.mean(results['albert_itr_method']['num_samples']), \
        robust_albert['overall'][4], robust_albert['min_N'][4], \
        robust_albert['min_alpha'][4], robust_albert['median'][4])
print("Our Method/Albert's Methods, Average FNR/FPR", \
        results['our_method']['avg_fnr'], '/', results['our_method']['avg_fpr'], \
        results['albert_itr_method']['avg_fnr'], '/', results['albert_itr_method']['avg_fpr'], \
        robust_albert['overall'][1], '/', robust_albert['overall'][2], \
        robust_albert['min_N'][1], '/', robust_albert['min_N'][2], \
        robust_albert['min_alpha'][1], '/', robust_albert['min_alpha'][2], \
        robust_albert['median'][1], '/', robust_albert['median'][2])


print("Our Method Average Eps Bar", np.mean(results['epsU']))
print("Albert's Robust Methods, Average Epsilons", robust_albert['overall'][5], \
        robust_albert['min_N'][5], robust_albert['min_alpha'][5], \
        robust_albert['median'][5])
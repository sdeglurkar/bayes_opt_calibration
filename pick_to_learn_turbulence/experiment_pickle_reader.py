import matplotlib.pyplot as plt
import numpy as np
import pickle

from pick_to_learn_settings import *


with open(f"{EXPERIMENT_PICKLE_NAME}.pkl", "rb") as f:
    results = pickle.load(f)

print(results.keys())
print(results['our_method'].keys())
print(results['albert_itr_method'].keys())
print(results['albert_robust_method'].keys())
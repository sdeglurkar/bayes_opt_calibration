import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from experiment_pickle_reader import *
from find_picktolearn_bound import *
from find_size_of_C import *
from pick_to_learn import PickToLearn
from pick_to_learn_settings import *

from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import get_env_and_policy


########################### SETUP ###########################
args = add_args(PARSER)

args.env, args.policy = get_env_and_policy(args)
args.f = batched_rollouts_generator(args.horizon, args.policy, args) 
args.rng = np.random.default_rng(args.random_seed)

if args.slice == 'slice_1':
    # Basic slice
    args.adversary_setting = [0.4, 0.0, -2.2, 0.3, 0.0, 0.0]  # ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz
    args.ego_setting = [0.0, 0.7, 0.0, 0.0]  # ego_vx, ego_vy, ego_z, ego_vz
    slice_str = 'basicslice_'
elif args.slice == 'slice_2':
    # New slice 1
    args.adversary_setting = [0.4, 0.0, -2.2, 0.3, 0.0, 0.0]  # ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz
    args.ego_setting = [0.0, 0.0, 0.05, -0.5]  # ego_vx, ego_vy, ego_z, ego_vz
    slice_str = 'newslice_'
else:
    raise Exception("Unknown slice!")

if args.input_dim <= 3:
    args.desired_N = 4000
elif args.input_dim == 4:
    args.desired_N = 8000
elif args.input_dim == 6:
    args.desired_N = 10000

if args.random_active_learning:
    random_str = 'randomacq_'
else:
    random_str = 'boundaryacq_'

SIZE_C = find_size_of_C(args.alpha, args.tolerance_alpha, args.beta_conformal)
print("Number of errors possible:", (1-np.ceil((1-args.alpha) * (SIZE_C + 1))/SIZE_C) * SIZE_C)
args.num_calibration_points = SIZE_C 
assert args.num_calibration_points >= (1-args.alpha)/args.alpha  # Necessary for conformal prediction
assert args.num_calibration_points >= SIZE_C  # Necessary for conformal prediction

EXPERIMENT_STRING = str(INPUT_DIM) + \
    'D_' + slice_str + random_str + 'N' + str(args.desired_N) + '_init' + \
        str(args.num_model_init_iters) + '_decay' + str(args.decay_factor) + \
        'thres' + str(args.ehat_threshold) + '_alpha' + str(args.alpha) + \
        '_tolalpha' + str(args.tolerance_alpha) 

args.picktolearn_logdir = 'drone_model_dir_' + EXPERIMENT_STRING
os.makedirs(args.picktolearn_logdir, exist_ok=True)
args.experiment_pickle_name = 'drone_' + EXPERIMENT_STRING
args.picktolearn_validation_logdir = 'drone_pickles_' + EXPERIMENT_STRING
os.makedirs(args.picktolearn_validation_logdir, exist_ok=True)

if args.experiment_pickle_reader:
    read_pickle(args.experiment_pickle_name, MULTIPLE_SEED_LIST)
    exit()

########################### RUN ###########################
obj = PickToLearn(args)
obj.setup()

results = {'T': [], 'epsU': []}
for i in range(len(obj.model_list)):
    model = obj.model_list[i]
    error_gp = obj.error_gp_list[i]
    rng_instance = obj.rng_list[i]
    obj.picktolearn_alg(model, error_gp, rng_instance, i)
    
obj.post_alg_all_seeds()
for i in range(len(obj.model_list)):
    epsL, epsU = find_epsLU(len(obj.T_x[i]), args.desired_N, args.delta)
    results['T'].append(len(obj.T_x[i]))
    results['epsU'].append(epsU)

obj.plot_multiple_models(obj.learned_V, obj.model_list, obj.seed_list,  
                                    obj.oned_x, obj.oned_y, obj.albert_levels,
                                    stage='final')
out1 = obj.run_validation()
print(results)
print("Size of C + Number of Initial GP Samples:", args.num_calibration_points + \
                                                        args.num_model_init_iters)
out2 = obj.validate_albert_method()

final_dict = out1 | out2
final_dict = final_dict | results
final_dict['size_C'] = args.num_calibration_points
final_dict['num_model_init_iters'] = args.num_model_init_iters
final_dict['num_functional_seeds'] = list(obj.seed_failed).count(False)
with open(f"{args.experiment_pickle_name}.pkl", 'wb') as f:
    pickle.dump(final_dict, f)


import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
import os
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import get_args, get_env_and_policy

from helper import *
from find_size_of_C import *

from scipy.stats import norm

########################### DYNAMICS SETTINGS ###########################
# Basic slice
# ADVERSARY_SETTING = [0.4, 0.0, -2.2, 0.3, 0.0, 0.0]  # ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz
# EGO_SETTING = [0.0, 0.7, 0.0, 0.0]  # ego_vx, ego_vy, ego_z, ego_vz
# New slice 1
# ADVERSARY_SETTING = [0.4, 0.0, -2.2, 0.3, 0.0, 0.0]  # ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz
# EGO_SETTING = [0.0, 0.0, 0.05, -0.5]  # ego_vx, ego_vy, ego_z, ego_vz
# Hardware slice
ADVERSARY_SETTING = [0.4, 0.0, -2.2, 0.0, 0.5, 0.0]  # ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz
EGO_SETTING = [0.0, 0.0, 0.5, 0.0]  # ego_vx, ego_vy, ego_z, ego_vz

# This is the environment boundary
# RANGE_X = [[-1.0, 1.0], [-1.0, 1.0], [-3.2, 0.0], [0.1, 1.0], [-1.0, 1.0], [-1.0, 1.0],
#             [-1.0, 1.0], [-1.0, 1.0], [-3.2, 0.0], [0.1, 0.5], [-1.0, 1.0], [-1.0, 1.0]]
# Choose a subset
RANGE_X = [[-1.0, 1.0], [-1.0, 1.0], [-2.7, 0.0], [0.1, 1.0], [-1.0, 1.0], [-1.0, 1.0],
            [-1.0, 1.0], [-1.0, 1.0], [-2.7, 0.0], [0.1, 0.5], [-1.0, 1.0], [-1.0, 1.0]]
HORIZON = 30

########################### GAUSSIAN PROCESS SETTINGS ###########################
INPUT_DIM = 2
NUM_MODEL_INIT_ITERS = 40 #10

CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NOISE_VAR = 0.001 
COST_THRES = 0.0 
LENGTH_SCALE = 0.25
MODEL_CANDIDATES_DISCRETIZATION = 0.1

########################### RANDOM SEED SETTINGS ###########################
RANDOM_SEED = 0 #100  # If only a single seed is being run
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = True
MULTIPLE_SEED_LIST = [0, 1, 2, 3, 17, 22, 24, 27, 31, 100] #[0, 1, 3, 17, 22] #[0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DESIRED_N = 4000 #3600 #500 #1000 #3600

if INPUT_DIM <= 3:
    DESIRED_N = 4000
elif INPUT_DIM == 4:
    DESIRED_N = 8000
elif INPUT_DIM == 6:
    DESIRED_N = 10000
DELTA = 1e-4

########################### ACQUISITION FN SETTINGS ###########################
ALPHA = 0.05
TOLERANCE_ALPHA = 0.03
RANDOM_ACQUISITION = False

DECAY_RATE = 0.95
BETA_CONFORMAL = 0.1 #1e-12
SIZE_C = find_size_of_C(ALPHA, TOLERANCE_ALPHA, BETA_CONFORMAL)
print("Number of errors possible:", (1-np.ceil((1-ALPHA) * (SIZE_C + 1))/SIZE_C) * SIZE_C)
NUM_CALIBRATION_POINTS = SIZE_C 
EHAT_THRESHOLD = 0.3
MAX_NUM_ACQUIRED_POINTS = 70 
assert NUM_CALIBRATION_POINTS >= (1-ALPHA)/ALPHA  # Necessary for conformal prediction
assert NUM_CALIBRATION_POINTS >= SIZE_C  # Necessary for conformal prediction
NUM_ERROR_GP_POINTS = 50  # Not used

########################### BASELINE SETTINGS ###########################
ALBERT_EPS = 0.05 #0.1
ALBERT_DELT = BETA_CONFORMAL #1e-12 #0.05
ALBERT_M = 7
ROBUST_ALBERT_N_SWEEP = [150, 200, 250, 300, 350, 500, 750, 1000]
ROBUST_ALBERT_ALPHA_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.9, 1.0]
# ROBUST_ALBERT_N_SWEEP = [150, 200]
# ROBUST_ALBERT_ALPHA_SWEEP = [0.0, 0.05]

########################### OTHER SETTINGS ###########################
EXPERIMENT_STRING = str(INPUT_DIM) + \
    'D_basicslice_boundaryacq_N8000_init40_decay0.95thres0.3_alpha0.15_tolalpha0.05'

# VALIDATION_DISCRETIZATION = [0.05, 0.01, 0.05, 0.01, 0.01, 0.01, \
#                             0.05, 0.01, 0.05, 0.01, 0.01, 0.01]
VALIDATION_DISCRETIZATION = [0.05, 0.05, 0.05, 0.01, 0.05, 0.05, \
                            0.05, 0.05, 0.05, 0.01, 0.05, 0.05]
if INPUT_DIM >= 4:
    VALIDATION_DISCRETIZATION = [0.08, 0.08, 0.08, 0.06, 0.08, 0.08, \
                            0.08, 0.08, 0.08, 0.06, 0.08, 0.08]
PLOT_DURING_ACQUISITION = True
FONTSIZE = 16
LOGDIR = 'drone_model_dir_' + EXPERIMENT_STRING
os.makedirs(LOGDIR, exist_ok=True)
EXPERIMENT_PICKLE_NAME = 'drone_' + EXPERIMENT_STRING
ERROR_GP_LOGDIR = 'drone_errorgp_dir_' + str(INPUT_DIM) + 'D'
VALIDATION_LOGDIR = 'drone_pickles_' + EXPERIMENT_STRING
os.makedirs(VALIDATION_LOGDIR, exist_ok=True)

########################### GET POLICY ###########################
ARGS = get_args()
ENV, POLICY = get_env_and_policy(ARGS)
F = batched_rollouts_generator(HORIZON, POLICY, ARGS) 


    
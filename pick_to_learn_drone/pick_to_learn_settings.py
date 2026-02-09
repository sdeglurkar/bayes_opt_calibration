import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import get_args, get_env_and_policy

from helper import *

from scipy.stats import norm

########################### DYNAMICS SETTINGS ###########################
# Give adversary (last 6 dims) the same range as ego (first 6 dims)
RANGE_X = [[-0.9, 0.9], [0.0, 0.1], [-2.6, 0.0], [0.6, 0.8], [0.0, 0.1], [0.0, 0.1],
        [-0.9, 0.9], [0.0, 0.1], [-2.6, 0.0], [0.6, 0.8], [0.0, 0.1], [0.0, 0.1]]
ADVERSARY_SETTING = [0.4, 0.0, -2.2, 0.3, 0.0, 0.0]
EGO_SETTING = [0.0, 0.7, 0.0, 0.0]
HORIZON = 30

########################### GAUSSIAN PROCESS SETTINGS ###########################
INPUT_DIM = 2
CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NOISE_VAR = 0.001 
COST_THRES = 0.0 
LENGTH_SCALE = 0.25
NUM_MODEL_INIT_ITERS = 40 #10
MODEL_CANDIDATES_DISCRETIZATION = 0.1

########################### RANDOM SEED SETTINGS ###########################
RANDOM_SEED = 0 #100  # If only a single seed is being run
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = True
MULTIPLE_SEED_LIST = [0, 1, 3] #[0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DELTA = 1e-4
DESIRED_N = 3600 #500 #1000 #3600

########################### ACQUISITION FN SETTINGS ###########################
ALPHA = 0.01 #0.05
NUM_CALIBRATION_POINTS = 100 #200 #100
NUM_ERROR_GP_POINTS = 50
EHAT_THRESHOLD = 0.3
MAX_NUM_ACQUIRED_POINTS = 50
assert NUM_CALIBRATION_POINTS >= (1-ALPHA)/ALPHA  # Necessary for conformal prediction

########################### OTHER SETTINGS ###########################
VALIDATION_DISCRETIZATION = 0.05
PLOT_DURING_ACQUISITION = False
PLOT_D = False
PLOT_VALIDATION_DATA = False
LOGDIR = 'drone_model_dir_' + str(INPUT_DIM) + 'D'
ERROR_GP_LOGDIR = 'drone_errorgp_dir_' + str(INPUT_DIM) + 'D'

ALBERT_EPS = 0.1
ALBERT_DELT = 0.05
ALBERT_M = 7

########################### GET POLICY ###########################
ARGS = get_args()
ENV, POLICY = get_env_and_policy(ARGS)
F = batched_rollouts_generator(HORIZON, POLICY, ARGS) 


    
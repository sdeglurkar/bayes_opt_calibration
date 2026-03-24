import sys 
import os
sys.path.append(os.getcwd())
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import get_parser

from helper import *
from find_size_of_C import *

from scipy.stats import norm

########################### DYNAMICS SETTINGS ###########################
SLICE = 'slice_1'

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
RANDOM_SEED = 0   # If only a single seed is being run
# RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = False
MULTIPLE_SEED_LIST = [0, 1, 2, 3, 17, 22, 24, 27, 31, 100] 
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
# if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DESIRED_N = 4000 

# if INPUT_DIM <= 3:
#     DESIRED_N = 4000
# elif INPUT_DIM == 4:
#     DESIRED_N = 8000
# elif INPUT_DIM == 6:
#     DESIRED_N = 10000
DELTA = 1e-4

########################### ACQUISITION FN SETTINGS ###########################
ALPHA = 0.05
TOLERANCE_ALPHA = 0.03
RANDOM_ACQUISITION = False

DECAY_RATE = 0.95
BETA_CONFORMAL = 0.1 
# SIZE_C = find_size_of_C(ALPHA, TOLERANCE_ALPHA, BETA_CONFORMAL)
# print("Number of errors possible:", (1-np.ceil((1-ALPHA) * (SIZE_C + 1))/SIZE_C) * SIZE_C)
# NUM_CALIBRATION_POINTS = SIZE_C 
EHAT_THRESHOLD = 0.3
MAX_NUM_ACQUIRED_POINTS = 70 
# assert NUM_CALIBRATION_POINTS >= (1-ALPHA)/ALPHA  # Necessary for conformal prediction
# assert NUM_CALIBRATION_POINTS >= SIZE_C  # Necessary for conformal prediction

########################### BASELINE SETTINGS ###########################
ALBERT_EPS = 0.05 
ALBERT_DELT = BETA_CONFORMAL 
ALBERT_M = 7
ROBUST_ALBERT_N_SWEEP = [150, 200, 250, 300, 350, 500, 750, 1000]
ROBUST_ALBERT_LEVEL_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.9, 1.0]

########################### OTHER SETTINGS ###########################
# EXPERIMENT_STRING = str(INPUT_DIM) + \
#     'D_basicslice_boundaryacq_N4000_init40_decay0.95thres0.3_alpha0.05_tolalpha0.03'

VALIDATION_DISCRETIZATION = [0.05, 0.05, 0.05, 0.01, 0.05, 0.05, \
                            0.05, 0.05, 0.05, 0.01, 0.05, 0.05]
if INPUT_DIM >= 4:
    VALIDATION_DISCRETIZATION = [0.08, 0.08, 0.08, 0.06, 0.08, 0.08, \
                            0.08, 0.08, 0.08, 0.06, 0.08, 0.08]
PLOT_DURING_ACQUISITION = False
FONTSIZE = 16
# LOGDIR = 'drone_model_dir_' + EXPERIMENT_STRING
# os.makedirs(LOGDIR, exist_ok=True)
# EXPERIMENT_PICKLE_NAME = 'drone_' + EXPERIMENT_STRING
# ERROR_GP_LOGDIR = 'drone_errorgp_dir_' + str(INPUT_DIM) + 'D'
# VALIDATION_LOGDIR = 'drone_pickles_' + EXPERIMENT_STRING
# os.makedirs(VALIDATION_LOGDIR, exist_ok=True)

########################### GET POLICY ###########################
# ARGS = get_args()
# ENV, POLICY = get_env_and_policy(ARGS)
# F = batched_rollouts_generator(HORIZON, POLICY, ARGS) 


########################### ARGUMENT PARSING ###########################
PARSER = get_parser()

def add_args(parser):
    parser.add_argument("--input_dim", type=int, default=INPUT_DIM, 
        help="Dimension of state space for verification")
    parser.add_argument("--num_model_init_iters", type=int, default=NUM_MODEL_INIT_ITERS, 
        help="Number of samples for initial GP fit")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED, 
        help="Random seed for Approximate Pick-to-Learn")
    parser.add_argument("--multiple_seeds", type=bool, default=MULTIPLE_SEEDS, 
        help="True for multiple runs")
    parser.add_argument("--desired_N", type=int, default=DESIRED_N, 
        help="Size of D in Approximate Pick-to-Learn")
    parser.add_argument("--alpha", type=float, default=ALPHA, 
        help="Value of alpha for conformal prediction")
    parser.add_argument("--tolerance_alpha", type=float, default=TOLERANCE_ALPHA, 
        help="Value of epsilon_alpha for conformal prediction")
    parser.add_argument("--delta", type=float, default=DELTA, 
        help="Delta confidence parameter")
    parser.add_argument("--random_active_learning", type=bool, default=RANDOM_ACQUISITION, 
        help="True if the active learning strategy is random sampling")
    parser.add_argument("--slice", type=str, default=SLICE, 
        help="Either slice 1 or slice 2 of the 12D state space")
    parser.add_argument("--decay_rate", type=float, default=DECAY_RATE, 
        help="Decay rate of the active learning strategy")
    parser.add_argument("--beta_conformal", type=float, default=BETA_CONFORMAL, 
        help="Beta confidence parameter")
    parser.add_argument("--baseline_epsilon", type=float, default=ALBERT_EPS, 
        help="Epsilon for iterative baseline method")
    parser.add_argument("--horizon", type=int, default=HORIZON, 
        help="Time horizon")
    parser.add_argument("--ehat_threshold", type=float, default=EHAT_THRESHOLD, 
        help="Threshold on e_hat to terminate Approximate Pick-to-Learn algorithm")
    parser.add_argument("--plot_during", type=bool, default=PLOT_DURING_ACQUISITION, 
        help="Plotting the reachable set at each iteration")
    parser.add_argument("--max_num_acquired_points", type=int, default=MAX_NUM_ACQUIRED_POINTS, 
        help="Maximum number of iterations")
    parser.add_argument("--experiment_pickle_reader", type=bool, default=False, 
        help="True if only analyzing experiment results")

    args = parser.parse_known_args()[0]
    return args

import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script import env_utils

from helper import *

from scipy.stats import norm

########################### DYNAMICS/REACHABILITY SETTINGS ###########################
# THETA_VALUE = 0.0
# DUBINS_VELOCITY = 10.0
# DT = 0.01
# FINAL_TIME = -1.0
# TRAJ_TIME_STEPS = int(np.abs(FINAL_TIME)/DT)
# goal_R = 5
# RANGE_X = [[-15, 15], [-15, 15]]
HORIZON = 30

########################### GAUSSIAN PROCESS SETTINGS ###########################
INPUT_DIM = 2
CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NOISE_VAR = 0.001 
COST_THRES = 0.0 
LENGTH_SCALE = 0.25
# NUM_MODEL_INIT_ITERS = 10 #40    # Amount of initial random samples

########################### RANDOM SEED SETTINGS ###########################
RANDOM_SEED = 0 #100  # If only a single seed is being run
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = False
MULTIPLE_SEED_LIST = [0, 1, 3] #[0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DELTA = 1e-4
DESIRED_N = 3600 #500 #1000 #3600

########################### ACQUISITION FN SETTINGS ###########################
ALPHA = 0.01 #0.05
NUM_CALIBRATION_POINTS = 200 #100
NUM_ERROR_GP_POINTS = 50
EHAT_THRESHOLD = 0.3
MAX_NUM_ACQUIRED_POINTS = 50
assert NUM_CALIBRATION_POINTS >= (1-ALPHA)/ALPHA  # Necessary for conformal prediction

########################### OTHER SETTINGS ###########################
VALIDATION_DISCRETIZATION = 0.1
PLOT_DURING_ACQUISITION = False
PLOT_D = False
PLOT_VALIDATION_DATA = False
LOGDIR = 'drone_model_dir'
ERROR_GP_LOGDIR = 'drone_errorgp_dir'

########################### GET OPTIMAL VALUE FUNCTION -- BRT ###########################
# ALL_VALUES, THETA_INDEX, GRID, DYNAMICS = \
#         solve_brt(DUBINS_VELOCITY, THETA_VALUE, goal_R, FINAL_TIME, RANGE_X)

# F = batched_rollouts_generator(ALL_VALUES, GRID, DYNAMICS, TRAJ_TIME_STEPS, DT, 
#                                 goal_R, THETA_VALUE)

args = env_utils.get_args()
env, policy = env_utils.get_env_and_policy(args)
F = batched_rollouts_generator(HORIZON, policy, args) 


    
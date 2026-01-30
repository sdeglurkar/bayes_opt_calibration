from scipy.stats import norm

from helper import *
from run_reachability import solve_brt

########################### DYNAMICS/REACHABILITY SETTINGS ###########################
THETA_VALUE = 0.0
DUBINS_VELOCITY = 10.0
DT = 0.01
FINAL_TIME = -1.0
TRAJ_TIME_STEPS = int(np.abs(FINAL_TIME)/DT)
goal_R = 5
RANGE_X = [[-15, 15], [-15, 15]]

########################### GAUSSIAN PROCESS SETTINGS ###########################
INPUT_DIM = 2
CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NOISE_VAR = 0.001 
COST_THRES = 0.0 
LENGTH_SCALE = 0.25
NUM_MODEL_INIT_ITERS = 40 #10 #40    # Amount of initial random samples

########################### RANDOM SEED SETTINGS ###########################
RANDOM_SEED = 0 #100  # If only a single seed is being run
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = False
MULTIPLE_SEED_LIST = [0, 1, 3] #[0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DELTA = 1e-4
WORLD_DISCRETIZATION = 0.5  # Dictates the size of D if gridding the space
DESIRED_N = 3600 #500 #1000 #3600

########################### ACQUISITION FN SETTINGS ###########################
ALPHA = 0.01 #0.05
NUM_CALIBRATION_POINTS = 100 #200 #100
NUM_ERROR_GP_POINTS = 50
EHAT_THRESHOLD = 0.3
MAX_NUM_ACQUIRED_POINTS = 50
assert NUM_CALIBRATION_POINTS >= (1-ALPHA)/ALPHA  # Necessary for conformal prediction

########################### OTHER SETTINGS ###########################
VALIDATION_DISCRETIZATION = 0.5
PLOT_DURING_ACQUISITION = False
PLOT_D = False
PLOT_VALIDATION_DATA = False
LOGDIR = 'approx_picktolearn_model_dir'
ERROR_GP_LOGDIR = 'approx_picktolearn_errorgp_dir'

########################### GET OPTIMAL VALUE FUNCTION -- BRT ###########################
ALL_VALUES, THETA_INDEX, GRID, DYNAMICS = \
        solve_brt(DUBINS_VELOCITY, THETA_VALUE, goal_R, FINAL_TIME, RANGE_X)

F = batched_rollouts_generator(ALL_VALUES, GRID, DYNAMICS, TRAJ_TIME_STEPS, DT, 
                                goal_R, THETA_VALUE)


    
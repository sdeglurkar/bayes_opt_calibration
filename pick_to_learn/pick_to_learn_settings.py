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
CONF_THRES = 0.9
BETA = norm.ppf(CONF_THRES)
NUM_MODEL_INIT_ITERS = 40 #10   # Amount of initial random samples
# NUM_BO_ITERS = 10 #30  # Set to 0 if only random sampling is desired

########################### RANDOM SEED SETTINGS ###########################
RANDOM_SEED = 0 #100  # If only a single seed is being run
RNG = np.random.default_rng(RANDOM_SEED)
MULTIPLE_SEEDS = False 
MULTIPLE_SEED_LIST = [0, 1, 2, 3, 17, 22, 100]
MULTIPLE_RNG_LIST = [np.random.default_rng(seed) for seed in MULTIPLE_SEED_LIST]
if MULTIPLE_SEEDS: assert len(MULTIPLE_SEED_LIST) > 0

########################### PICK-TO-LEARN BOUND SETTINGS ###########################
DELTA = 1e-4
WORLD_DISCRETIZATION = 0.5  # Dictates the size of D if gridding the space
DESIRED_N = 3600 #500 #1000 #3600

########################### ACQUISITION FN SETTINGS ###########################
ALPHA = 0.05
NUM_CALIBRATION_POINTS = 100
NUM_ERROR_GP_POINTS = 50

########################### OTHER SETTINGS ###########################
VALIDATION_DISCRETIZATION = 0.5
PLOT_DURING_ACQUISITION = False
PLOT_D = False
PLOT_VALIDATION_DATA = False

########################### GET OPTIMAL VALUE FUNCTION -- BRT ###########################
range_x = RANGE_X
dubins_velocity = DUBINS_VELOCITY 
theta_value = THETA_VALUE 
final_time = FINAL_TIME

all_values, theta_index, grid, dynamics = \
        solve_brt(dubins_velocity, theta_value, goal_R, final_time, range_x)

F = batched_rollouts_generator(all_values, grid, dynamics, TRAJ_TIME_STEPS, DT, goal_R, theta_value)


    
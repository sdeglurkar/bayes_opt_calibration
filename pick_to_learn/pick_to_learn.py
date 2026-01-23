import GPy
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
from scipy.stats import norm

from conformal_prediction import get_quantile_for_interval_score_fn
from error_estimate_model import ErrorGP
from find_picktolearn_bound import find_epsLU
from helper import *
from main_model import MainGP
from pick_to_learn_settings import *

class PickToLearn():
    def __init__(self):
        self.T_x = []
        self.T_y = []
        self.D_x = None 
        self.D_y = None 
        self.acq_calib_candidates = None 
        self.acq_calib_true_costs = None 
        self.error_gp_candidates = None 
        self.error_gp_true_costs = None 
        self.validation_candidates = None
        self.validation_true_costs = None 
        self.model_list = []
        self.error_gp_list = []
        self.rng_list = MULTIPLE_RNG_LIST

    def setup(self):
        self.D_x, self.D_y = self.get_D()
        self.acq_calib_candidates, self.acq_calib_true_costs = \
            self.get_acquisition_fn_calib_dataset()
        self.error_gp_candidates, self.error_gp_true_costs = \
            self.get_error_gp_dataset()
        self.validation_candidates, self.validation_true_costs = \
            self.get_validation_dataset()
        self.candidates, self.oned_x, self.oned_y = self.get_candidates_helper() 
        if MULTIPLE_SEEDS:
            self.model_list = self.fit_initial_model_multiple_seeds(MULTIPLE_RNG_LIST, 
                                                                    MULTIPLE_SEED_LIST, 
                                                                    self.candidates)
            for model in self.model_list:
                error_gp_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
                                                    self.error_gp_true_costs, BETA)
                error_gp = self.fit_initial_error_gp(self.error_gp_candidates, error_gp_ys)
                self.error_gp_list.append(error_gp)
                self.T_x.append([])
                self.T_y.append([])
        else:
            model = self.fit_initial_model(RNG, RANDOM_SEED, self.candidates)
            self.model_list = [model]
            error_gp_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
                                                    self.error_gp_true_costs, BETA)
            error_gp = self.fit_initial_error_gp(self.error_gp_candidates, error_gp_ys)
            self.error_gp_list = [error_gp]
            self.T_x.append([])
            self.T_y.append([])

    def get_D(self):
        if os.path.isfile(f"D_{DESIRED_N}.pkl"):
            with open(f"D_{DESIRED_N}.pkl", "rb") as f:
                D_candidates, D_true_costs = pickle.load(f)
        else:
            D_candidates, D_true_costs = \
                get_ground_truths_for_random_points(RANGE_X, RNG, F, DESIRED_N)
            with open(f'D_{DESIRED_N}.pkl', 'wb') as f:
                pickle.dump([D_candidates, D_true_costs], f)
        N = len(D_candidates)
        assert N == DESIRED_N
        if PLOT_D:
            plt.scatter(D_candidates[:, 0], D_candidates[:, 1])
            plt.show()
        
        return D_candidates, D_true_costs

    def get_error_gp_dataset(self):
        if os.path.isfile(f"error_gp_data_{NUM_ERROR_GP_POINTS}.pkl"):
            with open(f"error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", "rb") as f:
                error_gp_candidates, error_gp_true_costs = pickle.load(f)
        else:
            error_gp_candidates, error_gp_true_costs = \
                get_ground_truths_for_random_points(RANGE_X, RNG, F, NUM_ERROR_GP_POINTS)
            with open(f"error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", 'wb') as f:
                pickle.dump([error_gp_candidates, error_gp_true_costs], f)

        return error_gp_candidates, error_gp_true_costs

    def get_acquisition_fn_calib_dataset(self):
        if os.path.isfile(f"acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl"):
            with open(f"acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl", "rb") as f:
                acq_calib_candidates, acq_calib_true_costs = pickle.load(f)
        else:
            acq_calib_candidates, acq_calib_true_costs = \
                get_ground_truths_for_random_points(RANGE_X, RNG, F, NUM_CALIBRATION_POINTS)
            with open(f"acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl", 'wb') as f:
                pickle.dump([acq_calib_candidates, acq_calib_true_costs], f)

        return acq_calib_candidates, acq_calib_true_costs
        
    def get_validation_dataset(self):
        if os.path.isfile("validation_data.pkl"):
            with open("validation_data.pkl", "rb") as f:
                validation_candidates, validation_true_costs = pickle.load(f)
        else:
            validation_candidates, validation_true_costs = \
                get_ground_truths_for_a_grid(F, RANGE_X, discretization=VALIDATION_DISCRETIZATION)
            with open('validation_data.pkl', 'wb') as f:
                pickle.dump([validation_candidates, validation_true_costs], f)
        if PLOT_VALIDATION_DATA:
            plt.scatter(validation_candidates[:, 0], validation_candidates[:, 1])
            plt.show()
    
    def get_candidates_helper(self):
        oned_x = np.arange(RANGE_X[0][0], RANGE_X[0][1], 1.0)
        oned_y = np.arange(RANGE_X[1][0], RANGE_X[1][1], 1.0)
        xv, yv = np.meshgrid(oned_x, oned_y)
        candidates = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
        return candidates, oned_x, oned_y

    def fit_initial_model(self, rng_instance, seed_val, candidates):
        mean_function = GPy.core.Mapping(2,1)
        mean_function.f = lambda x: np.expand_dims(x[:, 0]**2 + x[:, 1]**2 - goal_R, -1)
        mean_function.update_gradients = lambda a,b: 0
        mean_function.gradients_X = lambda a,b: 0
        np.random.seed(seed_val)
        model = MainGP(F, mean_function, INPUT_DIM, candidates, RANGE_X, 
                        NOISE_VAR, COST_THRES, CONF_THRES, LENGTH_SCALE, LOGDIR)
        model.initial_setup(NUM_MODEL_INIT_ITERS, rng_instance)

        return model

    def fit_initial_model_multiple_seeds(self, rng_list, seed_list, candidates):
        model_list = []
        for i in range(len(rng_list)):
            rng = rng_list[i]
            seed_val = seed_list[i] 
            model = self.fit_initial_model(rng, seed_val, candidates)
            model_list.append(model)
        return model_list
    
    def plot_model(self, model, candidates, oned_x, oned_y, seed=None):
        if seed:
            fig_name = LOGDIR + f'/gp_init{seed}.png'
            fig_name_colorbar = LOGDIR + f'/gp_init_colorbar{seed}.png'
            mile_name = LOGDIR + f'/mile_init{seed}.png'
            plot_main_gp(model, GRID, candidates, oned_x, oned_y, ALL_VALUES, 
                    THETA_INDEX, BETA, fig_name, fig_name_colorbar, mile_name)
        else:
            fig_name = LOGDIR + f'/gp_init.png'
            fig_name_colorbar = LOGDIR + f'/gp_init_colorbar.png'
            mile_name = LOGDIR + f'/mile_init.png'
            plot_main_gp(model, GRID, candidates, oned_x, oned_y, ALL_VALUES, 
                    THETA_INDEX, BETA, fig_name, fig_name_colorbar, mile_name)                                      

    def plot_multiple_models(self, model_list, seed_list, candidates, oned_x, oned_y):
        for i in range(len(model_list)):
            model = model_list[i]
            seed = seed_list[i]
            self.plot_model(model, candidates, oned_x, oned_y, seed)

    def fit_initial_error_gp(self, X, Y):
        mean_function = None 
        error_gp = ErrorGP(F, mean_function, INPUT_DIM, RANGE_X, NOISE_VAR, 
                        LENGTH_SCALE, ERROR_GP_LOGDIR)
        error_gp.fit(X, Y)
        return error_gp 
    
    def plot_error_gp(self, error_gp, candidates, oned_x, oned_y, fig_name):
        mu, _ = error_gp.m.predict(candidates, full_cov=False)
        mu = mu.reshape(len(oned_x), len(oned_y))
        plt.figure()
        plt.contourf(oned_x,
                    oned_y,
                    mu)
        plt.colorbar()
        plt.savefig(fig_name)
    
    def picktolearn_one_iteration(self, model, error_gp, new_x, rng_instance, model_idx):
        model.acq_cache.append(new_x)
        self.T_x[model_idx].append(new_x)
        self.T_y[model_idx].append(model.f(new_x))
        model.optimize_given_T(self.T_x[model_idx], self.T_y[model_idx])
        remaining = np.setdiff1d(self.D_x, self.T_x[model_idx])
        # Now re-fit the error GP
        error_gp_new_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
                                                    self.error_gp_true_costs, BETA)
        error_gp.fit(self.error_gp_candidates, error_gp_new_ys)
        error_function = error_gp.forward()
        # Get a_{h,eta}(z) and u_{h,eta}(z)
        final_scores, error_variances = model.score_function(remaining, 
                                        rng_instance, error_function, BETA)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(self.acq_calib_candidates, 
                                            self.acq_calib_true_costs, BETA)
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, ALPHA)
        ehat = final_scores + llambda * error_variances
        # Find the new z
        argmax_index = np.argmax(ehat)
        new_x = remaining[argmax_index]
        ehat_value = ehat[argmax_index]
        return new_x, ehat_value
    
    def picktolearn_alg(self, model, error_gp, rng_instance, model_idx):
        error_function = error_gp.forward()
        # Get a_{h,eta}(z) and u_{h,eta}(z)
        final_scores, error_variances = model.score_function(self.D_x, 
                                        rng_instance, error_function, BETA)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(self.acq_calib_candidates, 
                                            self.acq_calib_true_costs, BETA)
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, ALPHA)
        ehat = final_scores + llambda * error_variances
        # Find the new z
        argmax_index = np.argmax(ehat)
        new_x = self.D_x[argmax_index]
        ehat_value = ehat[argmax_index]

        while ehat_value >= EHAT_THRESHOLD and \
                len(self.T_x[model_idx]) < MAX_NUM_ACQUIRED_POINTS:
            new_x, ehat_value = \
                self.picktolearn_one_iteration(model, error_gp, new_x, 
                                                rng_instance, model_idx)

    def run_validation(self):
        results_dict = {}
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            seed = MULTIPLE_SEED_LIST[i]
            tpr, fpr, tnr, fnr = validate_final_level_set(model, self.validation_candidates, 
                                                        self.validation_true_costs, BETA)
            results_dict[seed] = [tpr, fpr, tnr, fnr]

        keys = list(results_dict.keys())
        all_tprs = [results_dict[key][0] for key in keys]
        all_fnrs = [results_dict[key][3] for key in keys]
        all_fprs = [results_dict[key][1] for key in keys]
        all_tnrs = [results_dict[key][2] for key in keys]

        for key in keys:
            # TPR + FNR = 1
            # FPR + TNR = 1
            print("\nSeed: ", key)
            print("True Positive Rate: ", results_dict[key][0])
            print("False Negative Rate: ", results_dict[key][3])
            print("False Positive Rate: ", results_dict[key][1])
            print("True Negative Rate: ", results_dict[key][2])

        print("\nAverage TPR Over All Seeds: ", np.mean(all_tprs))
        print("Average FNR Over All Seeds: ", np.mean(all_fnrs))
        print("Average FPR Over All Seeds: ", np.mean(all_fprs))
        print("Average TNR Over All Seeds: ", np.mean(all_tnrs))

    
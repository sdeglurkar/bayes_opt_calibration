import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script import env_utils
from env_utils import evaluate_V

from conformal_prediction import get_quantile_for_interval_score_fn
from error_estimate_model import ErrorGP
from helper import *
from main_model import MainGP
from pick_to_learn_settings import *

import GPy
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
from scipy.stats import norm


class PickToLearn():
    def __init__(self):
        self.policy = POLICY
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
        self.seed_list = MULTIPLE_SEED_LIST
        self.llambdas = []

    def setup(self):
        print("Setting up Pick-to-Learn")
        self.D_x, self.D_y = self.get_D()
        self.acq_calib_candidates, self.acq_calib_true_costs = \
            self.get_acquisition_fn_calib_dataset()
        self.error_gp_candidates, self.error_gp_true_costs = \
            self.get_error_gp_dataset()
        self.validation_candidates, self.validation_true_costs = \
            self.get_validation_dataset()
        self.candidates, self.oned_x, self.oned_y = self.get_candidates_helper() 
        if MULTIPLE_SEEDS:
            self.model_list = self.fit_initial_model_multiple_seeds(self.rng_list, 
                                                                    self.seed_list, 
                                                                    self.candidates)
            for i in range(len(self.model_list)):
                model = self.model_list[i]
                # seed = self.seed_list[i]
                # error_gp_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
                #                                     self.error_gp_true_costs, BETA)
                # error_gp = self.fit_initial_error_gp(self.error_gp_candidates, error_gp_ys)
                # self.plot_error_gp(error_gp, self.candidates, self.oned_x, self.oned_y, 
                #                     ERROR_GP_LOGDIR + f'/gp_init_colorbar{seed}.png')
                # self.error_gp_list.append(error_gp)
                self.error_gp_list.append(None)
                self.T_x.append(np.array([[]]))
                self.T_y.append(np.array([[]]))
                self.llambdas.append(np.array([]))
        else:
            model = self.fit_initial_model(RNG, RANDOM_SEED, self.candidates)
            self.model_list = [model]
            # error_gp_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
            #                                         self.error_gp_true_costs, BETA)
            # error_gp = self.fit_initial_error_gp(self.error_gp_candidates, error_gp_ys)
            # self.plot_error_gp(error_gp, self.candidates, self.oned_x, self.oned_y, 
            #                         ERROR_GP_LOGDIR + f'/gp_init_colorbar.png')
            # self.error_gp_list = [error_gp]
            self.error_gp_list = [None]
            self.T_x.append(np.array([[]]))
            self.T_y.append(np.array([[]]))
            self.llambdas.append(np.array([]))
        
        # self.plot_multiple_models(self.model_list, self.seed_list, 
        #                         self.candidates, self.oned_x, self.oned_y)
        
        print("Done with setup!")

    def get_D(self):
        print("Obtaining D")
        if os.path.isfile(f"drone_pickles/D_{DESIRED_N}.pkl"):
            with open(f"drone_pickles/D_{DESIRED_N}.pkl", "rb") as f:
                D_candidates, D_true_costs = pickle.load(f)
        else:
            D_candidates, D_true_costs = \
                get_ground_truths_for_random_points(RANGE_X, RNG, F, DESIRED_N, 
                                                    INPUT_DIM)
            with open(f'drone_pickles/D_{DESIRED_N}.pkl', 'wb') as f:
                pickle.dump([D_candidates, D_true_costs], f)
        N = len(D_candidates)
        assert N == DESIRED_N
        if PLOT_D:
            plt.scatter(D_candidates[:, 0], D_candidates[:, 1])
            plt.show()
        print("Done!")
        
        return D_candidates, D_true_costs

    def get_error_gp_dataset(self):
        print("Obtaining Error GP Dataset")
        if os.path.isfile(f"drone_pickles/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl"):
            with open(f"drone_pickles/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", "rb") as f:
                error_gp_candidates, error_gp_true_costs = pickle.load(f)
        else:
            error_gp_candidates, error_gp_true_costs = \
                get_ground_truths_for_random_points(RANGE_X, RNG, F, 
                                                    NUM_ERROR_GP_POINTS, INPUT_DIM)
            with open(f"drone_pickles/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", 'wb') as f:
                pickle.dump([error_gp_candidates, error_gp_true_costs], f)
        print("Done!")

        return error_gp_candidates, error_gp_true_costs

    def get_acquisition_fn_calib_dataset(self):
        print("Obtaining C")
        if os.path.isfile(f"drone_pickles/acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl"):
            with open(f"drone_pickles/acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl", "rb") as f:
                acq_calib_candidates, acq_calib_true_costs = pickle.load(f)
        else:
            acq_calib_candidates, acq_calib_true_costs = \
                get_ground_truths_for_random_points(RANGE_X, RNG, F, NUM_CALIBRATION_POINTS,
                                                    INPUT_DIM)
            with open(f"drone_pickles/acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl", 'wb') as f:
                pickle.dump([acq_calib_candidates, acq_calib_true_costs], f)
        print("Done!")

        return acq_calib_candidates, acq_calib_true_costs
        
    def get_validation_dataset(self):
        print("Obtaining Validation Dataset")
        if os.path.isfile("drone_pickles/validation_data.pkl"):
            with open("drone_pickles/validation_data.pkl", "rb") as f:
                validation_candidates, validation_true_costs = pickle.load(f)
        else:
            validation_candidates, validation_true_costs, _, _ = \
                get_ground_truths_for_a_grid(F, RANGE_X, VALIDATION_DISCRETIZATION, INPUT_DIM)
            with open('drone_pickles/validation_data.pkl', 'wb') as f:
                pickle.dump([validation_candidates, validation_true_costs], f)
        if PLOT_VALIDATION_DATA:
            plt.scatter(validation_candidates[:, 0], validation_candidates[:, 1])
            plt.show()
        print("Done!")

        return validation_candidates, validation_true_costs
    
    def get_candidates_helper(self, discretization=0.1):
        candidates, _, oned_x, oned_y = get_ground_truths_for_a_grid(F, RANGE_X, 
                                                discretization, INPUT_DIM,
                                                get_costs=False)
        return candidates, oned_x, oned_y

    def fit_initial_model(self, rng_instance, seed_val, candidates):
        print("Fitting Initial Model")
        mean_function = GPy.core.Mapping(2,1)
        mean_function.f = lambda x: evaluate_V(x, self.policy)
        mean_function.update_gradients = lambda a,b: 0
        mean_function.gradients_X = lambda a,b: 0
        np.random.seed(seed_val)
        model = MainGP(F, mean_function, INPUT_DIM, candidates, RANGE_X, 
                        NOISE_VAR, COST_THRES, CONF_THRES, LENGTH_SCALE, logdir=LOGDIR)
        model.initial_setup(NUM_MODEL_INIT_ITERS, rng_instance, to_plot=False)
        print("Done!")

        return model

    def fit_initial_model_multiple_seeds(self, rng_list, seed_list, candidates):
        model_list = []
        for i in range(len(rng_list)):
            rng = rng_list[i]
            seed_val = seed_list[i] 
            model = self.fit_initial_model(rng, seed_val, candidates)
            model_list.append(model)
        return model_list
    
    def plot_model(self, model, candidates, oned_x, oned_y, seed=None, stage='init'):
        if seed is not None:
            fig_name = LOGDIR + f'/gp_{stage}_{seed}.png'
            fig_name_colorbar = LOGDIR + f'/gp_{stage}_colorbar{seed}.png'
            mile_name = LOGDIR + f'/mile_{stage}_{seed}.png'
            plot_main_gp(model, GRID, candidates, oned_x, oned_y, ALL_VALUES, 
                    THETA_INDEX, BETA, fig_name, fig_name_colorbar, mile_name)
        else:
            fig_name = LOGDIR + f'/gp_{stage}.png'
            fig_name_colorbar = LOGDIR + f'/gp_{stage}_colorbar.png'
            mile_name = LOGDIR + f'/mile_{stage}.png'
            plot_main_gp(model, GRID, candidates, oned_x, oned_y, ALL_VALUES, 
                    THETA_INDEX, BETA, fig_name, fig_name_colorbar, mile_name)                                      

    def plot_multiple_models(self, model_list, seed_list, candidates, oned_x, oned_y,
                                    stage='init'):
        for i in range(len(model_list)):
            model = model_list[i]
            seed = seed_list[i]
            self.plot_model(model, candidates, oned_x, oned_y, seed, stage)
    
    def plot_colormap_points(self, points, colors, seed, name, stage):
        plt.figure()
        scatter = plt.scatter(points[:, 0], points[:, 1], c=colors, cmap='viridis')
        plt.colorbar(scatter)
        fig_name = LOGDIR + f'/{name}_{stage}_{seed}.png'
        plt.savefig(fig_name)

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
    
    def picktolearn_one_iteration(self, model, error_gp, new_x, rng_instance, model_idx,
                                        stage):
        if self.T_x[model_idx].size == 0:
            self.T_x[model_idx] = new_x
            self.T_y[model_idx] = model.f(new_x)
            model.acq_cache = new_x
        else: 
            self.T_x[model_idx] = np.append(self.T_x[model_idx], new_x, axis=0)
            self.T_y[model_idx] = np.append(self.T_y[model_idx], model.f(new_x), axis=0)
            model.acq_cache = np.append(model.acq_cache, new_x, axis=0)
        
        model.optimize_given_T(np.array(self.T_x[model_idx]), 
                                np.array(self.T_y[model_idx]))
        remaining = np.array([elem for elem in np.array(self.D_x) if \
                    elem not in self.T_x[model_idx]])
        
        # Now re-fit the error GP
        # error_gp_new_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
        #                                            self.error_gp_true_costs, BETA)
        # print(error_gp_new_ys)
        # scatter = plt.scatter(self.error_gp_candidates[:, 0], 
        #                     self.error_gp_candidates[:, 1], c=error_gp_new_ys, 
        #                     cmap='viridis')
        # plt.colorbar(scatter)
        # plt.show()
        # error_gp.fit(self.error_gp_candidates, error_gp_new_ys)
        # error_function = error_gp.forward()
        
        # Get a_{h,eta}(z) and u_{h,eta}(z)
        error_function = None
        final_scores, error_variances, nan_mask = model.score_function(self.acq_calib_candidates, 
                                        rng_instance, error_function, BETA, t=stage)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(self.acq_calib_candidates, 
                                            self.acq_calib_true_costs, BETA)
        e = e[nan_mask]
        # self.plot_colormap_points(self.acq_calib_candidates[nan_mask], e, 
        #                             self.seed_list[model_idx],
        #                             'calib_true_e', stage)
        # self.plot_colormap_points(self.acq_calib_candidates[nan_mask], final_scores, 
        #                             self.seed_list[model_idx],
        #                             'calib_score_fn', stage)
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, ALPHA)
        self.llambdas[model_idx] = np.append(self.llambdas[model_idx], llambda)
        final_scores, error_variances, nan_mask = model.score_function(remaining, 
                                        rng_instance, error_function, BETA, t=stage)
        ehat = final_scores + llambda * error_variances
        remaining = remaining[nan_mask]
        # print("\n\n\n\n\n\n", final_scores, llambda, error_variances, "\n\n\n\n\n\n\n")
        
        # self.plot_colormap_points(remaining, ehat, self.seed_list[model_idx],
        #                             'score_fn', stage)

        # Find the new z
        argmax_index = np.argmax(ehat)
        new_x = np.expand_dims(remaining[argmax_index], axis=0)
        ehat_value = ehat[argmax_index]
        return new_x, ehat_value
    
    def picktolearn_alg(self, model, error_gp, rng_instance, model_idx):
        # error_function = error_gp.forward()
        error_function = None
        # Get a_{h,eta}(z) and u_{h,eta}(z)
        final_scores, error_variances, nan_mask = model.score_function(self.acq_calib_candidates, 
                                        rng_instance, error_function, BETA, t=0)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(self.acq_calib_candidates, 
                                            self.acq_calib_true_costs, BETA)
        e = e[nan_mask]
        # self.plot_colormap_points(self.acq_calib_candidates[nan_mask], e, 
        #                             self.seed_list[model_idx],
        #                             'calib_true_e', 'init')
        # self.plot_colormap_points(self.acq_calib_candidates[nan_mask], final_scores, 
        #                             self.seed_list[model_idx],
        #                             'calib_score_fn', 'init')
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, ALPHA)
        self.llambdas[model_idx] = np.append(self.llambdas[model_idx], llambda)
        final_scores, error_variances, nan_mask = model.score_function(self.D_x, 
                                        rng_instance, error_function, BETA, t=0)
        ehat = final_scores + llambda * error_variances
        # self.plot_colormap_points(self.D_x[nan_mask], ehat, self.seed_list[model_idx],
        #                         'score_fn', 'init')

        # Find the new z
        argmax_index = np.argmax(ehat)
        new_x = np.expand_dims(self.D_x[argmax_index], axis=0)
        ehat_value = ehat[argmax_index]

        it = 0
        while ehat_value >= EHAT_THRESHOLD and \
                len(self.T_x[model_idx]) < MAX_NUM_ACQUIRED_POINTS:
            new_x, ehat_value = \
                self.picktolearn_one_iteration(model, error_gp, new_x, 
                                                rng_instance, model_idx,
                                                stage=it)
            it += 1
            # self.plot_model(model, self.candidates, self.oned_x, self.oned_y, 
            #                 seed=self.seed_list[model_idx], stage=it)
            
            print("\n\n\n\n\n\nSTAGE", it, "\n\n\n\n\n\n\n")
        
        seed = self.seed_list[model_idx]
        plt.figure()
        plt.plot(range(len(self.llambdas[model_idx])), self.llambdas[model_idx])
        plt.xlabel("Iterations")
        plt.ylabel("Lambda Value")
        plt.savefig(LOGDIR + f"/lambdas_{seed}.png")

    def run_validation(self):
        results_dict = {}
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            seed = self.seed_list[i]
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

    
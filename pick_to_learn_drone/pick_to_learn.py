import sys
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')
from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import evaluate_V_batch
from experiment_script.lin_scenario_opt import solve_iterative_method, visualize_set

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
import time


class PickToLearn():
    def __init__(self):
        self.policy = POLICY
        self.T_x = []
        self.T_y = []
        self.D_x = None 
        # self.D_y = None
        self.full_D_x = None  
        self.acq_calib_candidates = None 
        self.acq_calib_true_costs = None
        self.full_acq_calib_candidates = None  
        # self.error_gp_candidates = None 
        # self.error_gp_true_costs = None 
        # self.full_error_gp_candidates = None
        self.validation_candidates = None
        self.validation_true_costs = None 
        self.full_validation_candidates = None
        self.model_list = []
        self.error_gp_list = []
        self.rng_list = MULTIPLE_RNG_LIST
        self.seed_list = MULTIPLE_SEED_LIST
        self.llambdas = []
        self.state_expander = expand_state_based_on_model_dim(EGO_SETTING, 
                                                        ADVERSARY_SETTING,
                                                        INPUT_DIM)
        self.method_times = []

    def setup(self):
        print("\nSetting up Pick-to-Learn")
        t0 = time.time()
        # self.D_x, self.D_y, self.full_D_x = self.get_D()
        # self.acq_calib_candidates, self.acq_calib_true_costs, self.full_acq_calib_candidates = \
        #     self.get_acquisition_fn_calib_dataset()
        # self.error_gp_candidates, self.error_gp_true_costs, self.full_error_gp_candidates = \
        #     self.get_error_gp_dataset()  # Not currently used
        self.validation_candidates, self.validation_true_costs, self.full_validation_candidates, \
            self.learned_V = self.get_validation_dataset()
        self.candidates, self.oned_x, self.oned_y, self.full_candidates = \
            self.get_candidates_helper()  # May not be very necessary
        t1 = time.time()

        if MULTIPLE_SEEDS:
            self.albert_alphas = []
            self.albert_num_samples = []
            self.albert_times = []
            for i in range(len(self.rng_list)):
                print("\nRunning Albert's method!")
                rng = self.rng_list[i]
                alpha, total_time, _, total_num_samples = \
                    solve_iterative_method(ENV, ALBERT_EPS, ALBERT_DELT, ALBERT_M, 
                                            HORIZON, self.policy, INPUT_DIM, ARGS, 
                                            rng, RANGE_X, EGO_SETTING, 
                                            ADVERSARY_SETTING, alpha_init=-np.inf)
                self.albert_alphas.append(alpha)
                self.albert_num_samples.append(total_num_samples)
                self.albert_times.append(total_time)
            for i in range(len(self.rng_list)):
                # model = self.model_list[i]
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
            
            self.model_list, ttimes1 = \
                    self.fit_initial_model_multiple_seeds(self.rng_list, 
                                                        self.seed_list, 
                                                        self.candidates)
            self.D_C_list, ttimes2 = self.get_D_C_multiple_models(self.rng_list)
            self.method_times = [t2 + t3 + (t1-t0) for (t2, t3) in zip(ttimes1, ttimes2)]
        else:
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

            print("\nRunning Albert's method!")
            alpha, total_time, _, total_num_samples = \
                solve_iterative_method(ENV, ALBERT_EPS, ALBERT_DELT, 
                                        ALBERT_M, HORIZON, self.policy, 
                                        INPUT_DIM, ARGS, RNG, RANGE_X,
                                        EGO_SETTING, ADVERSARY_SETTING,
                                        alpha_init=-np.inf)
            self.albert_alphas = [alpha]
            self.albert_num_samples = [total_num_samples]
            self.albert_times = [total_time]
            
            model, ttime1 = self.fit_initial_model(RNG, RANDOM_SEED, self.candidates)
            self.D_C_list, ttime2 = self.get_D_C_multiple_models([RNG])
            self.model_list = [model]
            self.method_times = [ttime1 + ttime2[0] + (t1-t0)]

        self.plot_multiple_models(self.learned_V, self.model_list, self.seed_list,  
                                    self.oned_x, self.oned_y, self.albert_alphas)

        print("Done with setup!")

    def get_D(self, rng_instance):
        print("Obtaining D")
        D_candidates, D_true_costs, full_D_candidates = \
                get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
                                                    ADVERSARY_SETTING, rng_instance, F, 
                                                    DESIRED_N, INPUT_DIM,
                                                    get_costs=False)
        # if os.path.isfile(f"drone_pickles_{INPUT_DIM}D/D_{DESIRED_N}.pkl"):
        #     with open(f"drone_pickles_{INPUT_DIM}D/D_{DESIRED_N}.pkl", "rb") as f:
        #         D_candidates, D_true_costs, full_D_candidates = pickle.load(f)
        # else:
        #     D_candidates, D_true_costs, full_D_candidates = \
        #         get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
        #                                             ADVERSARY_SETTING, RNG, F, 
        #                                             DESIRED_N, INPUT_DIM)
        #     with open(f'drone_pickles_{INPUT_DIM}D/D_{DESIRED_N}.pkl', 'wb') as f:
        #         pickle.dump([D_candidates, D_true_costs, full_D_candidates], f)
        N = len(D_candidates)
        assert N == DESIRED_N
        print("Done!")
        
        return D_candidates, D_true_costs, full_D_candidates

    def get_error_gp_dataset(self, rng_instance):
        print("Obtaining Error GP Dataset")
        error_gp_candidates, error_gp_true_costs, full_error_gp_candidates = \
                get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
                                                    ADVERSARY_SETTING, rng_instance, F, 
                                                    NUM_ERROR_GP_POINTS, INPUT_DIM)

        # if os.path.isfile(f"drone_pickles_{INPUT_DIM}D/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl"):
        #     with open(f"drone_pickles_{INPUT_DIM}D/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", "rb") as f:
        #         error_gp_candidates, error_gp_true_costs, full_error_gp_candidates = \
        #             pickle.load(f)
        # else:
        #     error_gp_candidates, error_gp_true_costs, full_error_gp_candidates = \
        #         get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
        #                                             ADVERSARY_SETTING, RNG, F, 
        #                                             NUM_ERROR_GP_POINTS, INPUT_DIM)
        #     with open(f"drone_pickles_{INPUT_DIM}D/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", 'wb') as f:
        #         pickle.dump([error_gp_candidates, error_gp_true_costs, \
        #                     full_error_gp_candidates], f)
        print("Done!")

        return error_gp_candidates, error_gp_true_costs, full_error_gp_candidates
    
    def get_initial_gp_dataset(self, rng_instance):
        print("Obtaining Initial GP Dataset")
        initial_gp_candidates, initial_gp_true_costs, full_initial_gp_candidates = \
            get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
                                                ADVERSARY_SETTING, rng_instance, F, 
                                                NUM_MODEL_INIT_ITERS, INPUT_DIM)
        print("Done!")

        return initial_gp_candidates, initial_gp_true_costs, full_initial_gp_candidates

    def get_acquisition_fn_calib_dataset(self, rng_instance):
        print("Obtaining C")
        acq_calib_candidates, acq_calib_true_costs, full_acq_calib_candidates = \
                get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
                                                    ADVERSARY_SETTING, rng_instance, F, 
                                                    NUM_CALIBRATION_POINTS,
                                                    INPUT_DIM)
        # if os.path.isfile(f"drone_pickles_{INPUT_DIM}D/acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl"):
        #     with open(f"drone_pickles_{INPUT_DIM}D/acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl", "rb") as f:
        #         acq_calib_candidates, acq_calib_true_costs, full_acq_calib_candidates = \
        #             pickle.load(f)
        # else:
        #     acq_calib_candidates, acq_calib_true_costs, full_acq_calib_candidates = \
        #         get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
        #                                             ADVERSARY_SETTING, RNG, F, 
        #                                             NUM_CALIBRATION_POINTS,
        #                                             INPUT_DIM)
        #     with open(f"drone_pickles_{INPUT_DIM}D/acq_calibration_data_{NUM_CALIBRATION_POINTS}.pkl", 'wb') as f:
        #         pickle.dump([acq_calib_candidates, acq_calib_true_costs, \
        #                     full_acq_calib_candidates], f)
        print("Done!")

        return acq_calib_candidates, acq_calib_true_costs, full_acq_calib_candidates
    
    def get_D_C_multiple_models(self, rng_list):
        ttimes = []
        D_C_list = []
        for rng in rng_list:
            t0 = time.time()
            D_candidates, D_true_costs, full_D_candidates = self.get_D(rng)
            acq_calib_candidates, acq_calib_true_costs, full_acq_calib_candidates = \
                self.get_acquisition_fn_calib_dataset(rng)
            dict_elem = {'D':[D_candidates, D_true_costs, full_D_candidates],
                        'C': [acq_calib_candidates, acq_calib_true_costs, \
                                full_acq_calib_candidates]}
            D_C_list.append(dict_elem)
            t1 = time.time()
            ttimes.append(t1-t0)
        return D_C_list, ttimes

    def get_validation_dataset(self):
        print("Obtaining Validation Dataset")
        if os.path.isfile(f"{VALIDATION_LOGDIR}/validation_data.pkl"):
            with open(f"{VALIDATION_LOGDIR}/validation_data.pkl", "rb") as f:
                validation_candidates, validation_true_costs, full_validation_candidates, \
                learned_V = \
                    pickle.load(f)
        else:
            validation_candidates, validation_true_costs, _, _, full_validation_candidates, \
            learned_V = \
                get_ground_truths_for_a_grid(RANGE_X, EGO_SETTING, ADVERSARY_SETTING, 
                                            F, VALIDATION_DISCRETIZATION, INPUT_DIM,
                                            self.policy,
                                            get_learned_V=True)
            with open(f'{VALIDATION_LOGDIR}/validation_data.pkl', 'wb') as f:
                pickle.dump([validation_candidates, validation_true_costs, \
                            full_validation_candidates, learned_V], f)
        print("Done!")

        return validation_candidates, validation_true_costs, full_validation_candidates, \
                learned_V
    
    def get_candidates_helper(self):
        candidates, _, oned_x, oned_y, full_candidates, _ = \
                                                get_ground_truths_for_a_grid(RANGE_X, 
                                                EGO_SETTING, ADVERSARY_SETTING, F,
                                                [MODEL_CANDIDATES_DISCRETIZATION], 
                                                INPUT_DIM, self.policy,
                                                get_costs=False)
        return candidates, oned_x, oned_y, full_candidates

    def fit_initial_model(self, rng_instance, seed_val, candidates):
        print("Fitting Initial Model")
        t0 = time.time()
        initial_gp_candidates, initial_gp_true_costs, full_initial_gp_candidates = \
            self.get_initial_gp_dataset(rng_instance)

        mean_function = GPy.core.Mapping(INPUT_DIM,1)
        mean_function.f = lambda x: evaluate_V_batch(self.state_expander(x), self.policy)
        mean_function.update_gradients = lambda a,b: 0
        mean_function.gradients_X = lambda a,b: 0
        np.random.seed(seed_val)
        model = MainGP(F, mean_function, INPUT_DIM, candidates, RANGE_X, 
                        NOISE_VAR, COST_THRES, CONF_THRES, LENGTH_SCALE, logdir=LOGDIR)
        # model.initial_setup(NUM_MODEL_INIT_ITERS, rng_instance, self.state_expander, 
        #                     to_plot=False)
        model.initial_setup_given_points(initial_gp_candidates, 
                                initial_gp_true_costs, to_plot=False)
        t1 = time.time()
        print("Done!")

        return model, t1-t0

    def fit_initial_model_multiple_seeds(self, rng_list, seed_list, candidates):
        model_list = []
        times = []
        for i in range(len(rng_list)):
            rng = rng_list[i]
            seed_val = seed_list[i] 
            model, ttime = self.fit_initial_model(rng, seed_val, candidates)
            model_list.append(model)
            times.append(ttime)
        return model_list, times

    def plot_model(self, model, learned_V, oned_x, oned_y, albert_alpha,
                    seed=None, stage='init'):
        if seed is not None:
            fig_name = LOGDIR + f'/gp_{stage}_{seed}.png'
            fig_name_colorbar = LOGDIR + f'/gp_{stage}_colorbar{seed}.png'
            plot_main_gp(learned_V, BETA, oned_x, oned_y,  
                albert_alpha, model, INPUT_DIM, EGO_SETTING, ADVERSARY_SETTING, 
                RANGE_X, VALIDATION_DISCRETIZATION, [MODEL_CANDIDATES_DISCRETIZATION],
                self.state_expander, fig_name, fig_name_colorbar)
        else:
            fig_name = LOGDIR + f'/gp_{stage}.png'
            fig_name_colorbar = LOGDIR + f'/gp_{stage}_colorbar.png'
            plot_main_gp(learned_V, BETA, oned_x, oned_y,  
                albert_alpha, model, INPUT_DIM, EGO_SETTING, ADVERSARY_SETTING,
                RANGE_X, VALIDATION_DISCRETIZATION, [MODEL_CANDIDATES_DISCRETIZATION],
                self.state_expander, fig_name, fig_name_colorbar)                                     

    def plot_multiple_models(self, learned_V, model_list, seed_list,  
                                oned_x, oned_y, albert_alphas, stage='init'):
        for i in range(len(model_list)):
            model = model_list[i]
            seed = seed_list[i]
            self.plot_model(model, learned_V, oned_x, oned_y, albert_alphas[i], 
                            seed, stage)
    
    def plot_colormap_points(self, points, colors, seed, name, stage):
        plt.figure()
        # Plot x and y only
        if INPUT_DIM == 2 or INPUT_DIM == 3:
            scatter = plt.scatter(points[:, 0], points[:, 1], c=colors, cmap='viridis')
        elif INPUT_DIM == 4 or INPUT_DIM == 6 or INPUT_DIM == 12:
            scatter = plt.scatter(points[:, 0], points[:, 2], c=colors, cmap='viridis')
        else:
            raise NotImplementedError
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
        tot_time = 0
        t0 = time.time()
       
        D_x = self.D_C_list[model_idx]['D'][0]
        C_x = self.D_C_list[model_idx]['C'][0]
        C_costs = self.D_C_list[model_idx]['C'][1]

        if self.T_x[model_idx].size == 0:
            self.T_x[model_idx] = new_x
            self.T_y[model_idx] = model.f(self.state_expander(new_x))
            model.acq_cache = new_x
        else: 
            self.T_x[model_idx] = np.append(self.T_x[model_idx], new_x, axis=0)
            self.T_y[model_idx] = np.append(self.T_y[model_idx], \
                                            model.f(self.state_expander(new_x)), axis=0)
            model.acq_cache = np.append(model.acq_cache, new_x, axis=0)
        
        model.optimize_given_T(np.array(self.T_x[model_idx]), 
                                np.array(self.T_y[model_idx]))
        remaining = np.array([elem for elem in np.array(D_x) if \
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
        final_scores, error_variances, nan_mask = model.score_function(C_x, 
                                        rng_instance, error_function, BETA, t=stage,
                                        decay_factor=DECAY_RATE)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(C_x, C_costs, BETA)
        e = e[nan_mask]
        t1 = time.time()
        self.plot_colormap_points(C_x[nan_mask], e, 
                                    self.seed_list[model_idx],
                                    'calib_true_e', stage)
        self.plot_colormap_points(C_x[nan_mask], final_scores, 
                                    self.seed_list[model_idx],
                                    'calib_score_fn', stage)
        t2 = time.time()
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, ALPHA)
        self.llambdas[model_idx] = np.append(self.llambdas[model_idx], llambda)
        final_scores, error_variances, nan_mask = model.score_function(remaining, 
                                        rng_instance, error_function, BETA, t=stage,
                                        decay_factor=DECAY_RATE)
        ehat = final_scores + llambda * error_variances
        remaining = remaining[nan_mask]
        # print("\n\n\n\n\n\n", final_scores, llambda, error_variances, "\n\n\n\n\n\n\n")
        
        t3 = time.time()
        self.plot_colormap_points(remaining, ehat, self.seed_list[model_idx],
                                    'score_fn', stage)

        # Find the new z
        t4 = time.time()
        argmax_index = np.argmax(ehat)
        new_x = np.expand_dims(remaining[argmax_index], axis=0)
        ehat_value = ehat[argmax_index]
        t5 = time.time()

        tot_time += (t1-t0) + (t3-t2) + (t5-t4)
        return new_x, ehat_value, tot_time
    
    def picktolearn_alg(self, model, error_gp, rng_instance, model_idx):
        tot_time = 0
        t0 = time.time()

        D_x = self.D_C_list[model_idx]['D'][0]
        C_x = self.D_C_list[model_idx]['C'][0]
        C_costs = self.D_C_list[model_idx]['C'][1]

        # error_function = error_gp.forward()
        error_function = None
        # Get a_{h,eta}(z) and u_{h,eta}(z)
        final_scores, error_variances, nan_mask = model.score_function(C_x, 
                                        rng_instance, error_function, BETA, t=0,
                                        decay_factor=DECAY_RATE)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(C_x, C_costs, BETA)
        e = e[nan_mask]
        t1 = time.time()
        self.plot_colormap_points(C_x[nan_mask], e, 
                                    self.seed_list[model_idx],
                                    'calib_true_e', 'init')
        self.plot_colormap_points(C_x[nan_mask], final_scores, 
                                    self.seed_list[model_idx],
                                    'calib_score_fn', 'init')
        t2 = time.time()
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, ALPHA)
        self.llambdas[model_idx] = np.append(self.llambdas[model_idx], llambda)
        final_scores, error_variances, nan_mask = model.score_function(D_x, 
                                        rng_instance, error_function, BETA, t=0,
                                        decay_factor=DECAY_RATE)
        ehat = final_scores + llambda * error_variances
        t3 = time.time()
        self.plot_colormap_points(D_x[nan_mask], ehat, self.seed_list[model_idx],
                                'score_fn', 'init')

        # Find the new z
        t4 = time.time()
        argmax_index = np.argmax(ehat)
        new_x = np.expand_dims(D_x[argmax_index], axis=0)
        ehat_value = ehat[argmax_index]
        t5 = time.time()

        tot_time += (t1-t0) + (t3-t2) + (t5-t4)

        it = 0
        while ehat_value >= EHAT_THRESHOLD and \
                len(self.T_x[model_idx]) < MAX_NUM_ACQUIRED_POINTS:
            new_x, ehat_value, ttime = \
                self.picktolearn_one_iteration(model, error_gp, new_x, 
                                                rng_instance, model_idx,
                                                stage=it)
            tot_time += ttime
            it += 1
            self.plot_model(model, self.learned_V, self.oned_x, self.oned_y, 
                            self.albert_alphas[model_idx], 
                            seed=self.seed_list[model_idx], stage=it)
            
            print("\n\n\n\n\n\nITERATION", it, "\n\n\n\n\n\n\n")
        
        seed = self.seed_list[model_idx]
        plt.figure()
        plt.plot(range(len(self.llambdas[model_idx])), self.llambdas[model_idx])
        plt.xlabel("Iterations")
        plt.ylabel("Lambda Value")
        plt.savefig(LOGDIR + f"/lambdas_{seed}.png")

        self.method_times[model_idx] += tot_time

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

        print("\nMethod total times and mean", self.method_times, \
                                            np.mean(self.method_times))

    def validate_albert_method(self):
        results_dict = {}
        if MULTIPLE_SEEDS:
            for i in range(len(self.rng_list)):
                seed = self.seed_list[i]
                tpr, fpr, tnr, fnr = validate_albert(self.policy, self.albert_alphas[i], 
                                                        self.validation_candidates, 
                                                        self.validation_true_costs,
                                                        self.state_expander)
                results_dict[seed] = [tpr, fpr, tnr, fnr]
        else:
            tpr, fpr, tnr, fnr = validate_albert(self.policy, self.albert_alphas[0], 
                                                    self.validation_candidates, 
                                                    self.validation_true_costs,
                                                    self.state_expander)
            results_dict[RANDOM_SEED] = [tpr, fpr, tnr, fnr]

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

        print("\nAlbert alphas and mean", self.albert_alphas, \
                                            np.mean(self.albert_alphas))
        print("Albert total times and mean", self.albert_times, \
                                            np.mean(self.albert_times))
        print("Albert total num samples and mean", self.albert_num_samples, \
                                            np.mean(self.albert_num_samples))
import sys
import os 
sys.path.append(os.getcwd())

from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import evaluate_V_batch
from experiment_script.lin_scenario_opt import solve_iterative_method, robust_scenario_opt

from conformal_prediction import get_quantile_for_interval_score_fn
from helper import *
from main_model import MainGP
from pick_to_learn_settings import *

import GPy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


class PickToLearn():
    def __init__(self, args):
        self.policy = args.policy
        self.T_x = []
        self.T_y = []
        self.D_x = None 
        # self.D_y = None
        self.full_D_x = None  
        self.acq_calib_candidates = None 
        self.acq_calib_true_costs = None
        self.full_acq_calib_candidates = None  
        self.validation_candidates = None
        self.validation_true_costs = None 
        self.full_validation_candidates = None
        self.model_list = []
        self.rng_list = MULTIPLE_RNG_LIST
        self.seed_list = MULTIPLE_SEED_LIST
        self.llambdas = []
        self.state_expander = expand_state_based_on_model_dim(args.ego_setting, 
                                                        args.adversary_setting,
                                                        args.input_dim)
        self.method_times = []
        if args.random_active_learning:
            self.model_score_fn_weights = [0.0, 0.0, 1.0]
        else:
            self.model_score_fn_weights = [1.0, 0.0, 0.001]
        
        self.args = args

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

        if self.args.multiple_seeds:
            self.albert_levels = []
            self.albert_num_samples = []
            self.albert_times = []
            self.robust_albert_arrays = []
            self.seed_failed = [False for _ in self.seed_list]
            for i in range(len(self.rng_list)):
                print("\nRunning Albert's methods!")
                rng = self.rng_list[i]
                level, total_time, _, total_num_samples, total_num_safety_violations = \
                    solve_iterative_method(self.args.env, self.args.baseline_epsilon, ALBERT_DELT, 
                                            ALBERT_M, self.args.horizon, self.policy, 
                                            self.args.input_dim, self.args, rng, RANGE_X, 
                                            self.args.ego_setting, self.args.adversary_setting, 
                                            alpha_init=-np.inf)
                self.albert_levels.append(level)
                self.albert_num_samples.append(total_num_samples)
                self.albert_times.append(total_time)
                robust_arr = np.zeros((len(ROBUST_ALBERT_N_SWEEP), \
                                        len(ROBUST_ALBERT_LEVEL_SWEEP), 3))
                for m in range(len(ROBUST_ALBERT_N_SWEEP)):
                    for n in range(len(ROBUST_ALBERT_LEVEL_SWEEP)):
                        desired_albert_N = ROBUST_ALBERT_N_SWEEP[m] 
                        desired_albert_level = ROBUST_ALBERT_LEVEL_SWEEP[n]
                        robust_eps, num_samples, num_safety_violations = \
                                robust_scenario_opt(
                                desired_albert_N, desired_albert_level, ALBERT_DELT, 
                                self.policy, self.args.input_dim, self.args.env, self.args, rng, 
                                self.args.horizon, RANGE_X, self.args.ego_setting, 
                                self.args.adversary_setting)
                        robust_arr[m, n, :] = np.array([robust_eps, num_samples, \
                                                            num_safety_violations])
                self.robust_albert_arrays.append(robust_arr)
            for i in range(len(self.rng_list)):
                # model = self.model_list[i]
                # seed = self.seed_list[i]
                # error_gp_ys = model.get_error_of_model_for_points(self.error_gp_candidates, \
                #                                     self.error_gp_true_costs, BETA)
                # error_gp = self.fit_initial_error_gp(self.error_gp_candidates, error_gp_ys)
                # self.plot_error_gp(error_gp, self.candidates, self.oned_x, self.oned_y, 
                #                     ERROR_GP_LOGDIR + f'/gp_init_colorbar{seed}.png')
                # self.error_gp_list.append(error_gp)
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
            self.T_x.append(np.array([[]]))
            self.T_y.append(np.array([[]]))
            self.llambdas.append(np.array([]))
            self.seed_failed = [False]

            print("\nRunning Albert's methods!")
            level, total_time, _, total_num_samples, total_num_safety_violations = \
                solve_iterative_method(self.args.env, self.args.baseline_epsilon, ALBERT_DELT, 
                                        ALBERT_M, self.args.horizon, self.policy, 
                                        self.args.input_dim, self.args, self.args.rng, RANGE_X,
                                        self.args.ego_setting, self.args.adversary_setting,
                                        alpha_init=-np.inf)
            self.albert_levels = [level]
            self.albert_num_samples = [total_num_samples]
            self.albert_times = [total_time]

            self.robust_albert_arrays = []
            robust_arr = np.zeros((len(ROBUST_ALBERT_N_SWEEP), \
                                        len(ROBUST_ALBERT_LEVEL_SWEEP), 3))
            for m in range(len(ROBUST_ALBERT_N_SWEEP)):
                for n in range(len(ROBUST_ALBERT_LEVEL_SWEEP)):
                    desired_albert_N = ROBUST_ALBERT_N_SWEEP[m] 
                    desired_albert_level = ROBUST_ALBERT_LEVEL_SWEEP[n]
                    robust_eps, num_samples, num_safety_violations = \
                            robust_scenario_opt(
                            desired_albert_N, desired_albert_level, ALBERT_DELT, 
                            self.policy, self.args.input_dim, self.args.env, self.args, 
                            self.args.rng, 
                            self.args.horizon, RANGE_X, self.args.ego_setting, 
                            self.args.adversary_setting)
                    robust_arr[m, n, :] = np.array([robust_eps, num_samples, \
                                                        num_safety_violations])
            self.robust_albert_arrays.append(robust_arr)
            
            model, ttime1 = self.fit_initial_model(self.args.rng, self.args.random_seed, \
                                                    self.candidates)
            self.D_C_list, ttime2 = self.get_D_C_multiple_models([self.args.rng])
            self.model_list = [model]
            self.method_times = [ttime1 + ttime2[0] + (t1-t0)]

        self.plot_multiple_models(self.learned_V, self.model_list, self.seed_list,  
                                    self.oned_x, self.oned_y, self.albert_levels)

        print("Done with setup!")

    def get_D(self, rng_instance):
        print("Obtaining D")
        D_candidates, D_true_costs, full_D_candidates = \
                get_ground_truths_for_random_points(RANGE_X, self.args.ego_setting, 
                                                    self.args.adversary_setting, rng_instance, 
                                                    self.args.f, 
                                                    self.args.desired_N, self.args.input_dim,
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

    # def get_error_gp_dataset(self, rng_instance):
    #     print("Obtaining Error GP Dataset")
    #     error_gp_candidates, error_gp_true_costs, full_error_gp_candidates = \
    #             get_ground_truths_for_random_points(RANGE_X, self.args.ego_setting, 
    #                                                 self.args.adversary_setting, rng_instance, 
    #                                                 self.args.f, 
    #                                                 NUM_ERROR_GP_POINTS, self.args.input_dim)

    #     # if os.path.isfile(f"drone_pickles_{INPUT_DIM}D/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl"):
    #     #     with open(f"drone_pickles_{INPUT_DIM}D/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", "rb") as f:
    #     #         error_gp_candidates, error_gp_true_costs, full_error_gp_candidates = \
    #     #             pickle.load(f)
    #     # else:
    #     #     error_gp_candidates, error_gp_true_costs, full_error_gp_candidates = \
    #     #         get_ground_truths_for_random_points(RANGE_X, EGO_SETTING, 
    #     #                                             ADVERSARY_SETTING, RNG, F, 
    #     #                                             NUM_ERROR_GP_POINTS, INPUT_DIM)
    #     #     with open(f"drone_pickles_{INPUT_DIM}D/error_gp_data_{NUM_ERROR_GP_POINTS}.pkl", 'wb') as f:
    #     #         pickle.dump([error_gp_candidates, error_gp_true_costs, \
    #     #                     full_error_gp_candidates], f)
    #     print("Done!")

    #     return error_gp_candidates, error_gp_true_costs, full_error_gp_candidates
    
    def get_initial_gp_dataset(self, rng_instance):
        print("Obtaining Initial GP Dataset")
        initial_gp_candidates, initial_gp_true_costs, full_initial_gp_candidates = \
            get_ground_truths_for_random_points(RANGE_X, self.args.ego_setting, 
                                                self.args.adversary_setting, rng_instance, 
                                                self.args.f, 
                                                self.args.num_model_init_iters, 
                                                self.args.input_dim)
        print("Done!")

        return initial_gp_candidates, initial_gp_true_costs, full_initial_gp_candidates

    def get_acquisition_fn_calib_dataset(self, rng_instance):
        print("Obtaining C")
        acq_calib_candidates, acq_calib_true_costs, full_acq_calib_candidates = \
                get_ground_truths_for_random_points(RANGE_X, self.args.ego_setting, 
                                                    self.args.adversary_setting, rng_instance, 
                                                    self.args.f, 
                                                    self.args.num_calibration_points,
                                                    self.args.input_dim)
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
        if os.path.isfile(f"{self.args.picktolearn_validation_logdir}/validation_data.pkl"):
            with open(f"{self.args.picktolearn_validation_logdir}/validation_data.pkl", "rb") as f:
                validation_candidates, validation_true_costs, full_validation_candidates, \
                learned_V = \
                    pickle.load(f)
        else:
            validation_candidates, validation_true_costs, _, _, full_validation_candidates, \
            learned_V = \
                get_ground_truths_for_a_grid(RANGE_X, self.args.ego_setting, 
                                            self.args.adversary_setting, 
                                            self.args.f, VALIDATION_DISCRETIZATION, 
                                            self.args.input_dim,
                                            self.policy,
                                            get_learned_V=True)
            # # print(validation_candidates, validation_true_costs)
            # lgdir = 'drone_pickles_' + str(INPUT_DIM) + 'D_basicslice_boundaryacq_N4000_init40_decay0.95thres0.3_alpha0.05_tolalpha0.03'
            # with open(f"{lgdir}/validation_data.pkl", "rb") as f:
            #     other_validation_candidates, other_validation_true_costs, _, \
            #     other_learned_V = \
            #         pickle.load(f)
            # differing_inds = np.where(validation_true_costs != other_validation_true_costs)[0]
            # print(differing_inds)
            # print(validation_true_costs[differing_inds[0]], other_validation_true_costs[differing_inds[0]])
            # print(validation_candidates[differing_inds[0]], other_validation_candidates[differing_inds[0]])
            # print(validation_true_costs[differing_inds[1:4]], other_validation_true_costs[differing_inds[1:4]])
            # print(validation_candidates[differing_inds[1:4]], other_validation_candidates[differing_inds[1:4]])
            # exit()
            with open(f'{self.args.picktolearn_validation_logdir}/validation_data.pkl', 'wb') as f:
                pickle.dump([validation_candidates, validation_true_costs, \
                            full_validation_candidates, learned_V], f)
        print("Done!")

        return validation_candidates, validation_true_costs, full_validation_candidates, \
                learned_V
    
    def get_candidates_helper(self):
        candidates, _, oned_x, oned_y, full_candidates, _ = \
                                                get_ground_truths_for_a_grid(RANGE_X, 
                                                self.args.ego_setting, self.args.adversary_setting, 
                                                self.args.f,
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
        model = MainGP(self.args.f, mean_function, self.args.input_dim, candidates, RANGE_X, 
                        NOISE_VAR, COST_THRES, CONF_THRES, LENGTH_SCALE, 
                        logdir=self.args.picktolearn_logdir)
        # model.initial_setup(NUM_MODEL_INIT_ITERS, rng_instance, self.state_expander, 
        #                     to_plot=False)
        model.initial_setup_given_points(initial_gp_candidates, 
                                initial_gp_true_costs, seed_val, to_plot=False)
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

    def plot_model(self, model, learned_V, oned_x, oned_y, albert_level,
                    seed=None, stage='init'):
        if seed is not None:
            fig_name = self.args.picktolearn_logdir + f'/gp_{stage}_{seed}.png'
            fig_name_colorbar = self.args.picktolearn_logdir + f'/gp_{stage}_colorbar{seed}.png'
            plot_main_gp(learned_V, BETA, oned_x, oned_y,  
                albert_level, model, self.args.input_dim, self.args.ego_setting, 
                self.args.adversary_setting, 
                RANGE_X, VALIDATION_DISCRETIZATION, [MODEL_CANDIDATES_DISCRETIZATION],
                self.state_expander, fig_name, fig_name_colorbar, FONTSIZE,
                stage)
        else:
            fig_name = self.args.picktolearn_logdir + f'/gp_{stage}.png'
            fig_name_colorbar = self.args.picktolearn_logdir + f'/gp_{stage}_colorbar.png'
            plot_main_gp(learned_V, BETA, oned_x, oned_y,  
                albert_level, model, self.args.input_dim, self.args.ego_setting, 
                self.args.adversary_setting,
                RANGE_X, VALIDATION_DISCRETIZATION, [MODEL_CANDIDATES_DISCRETIZATION],
                self.state_expander, fig_name, fig_name_colorbar, FONTSIZE,
                stage)                                     

    def plot_multiple_models(self, learned_V, model_list, seed_list,  
                                oned_x, oned_y, albert_levels, stage='init'):
        for i in range(len(model_list)):
            model = model_list[i]
            seed = seed_list[i]
            self.plot_model(model, learned_V, oned_x, oned_y, albert_levels[i], 
                            seed, stage)
    
    def plot_colormap_points(self, points, colors, seed, name, stage):
        plt.figure(figsize=(8,4))
        # Plot x and y only
        if self.args.input_dim == 2 or self.args.input_dim == 3:
            scatter = plt.scatter(points[:, 0], points[:, 1], c=colors, cmap='viridis')
        elif self.args.input_dim == 4 or self.args.input_dim == 6 or self.args.input_dim == 12:
            scatter = plt.scatter(points[:, 0], points[:, 2], c=colors, cmap='viridis')
        else:
            raise NotImplementedError
        cbar = plt.colorbar(scatter)
        cbar.ax.tick_params(labelsize=FONTSIZE)
        fig_name = self.args.picktolearn_logdir + f'/{name}_{stage}_{seed}.png'
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=1000)

    # def fit_initial_error_gp(self, X, Y):
    #     mean_function = None 
    #     error_gp = ErrorGP(F, mean_function, INPUT_DIM, RANGE_X, NOISE_VAR, 
    #                     LENGTH_SCALE, ERROR_GP_LOGDIR)
    #     error_gp.fit(X, Y)
    #     return error_gp 
    
    # def plot_error_gp(self, error_gp, candidates, oned_x, oned_y, fig_name):
    #     mu, _ = error_gp.m.predict(candidates, full_cov=False)
    #     mu = mu.reshape(len(oned_x), len(oned_y))
    #     plt.figure()
    #     plt.contourf(oned_x,
    #                 oned_y,
    #                 mu)
    #     plt.colorbar()
    #     plt.savefig(fig_name)
    
    def picktolearn_one_iteration(self, model, new_x, rng_instance, model_idx,
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
                                np.array(self.T_y[model_idx]), 
                                self.seed_list[model_idx])
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
        final_scores, error_variances, nan_mask = model.score_function(C_x, 
                                        rng_instance, BETA, t=stage,
                                        decay_factor=self.args.decay_rate,
                                        weights=self.model_score_fn_weights)
        # Run conformal prediction to get ehat
        e = model.get_error_of_model_for_points(C_x, C_costs, BETA)
        e = e[nan_mask]
        t1 = time.time()
        if self.args.plot_during:
            self.plot_colormap_points(C_x[nan_mask], e, 
                                        self.seed_list[model_idx],
                                        'calib_true_e', stage)
            self.plot_colormap_points(C_x[nan_mask], final_scores, 
                                        self.seed_list[model_idx],
                                        'calib_score_fn', stage)
        t2 = time.time()
        llambda = get_quantile_for_interval_score_fn(final_scores, error_variances, 
                                                        e, self.args.alpha)
        self.llambdas[model_idx] = np.append(self.llambdas[model_idx], llambda)
        final_scores, error_variances, nan_mask = model.score_function(remaining, 
                                        rng_instance, BETA, t=stage,
                                        decay_factor=self.args.decay_rate,
                                        weights=self.model_score_fn_weights)
        ehat = final_scores + llambda * error_variances
        remaining = remaining[nan_mask]
        # print("\n\n\n\n\n\n", final_scores, llambda, error_variances, "\n\n\n\n\n\n\n")
        
        t3 = time.time()
        if self.args.plot_during:
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
    
    def picktolearn_alg(self, model, rng_instance, model_idx):
        tot_time = 0
        t0 = time.time()

        D_x = self.D_C_list[model_idx]['D'][0]
        C_x = self.D_C_list[model_idx]['C'][0]
        C_costs = self.D_C_list[model_idx]['C'][1]

        # error_function = error_gp.forward()
        # Get a_{h,eta}(z) and u_{h,eta}(z)
        final_scores, error_variances, nan_mask = model.score_function(C_x, 
                                        rng_instance, BETA, t=0,
                                        decay_factor=self.args.decay_rate,
                                        weights=self.model_score_fn_weights)
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
                                        rng_instance, BETA, t=0,
                                        decay_factor=self.args.decay_rate,
                                        weights=self.model_score_fn_weights)
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
        while ehat_value >= self.args.ehat_threshold and \
                len(self.T_x[model_idx]) < self.args.max_num_acquired_points:
            new_x, ehat_value, ttime = \
                self.picktolearn_one_iteration(model, new_x, 
                                                rng_instance, model_idx,
                                                stage=it)
            tot_time += ttime
            it += 1
            if self.args.plot_during:
                self.plot_model(model, self.learned_V, self.oned_x, self.oned_y, 
                                self.albert_levels[model_idx], 
                                seed=self.seed_list[model_idx], stage=it)
            
            print("\n\n\n\n\n\nITERATION", it, "\n\n\n\n\n\n\n")
        
        if len(self.T_x[model_idx]) >= self.args.max_num_acquired_points:
            self.seed_failed[model_idx] = True

        seed = self.seed_list[model_idx]
        plt.figure()
        plt.plot(range(len(self.llambdas[model_idx])), self.llambdas[model_idx])
        plt.xlabel("Iterations", fontsize=FONTSIZE)
        plt.ylabel("Lambda Value", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(self.args.picktolearn_logdir + f"/lambdas_{seed}.png", dpi=1000)

        self.method_times[model_idx] += tot_time

    def post_alg_all_seeds(self):
        if self.args.multiple_seeds:
            self.seed_failed = np.array(self.seed_failed)
            self.method_times = np.array(self.method_times)[~self.seed_failed]
            self.albert_levels = np.array(self.albert_levels)[~self.seed_failed]
            self.albert_times = np.array(self.albert_times)[~self.seed_failed]
            self.albert_num_samples = np.array(self.albert_num_samples)[~self.seed_failed]
            self.model_list = np.array(self.model_list)[~self.seed_failed]
            self.rng_list = np.array(self.rng_list)[~self.seed_failed]
            self.seed_list = np.array(self.seed_list)[~self.seed_failed]
            self.T_x = [self.T_x[i] for i in range(len(self.seed_failed)) \
                        if not self.seed_failed[i]] 

    def run_validation(self):
        output = {}
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
        
        d = {}
        output['our_method'] = d
        d['method_times'] = self.method_times
        d['avg_tpr'] = np.mean(all_tprs)
        d['avg_fnr'] = np.mean(all_fnrs)
        d['avg_fpr'] = np.mean(all_fprs)
        d['avg_tnr'] = np.mean(all_tnrs)
        d['results'] = results_dict
        return output 

    def robust_albert_validation(self, robust_results_dict, robust_albert_arrays, 
                                seed, seed_index):
        '''
        Run validation for 4 different settings of level, each with a different 
        N and epsilon from the robust scenario optimization method.
        '''
        # Minimal epsilon overall
        robust_arr = robust_albert_arrays[seed_index][:, :, 0]  
        print("\nRobust arr: ", robust_arr)
        coords = np.unravel_index(np.argmin(robust_arr), robust_arr.shape)
        robust_level1 = ROBUST_ALBERT_LEVEL_SWEEP[coords[1]]
        robust_N1 = robust_albert_arrays[seed_index][coords[0], coords[1], 1]
        robust_numviol1 = robust_albert_arrays[seed_index][coords[0], coords[1], 2]
        robust_eps1 = robust_arr[coords]
        print("Level, N, num violations, and robust eps: ", robust_level1, 
                                        robust_N1, robust_numviol1, robust_eps1)
        # Minimal epsilon for minimal N
        new_arr = robust_arr[0, :]
        ind_for_level = np.argmin(new_arr)
        robust_level2 = ROBUST_ALBERT_LEVEL_SWEEP[ind_for_level]
        robust_N2 = robust_albert_arrays[seed_index][0, ind_for_level, 1]
        robust_numviol2 = robust_albert_arrays[seed_index][0, ind_for_level, 2]
        robust_eps2 = new_arr[ind_for_level]
        print("Level, N, num violations, and robust eps: ", robust_level2, 
                                        robust_N2, robust_numviol2, robust_eps2)
        # Minimal epsilon for minimal level
        new_arr = robust_arr[:, 0]
        ind_for_N = np.argmin(new_arr)
        robust_level3 = ROBUST_ALBERT_LEVEL_SWEEP[0]
        robust_N3 = robust_albert_arrays[seed_index][ind_for_N, 0, 1]
        robust_numviol3 = robust_albert_arrays[seed_index][ind_for_N, 0, 2]
        robust_eps3 = new_arr[ind_for_N]
        print("Level, N, num violations, and robust eps: ", robust_level3, 
                                        robust_N3, robust_numviol3, robust_eps3)
        # Median epsilon overall
        flattened_indices = np.argsort(robust_arr, axis=None)
        mid_index = len(flattened_indices) // 2
        median_flat_index = flattened_indices[mid_index]
        coords = np.unravel_index(median_flat_index, robust_arr.shape)
        robust_level4 = ROBUST_ALBERT_LEVEL_SWEEP[coords[1]]
        robust_N4 = robust_albert_arrays[seed_index][coords[0], coords[1], 1]
        robust_numviol4 = robust_albert_arrays[seed_index][coords[0], coords[1], 2]
        robust_eps4 = robust_arr[coords]
        print("Level, N, num violations, and robust eps: ", robust_level4, 
                                        robust_N4, robust_numviol4, robust_eps4)

        tpr1, fpr1, tnr1, fnr1 = validate_albert(self.policy, robust_level1, 
                                            self.validation_candidates, 
                                            self.validation_true_costs,
                                            self.state_expander)
        tpr2, fpr2, tnr2, fnr2 = validate_albert(self.policy, robust_level2, 
                                                self.validation_candidates, 
                                                self.validation_true_costs,
                                                self.state_expander)
        tpr3, fpr3, tnr3, fnr3 = validate_albert(self.policy, robust_level3, 
                                                self.validation_candidates, 
                                                self.validation_true_costs,
                                                self.state_expander)
        tpr4, fpr4, tnr4, fnr4 = validate_albert(self.policy, robust_level4, 
                                                self.validation_candidates, 
                                                self.validation_true_costs,
                                                self.state_expander)
        robust_results_dict[seed] = {'1': [tpr1, fpr1, tnr1, fnr1, \
                                        robust_level1, robust_N1, robust_numviol1,
                                        robust_eps1],
                                    '2': [tpr2, fpr2, tnr2, fnr2, \
                                        robust_level2, robust_N2, robust_numviol2,
                                        robust_eps2],
                                    '3': [tpr3, fpr3, tnr3, fnr3, \
                                        robust_level3, robust_N3, robust_numviol3,
                                        robust_eps3],
                                    '4': [tpr4, fpr4, tnr4, fnr4, \
                                        robust_level4, robust_N4, robust_numviol4,
                                        robust_eps4]}

    def validate_albert_method(self):
        ''' Validate both the iterative and robust scenario optimization methods. '''
        output = {}
        results_dict = {}
        robust_results_dict = {}
        if self.args.multiple_seeds:
            for i in range(len(self.rng_list)):
                seed = self.seed_list[i]
                self.robust_albert_validation(robust_results_dict, 
                                            self.robust_albert_arrays, 
                                            seed, i)
                
                tpr, fpr, tnr, fnr = validate_albert(self.policy, self.albert_levels[i], 
                                                        self.validation_candidates, 
                                                        self.validation_true_costs,
                                                        self.state_expander)
                results_dict[seed] = [tpr, fpr, tnr, fnr]
        else:
            if not self.seed_failed[0]:
                self.robust_albert_validation(robust_results_dict, 
                                                self.robust_albert_arrays, 
                                                self.args.random_seed, 0)

                tpr, fpr, tnr, fnr = validate_albert(self.policy, self.albert_levels[0], 
                                                        self.validation_candidates, 
                                                        self.validation_true_costs,
                                                        self.state_expander)
                results_dict[self.args.random_seed] = [tpr, fpr, tnr, fnr]
            
        keys = list(results_dict.keys())
        all_tprs = [results_dict[key][0] for key in keys]
        all_fnrs = [results_dict[key][3] for key in keys]
        all_fprs = [results_dict[key][1] for key in keys]
        all_tnrs = [results_dict[key][2] for key in keys]

        robust_tprs_1 = [robust_results_dict[key]['1'][0] for key in keys]
        robust_tprs_2 = [robust_results_dict[key]['2'][0] for key in keys]
        robust_tprs_3 = [robust_results_dict[key]['3'][0] for key in keys]
        robust_tprs_4 = [robust_results_dict[key]['4'][0] for key in keys]
        robust_fnrs_1 = [robust_results_dict[key]['1'][3] for key in keys]
        robust_fnrs_2 = [robust_results_dict[key]['2'][3] for key in keys]
        robust_fnrs_3 = [robust_results_dict[key]['3'][3] for key in keys]
        robust_fnrs_4 = [robust_results_dict[key]['4'][3] for key in keys]
        robust_fprs_1 = [robust_results_dict[key]['1'][1] for key in keys]
        robust_fprs_2 = [robust_results_dict[key]['2'][1] for key in keys]
        robust_fprs_3 = [robust_results_dict[key]['3'][1] for key in keys]
        robust_fprs_4 = [robust_results_dict[key]['4'][1] for key in keys]
        robust_tnrs_1 = [robust_results_dict[key]['1'][2] for key in keys]
        robust_tnrs_2 = [robust_results_dict[key]['2'][2] for key in keys]
        robust_tnrs_3 = [robust_results_dict[key]['3'][2] for key in keys]
        robust_tnrs_4 = [robust_results_dict[key]['4'][2] for key in keys]
        robust_Ns_1 = [robust_results_dict[key]['1'][5] for key in keys]
        robust_Ns_2 = [robust_results_dict[key]['2'][5] for key in keys]
        robust_Ns_3 = [robust_results_dict[key]['3'][5] for key in keys]
        robust_Ns_4 = [robust_results_dict[key]['4'][5] for key in keys]
        robust_epss_1 = [robust_results_dict[key]['1'][7] for key in keys]
        robust_epss_2 = [robust_results_dict[key]['2'][7] for key in keys]
        robust_epss_3 = [robust_results_dict[key]['3'][7] for key in keys]
        robust_epss_4 = [robust_results_dict[key]['4'][7] for key in keys]

        print("\nRobust Scenario Opt:")
        print("Minimal epsilon overall")
        print("Average TPR1 Over All Seeds: ", np.mean(robust_tprs_1))
        print("Average FNR1 Over All Seeds: ", np.mean(robust_fnrs_1))
        print("Average FPR1 Over All Seeds: ", np.mean(robust_fprs_1))
        print("Average TNR1 Over All Seeds: ", np.mean(robust_tnrs_1))
        print("Average N Over All Seeds: ", np.mean(robust_Ns_1))
        print("Average eps Over All Seeds: ", np.mean(robust_epss_1))
        print("Minimal epsilon for minimal N")
        print("Average TPR2 Over All Seeds: ", np.mean(robust_tprs_2))
        print("Average FNR2 Over All Seeds: ", np.mean(robust_fnrs_2))
        print("Average FPR2 Over All Seeds: ", np.mean(robust_fprs_2))
        print("Average TNR2 Over All Seeds: ", np.mean(robust_tnrs_2))
        print("Average N Over All Seeds: ", np.mean(robust_Ns_2))
        print("Average eps Over All Seeds: ", np.mean(robust_epss_2))
        print("Minimal epsilon for minimal level")
        print("Average TPR3 Over All Seeds: ", np.mean(robust_tprs_3))
        print("Average FNR3 Over All Seeds: ", np.mean(robust_fnrs_3))
        print("Average FPR3 Over All Seeds: ", np.mean(robust_fprs_3))
        print("Average TNR3 Over All Seeds: ", np.mean(robust_tnrs_3))
        print("Average N Over All Seeds: ", np.mean(robust_Ns_3))
        print("Average eps Over All Seeds: ", np.mean(robust_epss_3))
        print("Median epsilon overall")
        print("Average TPR4 Over All Seeds: ", np.mean(robust_tprs_4))
        print("Average FNR4 Over All Seeds: ", np.mean(robust_fnrs_4))
        print("Average FPR4 Over All Seeds: ", np.mean(robust_fprs_4))
        print("Average TNR4 Over All Seeds: ", np.mean(robust_tnrs_4))
        print("Average N Over All Seeds: ", np.mean(robust_Ns_4))
        print("Average eps Over All Seeds: ", np.mean(robust_epss_4))

        print("\nIterative Method:")
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

        print("\nAlbert levels and mean", self.albert_levels, \
                                            np.mean(self.albert_levels))
        print("Albert total times and mean", self.albert_times, \
                                            np.mean(self.albert_times))
        print("Albert total num samples and mean", self.albert_num_samples, \
                                            np.mean(self.albert_num_samples))
        

        d = {}
        output['albert_itr_method'] = d
        d['method_times'] = self.albert_times
        d['num_samples'] = self.albert_num_samples
        d['levels'] = self.albert_levels
        d['avg_tpr'] = np.mean(all_tprs)
        d['avg_fnr'] = np.mean(all_fnrs)
        d['avg_fpr'] = np.mean(all_fprs)
        d['avg_tnr'] = np.mean(all_tnrs)
        d['results'] = results_dict

        d = {}
        output['albert_robust_method'] = d
        d['overall'] = [np.mean(robust_tprs_1), np.mean(robust_fnrs_1), 
                np.mean(robust_fprs_1), np.mean(robust_tnrs_1), 
                np.mean(robust_Ns_1), np.mean(robust_epss_1)]
        d['min_N'] = [np.mean(robust_tprs_2), np.mean(robust_fnrs_2), 
                np.mean(robust_fprs_2), np.mean(robust_tnrs_2), 
                np.mean(robust_Ns_2), np.mean(robust_epss_2)]
        d['min_level'] = [np.mean(robust_tprs_3), np.mean(robust_fnrs_3), 
                np.mean(robust_fprs_3), np.mean(robust_tnrs_3), 
                np.mean(robust_Ns_3), np.mean(robust_epss_3)]
        d['median'] = [np.mean(robust_tprs_4), np.mean(robust_fnrs_4), 
                np.mean(robust_fprs_4), np.mean(robust_tnrs_4), 
                np.mean(robust_Ns_4), np.mean(robust_epss_4)]
        d['results'] = robust_results_dict

        return output 
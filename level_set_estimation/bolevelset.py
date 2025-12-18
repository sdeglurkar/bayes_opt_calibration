import GPy
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import dill

class BOLevelSet:
    def __init__(self, f, init_fn, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, 
                length_scale, logdir=None):
        self.f = f
        self.init_fn = init_fn
        self.input_dim = input_dim
        self.candidates = candidates
        self.range_x = range_x
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.cost_thres = cost_thres
        self.conf_thres = conf_thres
        self.logdir = logdir
        self.acq_cache = []
        self.num_counterexamples_list = []

    def initial_setup(self, warmstart_sample, rng_instance):
        X_init = []
        for i in range(self.input_dim):
            # X_i_init = np.random.uniform(
            #     self.range_x[i][0],
            #     self.range_x[i][1],
            #     (warmstart_sample,)
            # )
            X_i_init = rng_instance.uniform(
                self.range_x[i][0],
                self.range_x[i][1],
                (warmstart_sample,)
            )
            X_init.append(X_i_init)
        self.X = np.stack(X_init, axis=1)
        self.Y = self.f(self.X) 
        if self.init_fn is None:
            self.m = GPy.models.GPRegression(
                self.X, 
                self.Y, 
                GPy.kern.Matern52(self.input_dim, lengthscale=self.length_scale, ARD=False),
                noise_var = self.noise_var
                )
        else:
            self.m = GPy.models.GPRegression(
                self.X, 
                self.Y, 
                GPy.kern.Matern52(self.input_dim, lengthscale=self.length_scale, ARD=False),
                noise_var = self.noise_var,
                mean_function=self.init_fn
                )
        self.m.optimize(messages=True)
        self.acq = self.MILE(self.candidates, self.cost_thres, self.conf_thres)
        self.plot()
        self.save(self.logdir + f'/bols_init')
    
    def initial_setup_given_data(self, X, Y, to_plot=True, plot_iter=None):
        self.X = X
        self.Y = Y
        if self.init_fn is None:
            self.m = GPy.models.GPRegression(
                X, 
                Y, 
                GPy.kern.Matern52(self.input_dim, lengthscale=self.length_scale, ARD=False),
                noise_var = self.noise_var
                )
        else:
            self.m = GPy.models.GPRegression(
                X, 
                Y, 
                GPy.kern.Matern52(self.input_dim, lengthscale=self.length_scale, ARD=False),
                noise_var = self.noise_var,
                mean_function=self.init_fn
                )
        self.m.optimize(messages=True)
        # self.acq = self.MILE(self.candidates, self.cost_thres, self.conf_thres)
        if to_plot:
            self.plot(iter=plot_iter, plot_acq=False)
        self.save(self.logdir + f'/error_gp_init')

    def optimize_given_one_more_point(self, x, y, plot=True, save=False, iter=0):
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, y))
        self.m.set_XY(X=self.X, Y=self.Y)
        self.m.optimize(messages=True)
        if plot:
            self.plot(iter=iter, plot_acq=False)
        if save:
            self.save(self.logdir + f'/bols_{iter}')

    def MILE(self, x, cost_thres, conf_thres):
        beta = norm.ppf(conf_thres)
        objecs = np.zeros((len(x), 1))
        mu, cov = self.m.predict(x, full_cov=True, include_likelihood=True)
        var = np.diag(cov)
        for i in range(x.shape[0]):
            var_gpp = var - (cov[:, i]) ** 2 / (var[i] + self.noise_var)
            zvec = np.sqrt(var[i] + self.noise_var) / np.abs(cov[:, i]) * (cost_thres - mu - beta * np.sqrt(var_gpp))
            objecs[i] = np.sum(norm.cdf(zvec))
        return objecs

    @staticmethod
    def normalize(data, scale=1):
        return (data-np.min(data))/(np.max(data)-np.min(data)) * scale

    def optimize_once(self, original_cand_len, shaped_candidates, index, 
                            to_plot=True, plot=True, save=False, iter=0):
        # max_acq_idx = np.unravel_index(np.argmax(self.acq, axis=None), self.acq.shape)
        # x_next = self.candidates[max_acq_idx[0], :][np.newaxis, :]
        sorted_indices = np.argsort(np.squeeze(self.acq))[::-1]
        acq_idx = np.unravel_index(sorted_indices[index], (original_cand_len, original_cand_len))
        x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
        self.acq_cache.append(x_next)
        y_next = self.f(x_next) 
        self.X = np.vstack((self.X, x_next))
        self.Y = np.vstack((self.Y, y_next))
        self.m.set_XY(X=self.X, Y=self.Y)
        self.m.optimize(messages=True)
        self.acq = self.MILE(self.candidates, self.cost_thres, self.conf_thres)
        if to_plot and plot:
            self.plot(iter=iter)
        if save:
            self.save(self.logdir + f'/bols_{iter}')
    
    def optimize_once_error_gp(self, error_gp, original_cand_len, candidates, shaped_candidates,
                                rng_instance,
                                index, beta, to_plot=True, plot=True, save=False, iter=0):
        error_mu, error_var = error_gp.m.predict(candidates, full_cov=False)
        prob_error_greater_than = []
        for i in range(len(error_mu)):
            prob_error_greater_than.append(1 - \
                scipy.stats.norm.cdf(0.25, loc=error_mu[i], scale=np.sqrt(error_var[i])))
        prob_error_greater_than = np.array(prob_error_greater_than)/np.sum(prob_error_greater_than) # Normalize
        cumulative_sum = np.cumsum(prob_error_greater_than)
        assert (np.abs(cumulative_sum[-1] - 1.0)) < 1e-6
        rand_num = rng_instance.uniform(0, 1.0)
        sampled_index = np.where(cumulative_sum < rand_num)[0][-1]
        # sorted_indices = np.argsort(np.squeeze(error_mu))[::-1]
        # acq_idx = np.unravel_index(sorted_indices[index], (original_cand_len, original_cand_len))
        # x_next = shaped_candidates[acq_idx[0], acq_idx[1], :][np.newaxis, :]
        x_next = candidates[sampled_index][np.newaxis, :]
        self.acq_cache.append(x_next)
        y_next = self.f(x_next) 
        self.X = np.vstack((self.X, x_next))
        self.Y = np.vstack((self.Y, y_next))
        self.m.set_XY(X=self.X, Y=self.Y)
        self.m.optimize(messages=True)
        if to_plot and plot:
            self.plot(iter=iter, plot_acq=False)
        if save:
            self.save(self.logdir + f'/bols_{iter}')
    
    def optimize_once_counterexamples(self, counterexamples, rng_instance, 
                                        to_plot=True, plot=True, save=False, iter=0):
        if len(counterexamples) > 500:
            x_next = []
            for _ in range(3):
                x_1 = rng_instance.uniform(
                    self.range_x[0][0],
                    self.range_x[0][1])
                x_2 = rng_instance.uniform(
                    self.range_x[1][0],
                    self.range_x[1][1])
                x_next.append([x_1, x_2])
                self.acq_cache.append([[x_1, x_2]])
            y_next = self.f(x_next)
            self.X = np.vstack((self.X, x_next))
            self.Y = np.vstack((self.Y, y_next))
        else:
            random_counterexample_ind = rng_instance.integers(0, len(counterexamples)) 
            random_counterexample = counterexamples[random_counterexample_ind]
            x_next = [random_counterexample]
            self.acq_cache.append(x_next)
            y_next = self.f(x_next)
            self.X = np.vstack((self.X, x_next))
            self.Y = np.vstack((self.Y, y_next))
            if len(counterexamples) > 3:
                random_inds = [random_counterexample_ind]
                for _ in range(2):
                    random_ind = rng_instance.integers(0, len(counterexamples)) 
                    while random_ind in random_inds:
                        random_ind = rng_instance.integers(0, len(counterexamples)) 
                    random_inds.append(random_ind)
                    random_counterexample = counterexamples[random_ind]
                    x_next = [random_counterexample]
                    self.acq_cache.append(x_next)
                    y_next = self.f(x_next)
                    self.X = np.vstack((self.X, x_next))
                    self.Y = np.vstack((self.Y, y_next))

        self.m.set_XY(X=self.X, Y=self.Y)
        self.m.optimize(messages=True)
        if to_plot and plot:
            self.plot(iter=iter, plot_acq=False)
        if save:
            self.save(self.logdir + f'/bols_{iter}')
        
    def optimize_loop(self, original_cand_len, shaped_candidates, iters, to_plot=True, 
                        plot_every=10, save_every=10):
        # Schedule
        indices = []
        ctr1 = 0
        ctr2 = 0
        for i in range(iters):
            # if i < iters/3:
            #     indices.append(-1 - ctr1)  # Best error
            #     ctr1 += 1
            # elif i < iters/2:
            #     indices.append(int(original_cand_len/2) - ctr2)  # Median error
            #     ctr2 += 1
            # else:
            #     indices.append(0) # Worst error
            indices.append(0) # Worst error

        for i in range(iters):
            print(f"optimizing step {i}")
            index = indices[i]
            self.optimize_once(original_cand_len, shaped_candidates, index, to_plot,
                            plot=((i+1) % plot_every == 0), save=((i+1) % save_every == 0), iter=i)
        print("All points queried: ", np.array(self.acq_cache))

    def optimize_loop_error_gp(self, error_gp, original_cand_len, candidates, shaped_candidates,
                                calibration_points, costs_at_calibration_points, beta, rng_instance,
                                iters, to_plot=True, plot_every=1, save_every=10):
        # Schedule
        indices = []
        ctr1 = 0
        ctr2 = 0
        for i in range(iters):
            # if i < iters/3:
            #     indices.append(-1 - ctr1)  # Best error
            #     ctr1 += 1
            # elif i < iters/2:
            #     indices.append(int(original_cand_len/2) - ctr2)  # Median error
            #     ctr2 += 1
            # else:
            #     indices.append(0) # Worst error
            indices.append(0) # Worst error

        for i in range(iters):
            print(f"optimizing step {i}")
            index = indices[i]
            self.optimize_once_error_gp(error_gp, original_cand_len, candidates, shaped_candidates, rng_instance, 
                                index, beta, to_plot, 
                                plot=((i+1) % plot_every == 0), save=((i+1) % save_every == 0), 
                                iter=i)
            # Re-fit error GP
            mu, var = self.m.predict(calibration_points, full_cov=False)
            criterion = mu - beta * np.sqrt(var) # Conservative bc more likely to say state is in BRT
            # errors = np.abs(criterion - costs_at_calibration_points)
            errors = []
            for i in range(len(criterion)):
                if (costs_at_calibration_points[i] == 0 and criterion[i] <= 0): # Product is 0
                    errors.append(0)
                elif criterion[i] * costs_at_calibration_points[i] > 0:  # They have the same sign
                    errors.append(0)
                else: # Even if costs_at_calibration_points[i] < 0 and criterion[i] == 0, say it's wrong
                    errors.append(1) 
            errors = np.expand_dims(np.array(errors), -1)
            error_gp.initial_setup_given_data(calibration_points, errors, to_plot, plot_iter=i)
        print("All points queried: ", np.array(self.acq_cache))

    def optimize_loop_counterexamples(self, search_candidates, true_costs, beta, 
                                        rng_instance, iters, 
                                        to_plot=True, plot_every=1, save_every=10):
        for i in range(iters):
            print(f"optimizing step {i}")
            mu, var = self.m.predict(search_candidates, full_cov=False)
            criterion = mu - beta * np.sqrt(var) # Conservative bc more likely to say state is in BRT
        
            assert len(criterion) == len(true_costs)
            gp_below_zero = np.where(criterion <= 0)
            gp_above_zero = np.where(criterion > 0)
            true_below_zero = np.where(true_costs <= 0)
            true_above_zero = np.where(true_costs > 0) 

            true_pos = np.intersect1d(gp_below_zero, true_below_zero)
            false_pos = np.intersect1d(gp_below_zero, true_above_zero)
            true_neg = np.intersect1d(gp_above_zero, true_above_zero)
            false_neg = np.intersect1d(gp_above_zero, true_below_zero)

            counterexample_inds = np.union1d(false_pos, false_neg)
            counterexamples = search_candidates[counterexample_inds]

            self.num_counterexamples_list.append(len(counterexamples))
            print("COUNTEREXAMPLES", len(counterexamples))

            if len(counterexamples) == 0:
                print("\nNO COUNTEREXAMPLES")
            else:
                self.optimize_once_counterexamples(counterexamples, rng_instance, 
                                    to_plot, plot=((i+1) % plot_every == 0), save=((i+1) % save_every == 0), 
                                    iter=i)
            
        print("All points queried: ", np.array(self.acq_cache))
        print("Counterexamples over time: ", self.num_counterexamples_list)

    def extract_levelset(self, x_test):
        beta = norm.ppf(self.conf_thres)
        mu, var = self.m.predict(x_test, full_cov=False)
        criterion = mu + beta * np.sqrt(var)
        return x_test[(criterion < self.cost_thres).squeeze()]

    def plot(self, iter=0, plot_acq=True):
        if self.input_dim == 1:
            self.m.plot(plot_limits=np.array([self.range_x[0][0]-0.25, self.range_x[0][1]+0.25]))
            if plot_acq:
                plt.plot(self.candidates.flatten(), BOLevelSet.normalize(self.acq.flatten(), 10), c='orange')
            # 0-line
            plt.plot(self.candidates.flatten(), 0*self.candidates.flatten(), c='k')
            plt.savefig(self.logdir + f'/gp_{iter}.png')
        if self.input_dim == 2:
            self.m.plot()
            plt.savefig(self.logdir + f'/gp_{iter}.png')
            plt.figure()
            if plot_acq:
                x = self.candidates[:, 0].flatten()
                y = self.candidates[:, 1].flatten()
                original_cand_len = int(np.sqrt(len(x)))
                original_x = x.reshape((original_cand_len, original_cand_len))[0, :]
                original_y = y.reshape((original_cand_len, original_cand_len))[:, 0]
                z = self.acq.reshape((original_cand_len, original_cand_len))
                plt.contourf(original_x, original_y, BOLevelSet.normalize(z))
                plt.colorbar()
                plt.savefig(self.logdir + f'/mile_{iter}.png')
        if self.input_dim == 3:
            fixed_dims = [(2, 0.0)]
            self.m.plot(fixed_inputs=fixed_dims, projection='2d')
            plt.savefig(self.logdir + f'/gp_{iter}.png')

    def save(self, fname):
        temp_f = self.f
        self.f = None
        with open(fname, 'wb') as f:
            dill.dump(self, f)
        self.f = temp_f

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            return dill.load(f)
import GPy
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import dill

class MainGP:
    def __init__(self, f, init_fn, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, 
                length_scale, do_MILE=False, logdir=None):
        self.f = f
        self.init_fn = init_fn
        self.input_dim = input_dim
        self.candidates = candidates
        self.range_x = range_x
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.do_MILE = do_MILE
        self.cost_thres = cost_thres
        self.conf_thres = conf_thres
        self.logdir = logdir
        self.acq_cache = np.array([[]])
    
    def initial_setup(self, warmstart_sample, rng_instance, to_plot=True):
        X_init = []
        for i in range(self.input_dim):
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
        self.m.optimize_restarts(messages=True)
        if self.do_MILE:
            self.acq = self.MILE(self.candidates, self.cost_thres, self.conf_thres)
        if to_plot and self.logdir is not None:
            self.plot(plot_acq=self.do_MILE)
        self.save(self.logdir + f'/bols_init')

    def get_error_of_model_for_points(self, points_x, points_y_true, beta):
        mu, var = self.m.predict(points_x, full_cov=False)
        criterion = mu - beta * np.sqrt(var) # Conservative bc more likely to say state is in BRT
        # 1 is error, 0 is no error
        error = ((criterion * points_y_true <= 0) & (criterion != points_y_true)).astype(float)
        # error = np.where(criterion <= 0 and points_y_true > 0 or   
        #                 criterion > 0 and points_y_true <= 0, 1, 0)  
        return error

    def ground_truth_error(self, candidate_xs, beta, rng_instance, perturb=1e-4):
        '''
        Return e_h(z) for a desired set of z's.
        For calibration and debugging purposes.
        '''
        true_ys = self.f(candidate_xs)
        e = self.get_error_of_model_for_points(beta, candidate_xs, true_ys)
        rand_nums = rng_instance.uniform(low=0.0, high=1.0, size=e.shape)
        e = e + perturb * rand_nums  # To create a total order
        return e

    def score_function(self, candidate_xs, rng_instance, error_function, 
                            beta, t, decay_factor=0.9, weights=[1.0, 0.0, 0.001],
                            tol=1e-5):
        '''
        NOTE: error_function != ehat
        ehat = final_scores + beta * error_variances
        This function returns a_{h,eta}(z) and u_{h,eta}(z) for a desired set of z's.
        '''
        # dist_from_boundary = []
        # for candidate_x in candidate_xs:
        #     if candidate_x[0] <= -5 and candidate_x[0] >= -10 and candidate_x[1] >= -10 and candidate_x[1] <= 10:
        #         dist_from_boundary.append([1.0])
        #     else:
        #         dist_from_boundary.append([0.1])
        # dist_from_boundary = np.array(dist_from_boundary)
        mu, var = self.m.predict(candidate_xs, full_cov=False)
        criterion = mu - beta * np.sqrt(var) # Conservative bc more likely to say state is in BRT
        dist_from_boundary = np.abs(criterion)
        dist_from_boundary = 1 - dist_from_boundary/np.max(dist_from_boundary)  # Normalize
        
        dist_from_boundary = dist_from_boundary * decay_factor**t
        rand_nums = rng_instance.uniform(low=0.0, high=1.0, size=dist_from_boundary.shape)
        if error_function is not None:
            errors, error_variances = error_function(candidate_xs)
            errors = errors/sum(errors)  # Normalize
            final_scores = weights[0] * dist_from_boundary + weights[1] * errors + \
                            weights[2] * rand_nums
        else:
            error_variances = 0.1 * dist_from_boundary + tol
            final_scores = weights[0] * dist_from_boundary + \
                            weights[2] * rand_nums
        return final_scores, error_variances
    
    def optimize_given_T(self, T_x, T_y,
                                to_plot=True, plot=True, save=False, iter=0):
        self.X = np.vstack((self.X, T_x))
        self.Y = np.vstack((self.Y, T_y))
        self.m.set_XY(X=self.X, Y=self.Y)
        self.m.optimize_restarts(messages=False)
        if to_plot and plot and self.logdir is not None:
            self.plot(iter=iter, plot_acq=self.do_MILE)
        if save:
            self.save(self.logdir + f'/bols_{iter}')

    # def optimize_once(self, ehat, candidate_xs,
    #                         to_plot=True, plot=True, save=False, iter=0):
    #     '''
    #     ehat = final_scores + lambda * error_variances
    #     Pick the point that maximizes ehat and fit the model.
    #     '''
    #     argmax_index = np.argmax(ehat)
    #     x_next = candidate_xs[argmax_index]
    #     self.acq_cache.append(x_next)
    #     y_next = self.f(x_next) 
    #     self.X = np.vstack((self.X, x_next))
    #     self.Y = np.vstack((self.Y, y_next))
    #     self.m.set_XY(X=self.X, Y=self.Y)
    #     self.m.optimize_restarts(messages=True)
    #     if to_plot and plot and self.logdir is not None:
    #         self.plot(iter=iter)
    #     if save:
    #         self.save(self.logdir + f'/bols_{iter}')

    def extract_levelset(self, x_test):
        raise NotImplementedError
        beta = norm.ppf(self.conf_thres)
        mu, var = self.m.predict(x_test, full_cov=False)
        criterion = mu + beta * np.sqrt(var)
        return x_test[(criterion < self.cost_thres).squeeze()]

    def plot(self, iter=0, plot_acq=True):
        if self.input_dim == 1:
            self.m.plot(plot_limits=np.array([self.range_x[0][0]-0.25, self.range_x[0][1]+0.25]))
            if plot_acq:
                plt.plot(self.candidates.flatten(), MainGP.normalize(self.acq.flatten(), 10), c='orange')
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
                plt.contourf(original_x, original_y, MainGP.normalize(z))
                plt.colorbar()
                plt.savefig(self.logdir + f'/mile_{iter}.png')
        if self.input_dim == 3:
            raise NotImplementedError
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
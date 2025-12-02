import GPy
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import dill

class BOLevelSet:
    def __init__(self, f, init_fn, input_dim, candidates, range_x, noise_var, cost_thres, conf_thres, length_scale, logdir=None):
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

    def initial_setup(self, warmstart_sample):
        X_init = []
        for i in range(self.input_dim):
            X_i_init = np.random.uniform(
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
    
    def initial_setup_given_data(self, X, Y):
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
        self.plot(plot_acq=False)
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

    def optimize_once(self, plot=True, save=False, iter=0):
        max_acq_idx = np.unravel_index(np.argmax(self.acq, axis=None), self.acq.shape)
        x_next = self.candidates[max_acq_idx[0], :][np.newaxis, :]
        y_next = self.f(x_next) 
        self.X = np.vstack((self.X, x_next))
        self.Y = np.vstack((self.Y, y_next))
        self.m.set_XY(X=self.X, Y=self.Y)
        self.m.optimize(messages=True)
        self.acq = self.MILE(self.candidates, self.cost_thres, self.conf_thres)
        if plot:
            self.plot(iter=iter)
        if save:
            self.save(self.logdir + f'/bols_{iter}')
        
    def optimize_loop(self, iters, plot_every=10, save_every=10):
        for i in range(iters):
            print(f"optimizing step {i}")
            self.optimize_once(plot=((i+1) % plot_every == 0), save=((i+1) % save_every == 0), iter=i)

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
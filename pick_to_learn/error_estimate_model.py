import GPy
import numpy as np
import matplotlib.pyplot as plt
import dill

class ErrorGP:
    def __init__(self, f, init_fn, input_dim, range_x, noise_var, 
                length_scale, logdir=None):
        self.f = f
        self.init_fn = init_fn
        self.input_dim = input_dim
        self.range_x = range_x
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.logdir = logdir
    
    def fit(self, X, Y, to_plot=True, plot_iter=None):
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
        self.m.optimize_restarts(messages=True)
        if to_plot and self.logdir is not None:
            self.plot(iter=plot_iter)
        if self.logdir is not None: self.save(self.logdir + f'/error_gp_init')

    def forward(self):
        def call_fn(x):
            mu, var = self.m.predict(x, full_cov=False)
            return mu, var
        return call_fn 

    def plot(self, iter=0):
        if self.input_dim == 1:
            self.m.plot(plot_limits=np.array([self.range_x[0][0]-0.25, self.range_x[0][1]+0.25]))
            plt.savefig(self.logdir + f'/gp_{iter}.png')
        if self.input_dim == 2:
            self.m.plot()
            plt.savefig(self.logdir + f'/gp_{iter}.png')
            plt.figure()
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
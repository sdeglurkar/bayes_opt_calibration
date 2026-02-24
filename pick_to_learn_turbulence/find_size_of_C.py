# Code drawn from Angelopoulos and Bates, "A Gentle Introduction to Conformal 
# Prediction", 2022

import numpy as np
from scipy.optimize import brentq 
from scipy.stats import beta

def find_size_of_C(alpha, tolerance_alpha, beta_conformal):
    epsilons = [tolerance_alpha]
    for epsilon in epsilons:
        def _condition(n):
            l = np.floor((n+1)*alpha)
            a = n + 1 - l
            b = l
            if (beta.ppf(beta_conformal/2, a, b) < 1-alpha-epsilon) or \
                (beta.ppf(beta_conformal/2, a, b) > 1-alpha+epsilon):
                return -1
            else:
                return 1

        size_C = int(np.ceil(brentq(_condition,np.ceil(1/alpha),100000000000)))
    print("DESIRED SIZE C", size_C)
    return size_C
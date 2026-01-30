import numpy as np

def get_quantile_for_interval_score_fn(y_hat_data, u_data, y_data, alpha):
    n = len(y_data)
    quantile_val = np.ceil((1-alpha) * (n + 1))/n
    score = np.abs(y_hat_data - y_data)/u_data
    # print(score, quantile_val)
    return np.quantile(score, quantile_val)
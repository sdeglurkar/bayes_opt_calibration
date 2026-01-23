import numpy as np
from scipy.special import betainc

def find_epsLU(k, N, delta, tol=1e-10):
    # ---- Find epsL ----
    t1 = 0.0
    t2 = k/N
    while t2 - t1 > tol:
        t = 0.5 * (t1 + t2)
        left = (
            delta/3 * betainc(k + 1, N - k, t)
            + delta/6 * betainc(k + 1, 4 * N + 1 - k, t)
        )
        right = (
            (1 + delta /(6 * N))
            * t * N
            * (betainc(k, N - k + 1, t) - betainc(k + 1, N - k, t))
        )
        if left > right:
            t1 = t
        else:
            t2 = t
    epsL = t1

    # ---- Find epsU ----
    if k == N:
        epsU = 1.0
    else:
        t1 = k / N
        t2 = 1.0
        while t2 - t1 > tol:
            t = 0.5 * (t1 + t2)
            left = (
                (delta/2 - delta/6) * betainc(k + 1, N - k, t)
                + delta/6 * betainc(k + 1, 4 * N + 1 - k, t)
            )
            right = (
                (1 + delta/(6 * N))
                * t * N
                * (betainc(k, N - k + 1, t) - betainc(k + 1, N - k, t))
            )
            if left > right:
                t2 = t
            else:
                t1 = t

        epsU = t2

    return epsL, epsU


# N = 1000000
# size_T = 20
# delta = 1e-2

# epsL, epsU = find_epsLU(size_T, N, delta)
# print(epsU)
# print(size_T/N)
# print(epsL)


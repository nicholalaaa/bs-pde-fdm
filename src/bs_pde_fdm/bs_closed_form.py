import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, q=0.0, otype="call"):
    """
    Black-Scholes price with continuous dividend yield q.
    S, K, T, r, sigma, q can be scalars or numpy arrays (broadcastable).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    # Guard small T
    eps = 1e-12
    sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(np.maximum(S, eps)/np.maximum(K, eps)) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    disc_q = np.exp(-q*T)
    disc_r = np.exp(-r*T)
    if str(otype).lower().startswith("c"):
        return S*disc_q*norm.cdf(d1) - K*disc_r*norm.cdf(d2)
    else:
        return K*disc_r*norm.cdf(-d2) - S*disc_q*norm.cdf(-d1)

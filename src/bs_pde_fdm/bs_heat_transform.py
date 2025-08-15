import numpy as np


def bs_to_heat_params(r, q, sigma):
    """
    Return alpha,beta in V(S,t)=K*exp(alpha x + beta tau) * y(x,tau), x=ln(S/K), tau=0.5*sigma^2 (T-t)
    such that y satisfies the heat equation y_tau = y_xx.
    """
    # One standard mapping yields alpha = -(2*(r-q)/sigma**2 - 1)/2  and beta = -(2*r/sigma**2)/2
    # Here we provide one commonly used choice:
    alpha = 0.5*(1 - (2*(r - q)/sigma**2))
    beta = -((r + q)/sigma**2)
    return alpha, beta


def call_boundary_right(Smax, K, tau, r, q):
    """Asymptotic call boundary at large S."""
    return Smax*np.exp(-q*tau) - K*np.exp(-r*tau)


def put_boundary_left(K, tau, r, q):
    """Asymptotic put boundary near S=0."""
    return K*np.exp(-r*tau)

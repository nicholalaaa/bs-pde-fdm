import numpy as np
from bs_pde_fdm.fdm_theta_log import price_call_theta_logspace
from bs_pde_fdm.bs_closed_form import bs_price


def test_cn_log_converges_to_bs():
    # Vanilla setup
    S0 = K = 100.0
    T, r, q, sigma = 1.0, 0.03, 0.0, 0.20

    # HW-style parameters
    m = 400
    x0 = np.log(S0 / K) - 4.0 * sigma * np.sqrt(T)  # left boundary in x
    gamma_max = 0.45
    theta = 0.5  # Crank–Nicolson

    p_fdm, meta = price_call_theta_logspace(
        S0, K, T, r, q, sigma,
        m=m, x0=x0, gamma_max=gamma_max, theta=theta
    )
    p_bs = bs_price(S0, K, T, r, sigma, q=q, otype="call")

    # Convergence check
    assert abs(p_fdm - p_bs) < 5e-3

    # Sanity: gamma computed from meta respects the cap
    alpha = 0.5 * sigma * sigma
    dx, dtau = meta["dx"], meta["dtau"]
    gamma = alpha * dtau / (dx * dx)
    assert gamma <= gamma_max + 1e-12
    assert meta["N"] >= 1


def test_refinement_improves_accuracy():
    S0 = K = 100.0
    T, r, q, sigma = 1.0, 0.03, 0.0, 0.20
    x0 = np.log(S0 / K) - 4.0 * sigma * np.sqrt(T)
    gamma_max, theta = 0.45, 0.5

    p_bs = bs_price(S0, K, T, r, sigma, q=q, otype="call")

    p_coarse, _ = price_call_theta_logspace(S0, K, T, r, q, sigma, m=200, x0=x0, gamma_max=gamma_max, theta=theta)
    p_fine,   _ = price_call_theta_logspace(S0, K, T, r, q, sigma, m=400, x0=x0, gamma_max=gamma_max, theta=theta)

    err_coarse = abs(p_coarse - p_bs)
    err_fine   = abs(p_fine   - p_bs)
    assert err_fine <= err_coarse  # refinement should not get worse

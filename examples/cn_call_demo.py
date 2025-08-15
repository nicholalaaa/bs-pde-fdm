from bs_pde_fdm.fdm_theta_log import price_call_theta_logspace
from bs_pde_fdm.bs_closed_form import bs_price


def main():
    S0, K, T, r, q, sigma = 120.0, 119.0, 7/365, 0.005, 0.02, 0.10
    m = 20
    x0 = -0.05   # left boundary in x
    gamma_max = 24                         # your HW γ cap
    theta = 0.5                              # CN

    p_fdm, meta = price_call_theta_logspace(S0, K, T, r, q, sigma, m, x0, gamma_max, theta)
    p_bs = bs_price(S0, K, T, r, sigma, q=q, otype="call")
    print(f"FDM (log-space θ): {p_fdm:.6f}   |   BS: {p_bs:.6f}   |   N={meta['N']}, dx={meta['dx']:.5f}")


if __name__ == "__main__":
    main()

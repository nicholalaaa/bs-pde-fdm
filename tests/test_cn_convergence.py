import numpy as np
from bs_pde_fdm.fdm_cn import price_european_call_cn
from bs_pde_fdm.bs_closed_form import bs_price


def test_cn_converges_to_bs():
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.0, 0.2
    # Coarser grid (looser tolerance)
    p_cn, *_ = price_european_call_cn(S0, K, T, r, q, sigma, M=200, N=200)
    p_bs = bs_price(S0, K, T, r, sigma, q=q, otype="call")
    assert abs(p_cn - p_bs) < 2e-2

    # Finer grid (tighter tolerance)
    p_cn2, *_ = price_european_call_cn(S0, K, T, r, q, sigma, M=400, N=400)
    assert abs(p_cn2 - p_bs) < 8e-3

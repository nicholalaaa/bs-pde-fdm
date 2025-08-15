import numpy as np
from numpy.linalg import eigvals
from bs_pde_fdm.toeplitz_tridiag import tridiag_toeplitz_eigs, as_dense


def test_toeplitz_eigs_match_numpy():
    alpha, beta, gamma, n = 2.0, 1.0, 1.0, 20
    lam_analytic, _ = tridiag_toeplitz_eigs(alpha, beta, gamma, n)
    T = as_dense(alpha, beta, gamma, n)
    lam_num = np.sort(eigvals(T).real)
    lam_analytic_sorted = np.sort(lam_analytic)
    err = np.max(np.abs(lam_analytic_sorted - lam_num))
    assert err < 1e-10

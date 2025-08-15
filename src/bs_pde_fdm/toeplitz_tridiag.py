import numpy as np


def tridiag_toeplitz_eigs(alpha, beta, gamma, n):
    """
    Analytic eigenvalues/eigenvectors of an n x n tridiagonal Toeplitz matrix:
        T = diag(alpha) + diag(beta, 1) + diag(gamma, -1)
    For beta,gamma > 0, eigenvalues are:
        lambda_k = alpha + 2*sqrt(beta*gamma) * cos(k*pi/(n+1)), k=1..n
    and eigenvectors v^{(k)}_j ∝ (sqrt(beta/gamma))^j * sin(j*k*pi/(n+1)), j=1..n
    Returns (eigvals, eigvecs) with columns as eigenvectors (normalized).
    """
    k = np.arange(1, n+1)
    lam = alpha + 2.0*np.sqrt(beta*gamma)*np.cos(k*np.pi/(n+1))
    # Build eigenvectors
    ratio = np.sqrt(beta/gamma) if gamma != 0 else 1.0
    j = np.arange(1, n+1).reshape(-1, 1)  # rows
    V = (ratio**j) * np.sin(j * k*np.pi/(n+1))
    # Normalize columns
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    return lam, V


def as_dense(alpha, beta, gamma, n):
    """Construct the dense Toeplitz tridiagonal matrix."""
    T = np.zeros((n, n))
    np.fill_diagonal(T, alpha)
    np.fill_diagonal(T[1:], gamma)
    np.fill_diagonal(T[:,1:], beta)
    return T

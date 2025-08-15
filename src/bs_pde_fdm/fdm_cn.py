import numpy as np


def _thomas(lo, mn, up, rhs):
    """Solve tridiagonal system with lower/diag/upper (lengths n-1, n, n-1)."""
    n = len(mn)
    lo = lo.astype(float).copy()
    mn = mn.astype(float).copy()
    up = up.astype(float).copy()
    rhs = rhs.astype(float).copy()

    for k in range(1, n):
        m = lo[k-1] / mn[k-1]
        mn[k] -= m * up[k-1]
        rhs[k] -= m * rhs[k-1]
    x = np.empty(n, dtype=float)
    x[-1] = rhs[-1] / mn[-1]
    for k in range(n-2, -1, -1):
        x[k] = (rhs[k] - up[k] * x[k+1]) / mn[k]
    return x


def price_european_call_cn(
    S0, K, T, r, q, sigma,
    Smax_mult=5.0, M=400, N=400, theta=0.5
):
    """
    European call via θ-scheme (θ=0.5 => Crank-Nicolson) on a uniform S-grid.

    PDE in τ = T - t:  V_τ = 0.5 σ^2 S^2 V_SS + (r - q) S V_S - r V
    Grid: S_j = j ΔS,  j=0..M,  ΔS = S_max / M,  Δτ = T / N.

    Boundaries (call):
      V(0, τ)   = 0
      V(S_max, τ) ≈ S_max e^{-q τ} - K e^{-r τ}
    """
    S_max = Smax_mult * K
    M, N = int(M), int(N)
    dS = S_max / M
    dτ = T / N

    S = np.linspace(0.0, S_max, M + 1)
    V = np.maximum(S - K, 0.0)  # τ=0 payoff

    # Interior index 1..M-1 (we use the classic index form; i ≈ S_i / ΔS)
    i = np.arange(1, M, dtype=float)

    # Discretized operator coefficients (index form)
    # L V ≈ (A_i V_{i-1} + B_i V_i + C_i V_{i+1})
    # For a general θ-scheme:
    A = 0.5 * (sigma**2 * i**2) - 0.5 * (r - q) * i      # lower (i-1)
    B = -(sigma**2 * i**2) - r                            # main
    C = 0.5 * (sigma**2 * i**2) + 0.5 * (r - q) * i      # upper (i+1)

    # Build time-independent θ-scheme matrices:
    # Left (new time):  -θΔτ A,  1 - θΔτ B,  -θΔτ C
    # Right (old time): +(1-θ)Δτ A,  1 + (1-θ)Δτ B,  +(1-θ)Δτ C
    lo_A = -theta * dτ * A[1:]              # length M-2
    mn_A = 1.0  - theta * dτ * B           # length M-1
    up_A = -theta * dτ * C[:-1]             # length M-2

    lo_B = (1.0 - theta) * dτ * A[1:]       # length M-2
    mn_B = 1.0 + (1.0 - theta) * dτ * B    # length M-1
    up_B = (1.0 - theta) * dτ * C[:-1]      # length M-2

    # Time-stepping in τ from 0 → T
    for n in range(N):
        τ_old = n * dτ
        τ_new = (n + 1) * dτ

        # RHS = B * V_old (interior)
        V_old = V
        rhs = (mn_B * V_old[1:M]).copy()
        rhs[1:] += lo_B * V_old[1:M-1]      # contributions from V_{i-1}
        rhs[:-1] += up_B * V_old[2:M+0]     # contributions from V_{i+1}

        # Add old boundary contributions from B (Dirichlet):
        V_left_old = 0.0
        V_right_old = S_max * np.exp(-q * τ_old) - K * np.exp(-r * τ_old)
        rhs[0] += ((1.0 - theta) * dτ * A[0]) * V_left_old
        rhs[-1] += ((1.0 - theta) * dτ * C[-1]) * V_right_old

        # Move new boundary terms from A to RHS (Dirichlet at τ_new):
        V_left_new = 0.0
        V_right_new = S_max * np.exp(-q * τ_new) - K * np.exp(-r * τ_new)
        rhs[0] -= (-theta * dτ * A[0]) * V_left_new     # = +θΔτ A[0] * V_left_new
        rhs[-1] -= (-theta * dτ * C[-1]) * V_right_new   # = +θΔτ C[-1] * V_right_new

        # Solve for interior at new time
        x = _thomas(lo_A, mn_A, up_A, rhs)

        # Update full grid
        V_new = V_old.copy()
        V_new[0] = V_left_new
        V_new[-1] = V_right_new
        V_new[1:M] = x
        V = V_new

    # Interpolate to S0
    return float(np.interp(S0, S, V)), S, V

import numpy as np


def _thomas(lo, mn, up, rhs):
    n = len(mn)
    lo, mn, up, rhs = map(lambda a: np.asarray(a, float).copy(), (lo, mn, up, rhs))
    for k in range(1, n):
        m = lo[k-1] / mn[k-1]
        mn[k] -= m * up[k-1]
        rhs[k] -= m * rhs[k-1]
    x = np.empty(n, float)
    x[-1] = rhs[-1] / mn[-1]
    for k in range(n-2, -1, -1):
        x[k] = (rhs[k] - up[k] * x[k+1]) / mn[k]
    return x


def price_call_theta_logspace(
    S0, K, T, r, q, sigma,
    m,            # number of space intervals in x
    x0,           # left boundary in x = ln(S/K)
    gamma_max,    # cap for gamma := (σ^2/2)*Δτ / (Δx)^2
    theta=0.5,    # 0=Explicit, 0.5=CN, 1=Implicit
    width_std=4.0  # domain half-width ~ width_std*σ√T in x
):
    """
    θ-scheme on a UNIFORM x-grid (x=ln(S/K)), matching HW parameters (γ_max, m, x0).
    PDE in τ = T - t:
        V_τ = (σ^2/2)(V_xx - V_x) + (r - q) V_x - r V
          =  α V_xx + β V_x - r V,  with α = σ^2/2, β = (r - q) - σ^2/2
    We discretize directly in x (constant coefficients). γ := α Δτ / (Δx)^2 ≤ γ_max
    """
    # --- grid in x ---
    L = 2.0 * width_std * sigma * np.sqrt(max(T, 1e-12))       # total width in x
    dx = L / m
    x = x0 + dx * np.arange(m + 1)                             # j = 0..m
    S = K * np.exp(x)

    # --- coefficients (constant across i) ---
    alpha = 0.5 * sigma * sigma
    beta = (r - q) - 0.5 * sigma * sigma
    A = alpha / (dx*dx) - beta / (2*dx)    # lower (i-1)
    B = -2.0 * alpha / (dx*dx) - r         # main
    C = alpha / (dx*dx) + beta / (2*dx)    # upper (i+1)

    # --- choose N from gamma_max ---
    tau = T
    # γ = α Δτ / Δx^2  ⇒  N ≥ α T / (γ_max Δx^2)
    N = int(np.ceil((alpha * tau) / (gamma_max * dx * dx)))
    N = max(1, N)
    dtau = tau / N

    # --- assemble θ-scheme tri-diagonals (size m-1 for interior) ---
    loA = -theta * dtau * A * np.ones(m-1)
    mnA = 1.0 - theta * dtau * B * np.ones(m-1)
    upA = -theta * dtau * C * np.ones(m-1)

    loB = (1.0 - theta) * dtau * A * np.ones(m-1)
    mnB = 1.0 + (1.0 - theta) * dtau * B * np.ones(m-1)
    upB = (1.0 - theta) * dtau * C * np.ones(m-1)

    # --- terminal payoff at τ=0 ---
    V = np.maximum(S - K, 0.0)

    # --- time-stepping τ: 0 → T ---
    for n in range(N):
        tau_old = n * dtau
        tau_new = (n + 1) * dtau

        # RHS = B * V_old (interior 1..m-1)
        rhs = (mnB * V[1:m]).copy()
        rhs[1:] += loB[1:] * V[1:m-1]
        rhs[:-1] += upB[:-1] * V[2:m+0]

        # Old boundary contributions (Dirichlet)
        V_left_old = 0.0
        V_right_old = max(S[-1] * np.exp(-q * tau_old) - K * np.exp(-r * tau_old), 0.0)
        rhs[0] += loB[0] * V_left_old
        rhs[-1] += upB[-1] * V_right_old

        # Move new boundary terms from A-side to RHS
        V_left_new = 0.0
        V_right_new = max(S[-1] * np.exp(-q * tau_new) - K * np.exp(-r * tau_new), 0.0)
        rhs[0] -= loA[0] * V_left_new
        rhs[-1] -= upA[-1] * V_right_new

        # Solve and update
        V_new = V.copy()
        V_new[1:m] = _thomas(loA, mnA, upA, rhs)
        V_new[0] = V_left_new
        V_new[-1] = V_right_new
        V = V_new

    # interpolate back to S0
    price = float(np.interp(S0, S, V))
    return price, {"dx": dx, "N": N, "dtau": dtau, "x_grid": x, "S_grid": S}

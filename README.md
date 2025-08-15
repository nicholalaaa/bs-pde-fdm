# bs-pde-fdm

Black–Scholes **PDE** pricing with **finite differences**. Implements:
- Log-space **θ-scheme** (Explicit / **Crank–Nicolson** / Implicit) on a **uniform x=ln(S/K)** grid
- Parameters notation: **m** (space intervals), **x₀** (left boundary in x), **γ_max** (cap for γ)
- **Thomas** tridiagonal solver
- Closed-form **Black–Scholes** (for validation)
- Tests: CN price → BS as grid refines; Toeplitz tridiagonal eigenpairs

---

## Install
```bash
pip install -r requirements.txt
```

## Run (no package install needed)
```bash
export PYTHONPATH=$PWD/src         # Windows PowerShell: $env:PYTHONPATH="$PWD/src"
python -m examples.cn_call_demo    # demo using log-space θ-scheme
```

---

## Model & Discretization (matches HW)
We solve in **τ = T − t** on a **uniform x = ln(S/K)** grid with **m** intervals:

- Grid:  
  \(x_j = x_0 + j\,\Delta x,\; j=0..m\), where the domain width is about \(2\,w\,\sigma\sqrt{T}\) (default \(w=4\)).  
- Time steps: chosen so the CFL-like ratio  
  \[
  \gamma \;=\; \frac{\alpha\,\Delta\tau}{(\Delta x)^2}, \quad \alpha=\frac{\sigma^2}{2}
  \]
  satisfies **\(\gamma \le \gamma_{\max}\)**. We compute **N** from \(m, x_0, \gamma_{\max}\).
- PDE in log-space (constant coefficients):  
  \(V_\tau = \alpha V_{xx} + \beta V_x - rV,\;\; \alpha=\tfrac{\sigma^2}{2},\; \beta=(r-q)-\tfrac{\sigma^2}{2}\)

**Boundaries (call):** \(V(x_{\min},\tau)=0\), \(V(x_{\max},\tau)\approx S_{\max}e^{-q\tau}-Ke^{-r\tau}\) with \(S_{\max}=K e^{x_{\max}}\).  
**Terminal:** \(V(x,0)=\max(Ke^x-K,0)\).

---

## Quick demo
```python
from bs_pde_fdm.fdm_theta_log import price_call_theta_logspace
from bs_pde_fdm.bs_closed_form import bs_price
import numpy as np

S0=120.0, K=119.0; T=7/365; r=0.005; q=0.02; sigma=0.10
m = 20
x0 = -0.05   # left boundary in x
gamma_max = 24
theta = 0.5                               # Crank–Nicolson

p_fdm, meta = price_call_theta_logspace(S0,K,T,r,q,sigma,m,x0,gamma_max,theta)
p_bs = bs_price(S0,K,T,r,sigma,q=q,otype="call")
print(f"θ-scheme (log) price = {p_fdm:.6f} | BS = {p_bs:.6f} | N={meta['N']} | dx={meta['dx']:.5f}")
```

---

## Package layout
```
src/bs_pde_fdm/
  fdm_theta_log.py      # log-space θ-scheme (uses m, x0, gamma_max)
  bs_closed_form.py     # Black–Scholes (for validation)
  toeplitz_tridiag.py   # tridiagonal Toeplitz eigenpairs (analytic)
examples/
  cn_call_demo.py       # demo using log-space solver
tests/
  test_cn_convergence.py # CN (log-space) → BS; checks γ ≤ γ_max
  test_toeplitz.py       # eigenvalues vs NumPy
vba/
  HW11 Numerical Methods.xlsm  # VBA macro file developed in numerical course
```

---

## Tests
```bash
export PYTHONPATH=$PWD/src
pytest -q
```

---

## Notes & tips
- Start with **θ=0.5** (CN). If you explore θ≠0.5, expect explicit (θ=0) to need small γ.
- Choose **x₀** so the domain \([x_0, x_0+2w\sigma\sqrt{T}]\) covers the payoff mass (default \(w=4\) works well).
- Increasing **m** refines space; **γ_max** lowers Δτ (increases **N**).

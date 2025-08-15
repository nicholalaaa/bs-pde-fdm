# bs-pde-fdm

Black–Scholes **PDE** pricing with **finite differences**. Implements:
- **BS → heat-equation transform** helpers & boundary builders
- **θ-scheme** (Explicit, **Crank–Nicolson**, Implicit) on an S-grid
- **Thomas** tridiagonal solver
- Closed-form **Black–Scholes** (for validation)
- Examples and tests (convergence vs grid, stability vs θ)

## Install
```bash
pip install -r requirements.txt
```

## Quick demo
```bash
python -m examples.cn_call_demo
```
You should see a European call price from CN-FDM and the closed-form BS price.

## Modules
- `bs_closed_form.py` — Black–Scholes price for calls/puts with dividend yield
- `fdm_cn.py` — θ-scheme grid builder + stepper + Thomas solver
- `bs_heat_transform.py` — helpers for the BS→heat change of variables & boundaries
- `toeplitz_tridiag.py` — analytic eigenvalues/vectors for tridiagonal Toeplitz (with validation)

## Notes
- We discretize the **BS PDE** directly on an S-grid for the solver, which keeps the code approachable.
- The heat transform module shows the mapping and boundary logic in log-space (didactic; optional for running CN).
- Boundary conditions follow standard asymptotics: for calls, \(V(0,t)=0\), \(V(S_{max},t)≈S_{max}e^{-qτ}-Ke^{-rτ}\).
- Tests compare CN price to closed-form BS and check Toeplitz eigenpairs against `numpy.linalg`.


from bs_pde_fdm.fdm_cn import price_european_call_cn
from bs_pde_fdm.bs_closed_form import bs_price


def main():
    S0, K, T, r, q, sigma = 120.0, 119.0, 7/365, 0.005, 0.02, 0.1

    price_cn, Sgrid, Vgrid = price_european_call_cn(
        S0, K, T, r, q, sigma,
        Smax_mult=5.0, M=400, N=400, theta=0.5
    )
    price_bs = bs_price(S0, K, T, r, sigma, q=q, otype="call")

    print(f"Crank-Nicolson price: {price_cn:.6f}")
    print(f"Black-Scholes price:  {price_bs:.6f}")
    print(f"Abs error:            {abs(price_cn - price_bs):.6e}")


if __name__ == "__main__":
    main()

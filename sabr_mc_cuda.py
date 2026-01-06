import cupy as cp
import numpy as np
import time


# Original version with all random numbers pre-generated (less memory efficient)
# def sabr_mc_cuda(
#     F0, alpha0, beta, rho, nu,
#     T=30/365,                     # maturity
#     n_paths=200_000,              # Monte Carlo paths
#     n_steps=200,                  # time steps
# ):
#     dt = T / n_steps
#     sqrt_dt = np.sqrt(dt)

#     # Allocate paths on GPU
#     F = cp.full(n_paths, F0, dtype=cp.float64)
#     alpha = cp.full(n_paths, alpha0, dtype=cp.float64)

#     # Pre-generate random numbers on GPU
#     Z1 = cp.random.standard_normal((n_steps, n_paths))
#     Z2 = cp.random.standard_normal((n_steps, n_paths))

#     # Correlate Z2
#     Z2_corr = rho * Z1 + cp.sqrt(1 - rho**2) * Z2

#     # Time stepping (Euler)
#     for t in range(n_steps):
#         dW1 = Z1[t] * sqrt_dt
#         dW2 = Z2_corr[t] * sqrt_dt

#         # Update volatility and price
#         alpha = alpha * cp.exp(-0.5 * nu**2 * dt + nu * dW2)
#         F = F + alpha * cp.power(F, beta) * dW1

#     return F


# Memory-efficient version generating random numbers on-the-fly (Higher performance for larger n_paths)
def sabr_mc_cuda(
    F0, alpha0, beta, rho, nu,
    T=30/365,
    n_paths=200_000,
    n_steps=250
):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Allocate price & vol paths on GPU
    F = cp.full(n_paths, F0, dtype=cp.float32)
    alpha = cp.full(n_paths, alpha0, dtype=cp.float32)

    for _ in range(n_steps):
        Z1 = cp.random.standard_normal(n_paths, dtype=cp.float32)
        Z2 = cp.random.standard_normal(n_paths, dtype=cp.float32)
        Z2_corr = rho * Z1 + cp.sqrt(1 - rho**2) * Z2

        dW1 = Z1 * sqrt_dt
        dW2 = Z2_corr * sqrt_dt

        alpha = alpha * cp.exp(-0.5 * nu**2 * dt + nu * dW2) 
        F = F + alpha * (F ** beta) * dW1

    return F


if __name__ == "__main__":
    # SABR parameters (inputs)
    F0 = 22000          # Nifty forward price example
    alpha0 = 0.25       # initial volatility
    beta = 0.5          # elasticity
    rho = -0.3          # correlation b/w price and vol
    nu = 0.5            # vol of vol

    total_start = time.time()
    sim_start = time.time()
    result_paths = sabr_mc_cuda(F0, alpha0, beta, rho, nu,
                                T=30/365,
                                n_paths=4_000_000,
                                n_steps=250)
    cp.cuda.Stream.null.synchronize()     # wait for GPU to finish

    sim_end = time.time()
    price_start = time.time()
    
    # Compute option price from terminal distribution
    K = 22000
    payoff = cp.maximum(result_paths - K, 0)
    call_price = cp.mean(payoff).get()  # bring result to CPU

    price_end = time.time()
    total_end = time.time()

    print(f"\nSABR MC call price: {call_price:.6f}")
    print(f"Monte Carlo simulation time : {sim_end - sim_start:.4f} sec")
    print(f"Payoff averaging time      : {price_end - price_start:.4f} sec")
    print(f"Total runtime              : {total_end - total_start:.4f} sec")
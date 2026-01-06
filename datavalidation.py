import numpy as np
import pandas as pd
import cupy as cp

from scipy.stats import norm
from scipy.optimize import brentq, minimize

# Load data
df = pd.read_csv("CE-09-04-2025.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Expiry"] = pd.to_datetime(df["Expiry"])

# # Choose one expiry date
# expiry = df["Expiry"].iloc[0]
# df = df[df["Expiry"] == expiry].copy()

# # choose calibration date N days before expiry
# N_DAYS_BEFORE = 10
# calib_date = df[df["Date"] <= expiry]["Date"].max() - pd.Timedelta(days=N_DAYS_BEFORE)

# df = df[df["Date"] == calib_date].copy()

# print("Calibration date:", calib_date)
# print("Rows after date filter:", len(df))

#=============================================================================================
expiry = df["Expiry"].iloc[0]

# target calibration date (e.g. 10 days before expiry)
target_date = expiry - pd.Timedelta(days=10)

# find nearest available trading date ≤ target_date
available_dates = df["Date"].unique()
valid_dates = available_dates[available_dates <= target_date]

if len(valid_dates) == 0:
    raise ValueError(
        f"No trading data available before target calibration date {target_date}"
    )

calib_date = valid_dates.max()

df = df[df["Date"] == calib_date].copy()

print("Calibration date selected:", calib_date)
print("Rows after date filter:", len(df))
#=============================================================================================


F0 = df["Underlying Value"].iloc[0]   # forward proxy
T = (expiry - calib_date).days / 365.0
print("Time to maturity (years):", T)


# T = (expiry - df["Date"].iloc[0]).days / 365.0

K = df["Strike Price"].values
market_price = df["Close"].values

def black76_put(F, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)

def implied_vol_put(price, F, K, T):
    intrinsic = max(K - F, 0)

    # No-arbitrage bounds
    if price < intrinsic + 1e-6:
        return np.nan
    if price > K:
        return np.nan

    def obj(sigma):
        return black76_put(F, K, T, sigma) - price

    try:
        return brentq(obj, 1e-6, 3.0)
    except ValueError:
        return np.nan


df["IV"] = [
    implied_vol_put(p, F0, k, T)
    for p, k in zip(df["Close"], df["Strike Price"])
]

# =========================
# FINAL SABR CALIBRATION FILTER
# =========================

print("\n--- FINAL FILTER START ---")
print("Initial rows:", len(df))

# Ensure numeric
df["Strike Price"] = pd.to_numeric(df["Strike Price"], errors="coerce")
df["IV"] = pd.to_numeric(df["IV"], errors="coerce")

# Drop NaNs
df = df.dropna(subset=["Strike Price", "IV"]).copy()
print("After NaN drop:", len(df))

# Log-moneyness
df["log_moneyness"] = np.log(df["Strike Price"] / F0)

# Tight moneyness window
df = df[df["log_moneyness"].abs() < 0.05].copy()
print("After moneyness filter:", len(df))

# ATM-centered strike band
df = df[(df["Strike Price"] > F0 - 600) &
        (df["Strike Price"] < F0 + 600)].copy()
print("After ATM band:", len(df))

# PUT sanity (removes deep OTM puts)
df = df[df["Strike Price"] > 0.97 * F0].copy()
print("After PUT filter:", len(df))

# IV sanity
df = df[(df["IV"] > 0.08) & (df["IV"] < 0.80)].copy()
print("After IV bounds:", len(df))

print("FINAL rows:", len(df))
print("Strike range:", df["Strike Price"].min(), df["Strike Price"].max())
print("--- FINAL FILTER END ---\n")

K = df["Strike Price"].values
market_iv = df["IV"].values

print("FINAL number of strikes:", len(df))
print("Strike range:", df["Strike Price"].min(), df["Strike Price"].max())


def sabr_hagan_vol(F, K, T, alpha, beta, rho, nu):
    if F == K:
        fk = F**(1-beta)
        return (alpha / fk) * (
            1 + (
                ((1-beta)**2 / 24) * (alpha**2 / fk**2)
                + (rho * beta * nu * alpha) / (4 * fk)
                + ((2 - 3*rho**2) * nu**2) / 24
            ) * T
        )

    logFK = np.log(F / K)
    FK = (F * K)**((1 - beta) / 2)

    z = (nu / alpha) * FK * logFK
    xz = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))

    A = alpha / (FK * (1 + ((1-beta)**2 / 24) * logFK**2))
    B = z / xz
    C = 1 + (
        ((1-beta)**2 / 24) * (alpha**2 / FK**2)
        + (rho * beta * nu * alpha) / (4 * FK)
        + ((2 - 3*rho**2) * nu**2) / 24
    ) * T

    return A * B * C


beta = 0.7   # FIXED (market standard)

weights = np.exp(-np.abs(np.log(K / F0)) * 5)

def objective(params):
    alpha, rho, nu = params
    model_iv = np.array([
        sabr_hagan_vol(F0, k, T, alpha, beta, rho, nu) for k in K
    ])
    return np.sum(weights * (model_iv - market_iv)**2)

atm_idx = np.argmin(np.abs(K - F0))
atm_iv = market_iv[atm_idx]

alpha0 = atm_iv * (F0 ** (1 - beta))
rho0 = -0.3
nu0 = 0.6


print("Number of strikes used:", len(K))
print("First 5 strikes:", K[:5])
print("First 5 IVs:", market_iv[:5])



res = minimize(
    objective,
    x0=[alpha0, rho0, nu0],
    bounds = [
        (0.10, 3.00),   # alpha 
        (-0.8, -0.1),   # rho
        (0.2, 2.0)      # nu
    ],
    method="L-BFGS-B"
)

alpha, rho, nu = res.x

print("\nCALIBRATED SABR PARAMETERS")
print(f"alpha = {alpha:.4f}")
print(f"beta  = {beta:.2f}")
print(f"rho   = {rho:.3f}")
print(f"nu    = {nu:.3f}")


def sabr_mc_cuda(
    F0, alpha0, beta, rho, nu, T,
    n_paths=12_000_000,
    n_steps=200
):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    F = cp.full(n_paths, F0, dtype=cp.float32)
    alpha = cp.full(n_paths, alpha0, dtype=cp.float32)

    for _ in range(n_steps):
        Z1 = cp.random.standard_normal(n_paths, dtype=cp.float32)
        Z2 = cp.random.standard_normal(n_paths, dtype=cp.float32)
        Z2_corr = rho * Z1 + cp.sqrt(1 - rho**2) * Z2

        alpha = alpha * cp.exp(-0.5 * nu**2 * dt + nu * Z2_corr * sqrt_dt)
        F = F + alpha * (F**beta) * Z1 * sqrt_dt

    return F


i = len(df) // 2
K_test = K[i]
market_price = df["Close"].iloc[i]
# market_price_test = market_price[i]


paths = sabr_mc_cuda(F0, alpha, beta, rho, nu, T)

payoff = cp.maximum(K_test - paths, 0)  # PUT payoff
mc_price = cp.mean(payoff).get()

print("\nPRICE VALIDATION")
print(f"Strike        : {K_test}")
print(f"Market price  : {market_price:.2f}")
print(f"SABR MC price : {mc_price:.2f}")

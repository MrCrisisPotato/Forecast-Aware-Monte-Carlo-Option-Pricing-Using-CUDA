from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

def black76_call_price(F, K, T, sigma, r=0.0):
    if sigma <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r*T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def implied_vol_black76_call(price, F, K, T, r=0.0):
    def objective(sigma):
        return black76_call_price(F, K, T, sigma, r) - price

    return brentq(objective, 1e-6, 5.0)  # vol between 0.01% and 500%


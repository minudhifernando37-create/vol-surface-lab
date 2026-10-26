"""Black-Scholes-Merton pricing, Greeks and implied volatility.

Every function accepts scalars or NumPy arrays (broadcasting works), so the
same code prices one option or a whole grid of options at once.

Notation
--------
S     : spot price of the underlying
K     : strike price
T     : time to expiry in years (must be > 0)
r     : continuously-compounded risk-free rate   (0.04  = 4%)
q     : continuous dividend yield                (0.013 = 1.3%)
sigma : annualised volatility                    (0.20  = 20%)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes-Merton price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    fwd_s = S * np.exp(-q * T)   # spot, discounted for dividends
    pv_k = K * np.exp(-r * T)    # present value of the strike
    if option_type == "call":
        return fwd_s * norm.cdf(d1) - pv_k * norm.cdf(d2)
    if option_type == "put":
        return pv_k * norm.cdf(-d2) - fwd_s * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Change in option value per $1 move in the underlying."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Change in delta per $1 move in the underlying (same for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (np.asarray(S) * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Change in option value per 1 vol point (i.e. sigma moving 0.01)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.asarray(S) * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def bs_theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Change in option value per calendar day that passes."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        annual = (decay - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        annual = (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return annual / 365.0


def implied_vol(price, S, K, T, r, option_type="call", q=0.0, lo=1e-4, hi=5.0):
    """Volatility that makes the Black-Scholes price equal the market price.

    Black-Scholes price rises monotonically with sigma, so a bracketing
    root-finder (Brent's method) is reliable. Returns NaN when the price sits
    outside no-arbitrage bounds, since no volatility can explain it.
    """
    if not (T > 0 and price > 0):
        return np.nan
    fwd_s, pv_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if option_type == "call":
        lower, upper = max(fwd_s - pv_k, 0.0), fwd_s
    else:
        lower, upper = max(pv_k - fwd_s, 0.0), pv_k
    if not (lower < price < upper):
        return np.nan

    def objective(sigma):
        return float(bs_price(S, K, T, r, sigma, option_type, q)) - price

    try:
        return brentq(objective, lo, hi, xtol=1e-8)
    except ValueError:  # no sign change inside [lo, hi]
        return np.nan

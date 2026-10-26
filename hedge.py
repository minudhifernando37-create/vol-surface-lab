"""Delta-hedging simulator.

Scenario: a trading desk SELLS one option at the implied vol, collects the
premium, and buys/sells the underlying stock at each rebalance so the book
stays delta-neutral until expiry.

The key lesson this demonstrates:
  * If the stock's realised vol equals the implied vol the desk sold at, the
    hedged P&L averages ~0 and its spread shrinks as you rebalance more often.
  * If realised vol > implied vol, the short-option desk loses money (it is
    short gamma); if realised < implied, it makes money. Delta hedging removes
    direction risk but leaves you trading VOLATILITY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pricing import bs_delta, bs_price


def _payoff(S, K, option_type):
    return np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)


def simulate_delta_hedge(S0=100.0, K=100.0, T=0.25, r=0.04, q=0.0,
                         sigma_implied=0.20, sigma_realized=0.20, mu=None,
                         option_type="call", n_rebalances=63, n_paths=2000,
                         tc_bps=0.0, seed=42) -> dict:
    """Simulate many stock paths (geometric Brownian motion) and hedge each one.

    n_rebalances : number of hedge trades over the option's life
    tc_bps       : transaction cost per trade, in basis points of notional
    mu           : real-world drift of the stock (defaults to r)

    Returns a dict of arrays shaped (n_paths, n_rebalances + 1) plus final P&L.
    """
    mu = r if mu is None else mu
    rng = np.random.default_rng(seed)
    n, dt, tc = n_rebalances, T / n_rebalances, tc_bps / 1e4
    t = np.linspace(0.0, T, n + 1)

    z = rng.standard_normal((n_paths, n))
    log_ret = (mu - q - 0.5 * sigma_realized**2) * dt + sigma_realized * np.sqrt(dt) * z
    S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)]))

    shares = np.zeros_like(S)
    cash = np.zeros_like(S)
    option_value = np.zeros_like(S)
    costs = np.zeros(n_paths)

    premium = float(bs_price(S0, K, T, r, sigma_implied, option_type, q))
    delta0 = float(bs_delta(S0, K, T, r, sigma_implied, option_type, q))
    option_value[:, 0] = premium
    shares[:, 0] = delta0                      # short option -> buy delta shares
    first_cost = tc * abs(delta0) * S0
    cash[:, 0] = premium - delta0 * S0 - first_cost
    costs += first_cost

    for i in range(1, n + 1):
        c = cash[:, i - 1] * np.exp(r * dt)              # interest on cash
        c += shares[:, i - 1] * S[:, i] * q * dt          # dividends on shares held
        if i < n:
            tau = T - t[i]
            new_delta = bs_delta(S[:, i], K, tau, r, sigma_implied, option_type, q)
            trade = new_delta - shares[:, i - 1]
            cost = tc * np.abs(trade) * S[:, i]
            c -= trade * S[:, i] + cost
            costs += cost
            shares[:, i] = new_delta
            option_value[:, i] = bs_price(S[:, i], K, tau, r, sigma_implied, option_type, q)
        else:                                              # expiry: settle, no trade
            shares[:, i] = shares[:, i - 1]
            option_value[:, i] = _payoff(S[:, i], K, option_type)
        cash[:, i] = c

    # Mark-to-market value of the whole book: cash + stock - option we owe
    book_value = cash + shares * S - option_value
    hedged_pnl = book_value[:, -1]
    unhedged_pnl = premium * np.exp(r * T) - _payoff(S[:, -1], K, option_type)

    return dict(t=t, S=S, shares=shares, cash=cash, option_value=option_value,
                book_value=book_value, hedged_pnl=hedged_pnl,
                unhedged_pnl=unhedged_pnl, premium=premium, costs=costs)


def pnl_summary(result: dict) -> pd.DataFrame:
    """Side-by-side risk statistics for hedged vs unhedged P&L."""
    prem = result["premium"]
    rows = {}
    for label, pnl in (("Delta-hedged", result["hedged_pnl"]),
                       ("Unhedged (naked short)", result["unhedged_pnl"])):
        rows[label] = {
            "Mean P&L": pnl.mean(),
            "Std dev of P&L": pnl.std(),
            "Std dev as % of premium": 100 * pnl.std() / prem,
            "5% worst case (VaR 95%)": np.percentile(pnl, 5),
            "Worst path": pnl.min(),
        }
    return pd.DataFrame(rows)


def frequency_experiment(freqs=(4, 12, 52, 252), **kwargs) -> pd.DataFrame:
    """Hedging error vs rebalancing frequency (theory: std falls ~ 1/sqrt(N))."""
    T = kwargs.get("T", 0.25)
    out = []
    for f in freqs:
        n = max(1, int(round(f * T)))
        res = simulate_delta_hedge(n_rebalances=n, **kwargs)
        out.append({"Rebalances per year": f, "Trades over life": n,
                    "Mean P&L": res["hedged_pnl"].mean(),
                    "Std dev of P&L": res["hedged_pnl"].std(),
                    "Avg transaction costs": res["costs"].mean()})
    return pd.DataFrame(out)

"""Option-chain loading: live from Yahoo Finance, or a synthetic demo chain.

Both loaders return (spot, chain) where chain is a DataFrame with one row per
option and these columns:
    strike, type ('call'/'put'), expiry, T (years), bid, ask, last, mid,
    price_source ('quote' or 'last'), spread_pct, moneyness, volume,
    open_interest, yahoo_iv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pricing import bs_price


def clean_chain(chain: pd.DataFrame, spot: float, min_moneyness=0.7,
                max_moneyness=1.3, max_spread_pct=0.5,
                allow_last_price=True) -> pd.DataFrame:
    """Keep liquid, out-of-the-money options and attach a mid price.

    Why out-of-the-money only? OTM options are the most traded, and in-the-money
    prices are mostly intrinsic value, which makes their implied vol noisy.
    So we use puts below spot and calls above spot - the standard desk practice.
    """
    df = chain.copy()
    has_quote = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
    df["mid"] = np.where(has_quote, (df["bid"] + df["ask"]) / 2, df["last"])
    df["price_source"] = np.where(has_quote, "quote", "last")
    df["spread_pct"] = np.where(has_quote, (df["ask"] - df["bid"]) / df["mid"], np.nan)
    df["moneyness"] = df["strike"] / spot

    liquid = has_quote & (df["spread_pct"] <= max_spread_pct)
    if allow_last_price:  # outside US market hours Yahoo often shows bid/ask = 0
        liquid |= ~has_quote & (df["last"] > 0)
    otm = ((df["type"] == "put") & (df["strike"] < spot)) | \
          ((df["type"] == "call") & (df["strike"] >= spot))
    in_range = df["moneyness"].between(min_moneyness, max_moneyness)

    return df[liquid & otm & in_range].reset_index(drop=True)


def fetch_option_chain(ticker="SPY", max_expiries=8, min_days=7, max_days=400):
    """Download the option chain for `ticker` from Yahoo Finance via yfinance."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history returned for {ticker!r}.")
    spot = float(hist["Close"].iloc[-1])

    today = pd.Timestamp.today().normalize()
    expiries = [e for e in tk.options
                if min_days <= (pd.Timestamp(e) - today).days <= max_days]
    if not expiries:
        raise RuntimeError(f"No option expiries found for {ticker!r}.")
    # Spread the chosen expiries across the whole range instead of taking only the nearest
    idx = np.unique(np.linspace(0, len(expiries) - 1, max_expiries).round().astype(int))
    expiries = [expiries[i] for i in idx]

    frames = []
    for exp in expiries:
        oc = tk.option_chain(exp)
        for df, kind in ((oc.calls, "call"), (oc.puts, "put")):
            df = df[["strike", "bid", "ask", "lastPrice", "volume",
                     "openInterest", "impliedVolatility"]].copy()
            df["type"], df["expiry"] = kind, exp
            frames.append(df)

    chain = pd.concat(frames, ignore_index=True).rename(columns={
        "lastPrice": "last", "openInterest": "open_interest",
        "impliedVolatility": "yahoo_iv"})
    chain["T"] = (pd.to_datetime(chain["expiry"]) - today).dt.days / 365.0
    return spot, clean_chain(chain, spot)


def synthetic_chain(spot=100.0, r=0.04, q=0.0, seed=7):
    """A realistic-looking equity option chain for offline testing and demos.

    Vol has an equity-style shape: a downward skew (crash protection makes
    low-strike puts expensive), mild smile curvature, both flattening with
    maturity, and an upward-sloping term structure. Prices get a little noise
    and a bid/ask spread, so the IV solver has to work for its answer.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for days in (14, 30, 60, 91, 182, 273, 365):
        T = days / 365
        expiry = (pd.Timestamp.today().normalize() + pd.Timedelta(days=days)).date().isoformat()
        for K in np.round(spot * np.arange(0.70, 1.301, 0.025), 2):
            k = np.log(K / spot)
            sigma = max(0.16 + 0.03 * np.sqrt(T)
                        - 0.12 / np.sqrt(T) * k
                        + 0.25 / np.sqrt(T) * k**2, 0.05)
            for kind in ("call", "put"):
                fair = float(bs_price(spot, K, T, r, sigma, kind, q))
                if fair < 0.01:
                    continue
                mid = fair * (1 + rng.normal(0, 0.01))
                half_spread = max(0.005, 0.02 * mid)
                rows.append(dict(strike=K, type=kind, expiry=expiry, T=T,
                                 bid=round(mid - half_spread, 2),
                                 ask=round(mid + half_spread, 2),
                                 last=round(mid, 2), volume=np.nan,
                                 open_interest=np.nan, yahoo_iv=np.nan,
                                 true_iv=sigma))
    return spot, clean_chain(pd.DataFrame(rows), spot)

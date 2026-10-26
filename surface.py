"""Implied-volatility calculation and volatility-surface charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

from pricing import implied_vol


def add_implied_vols(chain: pd.DataFrame, spot: float, r: float, q: float = 0.0) -> pd.DataFrame:
    """Back out an implied vol for every option in the chain."""
    df = chain.copy()
    df["iv"] = [implied_vol(p, spot, K, T, r, kind, q)
                for p, K, T, kind in zip(df["mid"], df["strike"], df["T"], df["type"])]
    df = df.dropna(subset=["iv"])
    df = df[df["iv"].between(0.01, 3.0)].copy()
    df["log_moneyness"] = np.log(df["strike"] / spot)
    df["days"] = (df["T"] * 365).round().astype(int)
    return df.reset_index(drop=True)


def surface_grid(df: pd.DataFrame, n_x=45, n_t=45):
    """Interpolate the scattered (moneyness, T, iv) points onto a regular grid."""
    x = np.linspace(df["moneyness"].min(), df["moneyness"].max(), n_x)
    t = np.linspace(df["T"].min(), df["T"].max(), n_t)
    X, Tg = np.meshgrid(x, t)
    Z = griddata((df["moneyness"], df["T"]), df["iv"], (X, Tg), method="linear")
    return X, Tg, Z


def atm_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """ATM implied vol per expiry, interpolated at moneyness = 1."""
    out = []
    for (expiry, T), g in df.groupby(["expiry", "T"]):
        g = g.sort_values("moneyness")
        if g["moneyness"].min() <= 1.0 <= g["moneyness"].max():
            out.append(dict(expiry=expiry, T=T, days=round(T * 365),
                            atm_iv=float(np.interp(1.0, g["moneyness"], g["iv"]))))
    return pd.DataFrame(out).sort_values("T").reset_index(drop=True)


def plot_surface(df: pd.DataFrame, title="Implied volatility surface") -> go.Figure:
    X, Tg, Z = surface_grid(df)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Tg * 365, z=Z * 100, colorscale="Viridis", opacity=0.92,
        colorbar=dict(title="IV %"),
        hovertemplate="K/S %{x:.3f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=df["moneyness"], y=df["T"] * 365, z=df["iv"] * 100, mode="markers",
        marker=dict(size=2.5, color="white", line=dict(width=0.5, color="black")),
        name="Market points",
        hovertemplate="K/S %{x:.3f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
    fig.update_layout(
        title=title, height=650, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        scene=dict(xaxis_title="Moneyness (K / S)", yaxis_title="Days to expiry",
                   zaxis_title="Implied vol (%)",
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
    return fig


def plot_smiles(df: pd.DataFrame) -> go.Figure:
    """One IV-vs-moneyness curve per expiry: shows the skew/smile directly."""
    fig = go.Figure()
    for (expiry, days), g in df.groupby(["expiry", "days"]):
        g = g.sort_values("moneyness")
        fig.add_trace(go.Scatter(x=g["moneyness"], y=g["iv"] * 100, mode="lines+markers",
                                 marker=dict(size=4), name=f"{expiry} ({days}d)"))
    fig.add_vline(x=1.0, line_dash="dot", line_color="grey")
    fig.update_layout(title="Volatility smile / skew by expiry", height=450,
                      xaxis_title="Moneyness (K / S)", yaxis_title="Implied vol (%)")
    return fig


def plot_term_structure(df: pd.DataFrame) -> go.Figure:
    ts = atm_term_structure(df)
    fig = go.Figure(go.Scatter(x=ts["days"], y=ts["atm_iv"] * 100, mode="lines+markers"))
    fig.update_layout(title="ATM volatility term structure", height=400,
                      xaxis_title="Days to expiry", yaxis_title="ATM implied vol (%)")
    return fig


def plot_iv_check(df: pd.DataFrame) -> go.Figure:
    """Our solver's IV against Yahoo's published IV - a quick validation."""
    d = df.dropna(subset=["yahoo_iv"])
    d = d[d["yahoo_iv"] > 0]
    fig = go.Figure(go.Scatter(x=d["yahoo_iv"] * 100, y=d["iv"] * 100, mode="markers",
                               marker=dict(size=5, opacity=0.6),
                               text=d["expiry"] + " K=" + d["strike"].astype(str)))
    if len(d):
        lo, hi = d[["iv", "yahoo_iv"]].min().min() * 100, d[["iv", "yahoo_iv"]].max().max() * 100
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dot", color="grey"))
    fig.update_layout(title="Our implied vol vs Yahoo's", height=420,
                      xaxis_title="Yahoo IV (%)", yaxis_title="Our IV (%)")
    return fig

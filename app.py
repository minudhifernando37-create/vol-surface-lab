"""Streamlit dashboard: volatility surface + delta-hedging simulator.

Run with:   streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import fetch_option_chain, synthetic_chain
from hedge import frequency_experiment, pnl_summary, simulate_delta_hedge
from surface import (add_implied_vols, atm_term_structure, plot_iv_check,
                     plot_smiles, plot_surface, plot_term_structure)

st.set_page_config(page_title="Vol Surface Lab", layout="wide")
st.title("Equity Derivatives: Volatility Surface & Delta Hedging")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Market data")
    source = st.radio("Data source", ["Live (Yahoo Finance)", "Synthetic demo"], index=1)
    ticker = st.text_input("Ticker", "SPY").strip().upper()
    max_expiries = st.slider("Number of expiries", 3, 15, 8)
    st.header("Model inputs")
    r = st.number_input("Risk-free rate (decimal)", value=0.040, step=0.0025, format="%.4f",
                        help="Use a T-bill yield matching your option maturities.")
    q = st.number_input("Dividend yield (decimal)", value=0.000, step=0.001, format="%.4f",
                        help="e.g. roughly 0.012 for SPY. 0 is fine to start.")


@st.cache_data(ttl=600, show_spinner="Loading option chain...")
def load_chain(source, ticker, max_expiries, r, q):
    if source.startswith("Live"):
        spot, chain = fetch_option_chain(ticker, max_expiries=max_expiries)
    else:
        spot, chain = synthetic_chain(r=r, q=q)
    return spot, add_implied_vols(chain, spot, r, q)


try:
    spot, df = load_chain(source, ticker, max_expiries, r, q)
except Exception as exc:  # network errors, rate limits, bad tickers...
    st.error(f"Could not load data: {exc}")
    st.info("Try another ticker, try again in a minute, or switch to 'Synthetic demo'.")
    st.stop()

if df.empty:
    st.warning("No usable options after cleaning. US markets may be closed - try 'Synthetic demo'.")
    st.stop()

ts = atm_term_structure(df)
atm_iv = float(ts["atm_iv"].iloc[0]) if len(ts) else float(df["iv"].median())
label = ticker if source.startswith("Live") else "Synthetic"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{label} spot", f"{spot:,.2f}")
c2.metric("Options used", f"{len(df):,}")
c3.metric("Expiries", df["expiry"].nunique())
c4.metric("Front-month ATM IV", f"{atm_iv:.1%}")

stale = (df["price_source"] == "last").mean()
if stale > 0.2:
    st.caption(f"Note: {stale:.0%} of options are priced from last trade, not live bid/ask "
               "(US market likely closed), so the surface will be noisier.")

tab_surface, tab_smile, tab_hedge, tab_data = st.tabs(
    ["Volatility surface", "Smiles & term structure", "Delta hedging", "Data & validation"])

# ---------------------------------------------------------------- surface
with tab_surface:
    st.plotly_chart(plot_surface(df, f"{label} implied volatility surface"))
    st.markdown(
        "**How to read it:** each white dot is one option's implied vol. Equity "
        "surfaces usually slope *down* from left to right (the **skew**): low-strike "
        "puts are bid up as crash insurance. The skew is steepest at short maturities.")

# ---------------------------------------------------------------- smiles
with tab_smile:
    st.plotly_chart(plot_smiles(df))
    st.plotly_chart(plot_term_structure(df))

# ---------------------------------------------------------------- hedging
with tab_hedge:
    st.markdown(
        "The desk **sells one option** at the implied vol below and delta-hedges it "
        "with the stock until expiry. Change *realised* vol to see who wins: if the "
        "stock moves more than the vol you sold, hedging costs more than the premium.")
    h1, h2, h3 = st.columns(3)
    option_type = h1.selectbox("Option sold", ["call", "put"])
    K = h1.number_input("Strike", value=float(round(spot)), step=1.0)
    days = h1.slider("Days to expiry", 10, 365, 90)
    sigma_imp = h2.slider("Implied vol sold at (%)", 5.0, 80.0,
                          float(round(min(max(atm_iv * 100, 5.0), 80.0), 1)), 0.5) / 100
    sigma_real = h2.slider("Realised vol of the stock (%)", 5.0, 80.0,
                           float(round(min(max(atm_iv * 100, 5.0), 80.0), 1)), 0.5) / 100
    freq_label = h3.selectbox("Rebalance", ["Daily", "Weekly", "Monthly"])
    tc_bps = h3.slider("Transaction cost (bps per trade)", 0.0, 20.0, 1.0, 0.5)
    n_paths = h3.select_slider("Simulated paths", [200, 500, 1000, 2000, 5000], 1000)

    T = days / 365
    per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}[freq_label]
    n_reb = max(1, int(round(per_year * T)))
    params = dict(S0=spot, K=K, T=T, r=r, q=q, sigma_implied=sigma_imp,
                  sigma_realized=sigma_real, option_type=option_type,
                  n_paths=n_paths, tc_bps=tc_bps)
    res = simulate_delta_hedge(n_rebalances=n_reb, **params)

    m1, m2, m3 = st.columns(3)
    m1.metric("Premium received", f"{res['premium']:.2f}")
    m2.metric("Mean hedged P&L", f"{res['hedged_pnl'].mean():.2f}")
    m3.metric("Hedged P&L std dev", f"{res['hedged_pnl'].std():.2f}",
              help=f"Unhedged std dev: {res['unhedged_pnl'].std():.2f}")

    # One sample path, step by step
    days_axis = res["t"] * 365
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Stock price (sample path)",
                                        "Shares held = hedge ratio (delta)",
                                        "Book value: cash + stock - option owed"))
    fig.add_trace(go.Scatter(x=days_axis, y=res["S"][0], name="Stock"), row=1, col=1)
    fig.add_hline(y=K, line_dash="dot", line_color="grey", row=1, col=1)
    fig.add_trace(go.Scatter(x=days_axis, y=res["shares"][0], name="Shares",
                             line_shape="hv"), row=2, col=1)
    fig.add_trace(go.Scatter(x=days_axis, y=res["book_value"][0], name="Book value"),
                  row=3, col=1)
    fig.update_layout(height=720, showlegend=False)
    fig.update_xaxes(title_text="Days since trade", row=3, col=1)
    st.plotly_chart(fig)

    # Distribution across all paths
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=res["unhedged_pnl"], name="Unhedged", opacity=0.55, nbinsx=80))
    hist.add_trace(go.Histogram(x=res["hedged_pnl"], name="Delta-hedged", opacity=0.75, nbinsx=80))
    hist.update_layout(barmode="overlay", height=420, title="Final P&L across all paths",
                       xaxis_title="P&L per option", yaxis_title="Number of paths")
    st.plotly_chart(hist)
    st.dataframe(pnl_summary(res).style.format("{:.2f}"))

    st.subheader("How often should the desk rebalance?")
    freq_df = frequency_experiment(freqs=(4, 12, 52, 252), **params)
    st.dataframe(freq_df.style.format({"Mean P&L": "{:.3f}", "Std dev of P&L": "{:.3f}",
                                       "Avg transaction costs": "{:.3f}"}), hide_index=True)
    st.caption("More frequent hedging shrinks P&L noise (roughly with the square root "
               "of the number of trades) but racks up transaction costs - the core "
               "trade-off a real desk manages.")

# ---------------------------------------------------------------- data
with tab_data:
    if df["yahoo_iv"].notna().any():
        st.plotly_chart(plot_iv_check(df))
        st.caption("Points near the dotted line mean our Black-Scholes solver agrees with "
                   "Yahoo. Gaps usually come from different rate/dividend assumptions or "
                   "Yahoo using last price instead of mid.")
    show = ["expiry", "days", "type", "strike", "moneyness", "bid", "ask", "mid",
            "price_source", "iv", "yahoo_iv", "volume", "open_interest"]
    st.dataframe(df[show], hide_index=True)
    st.download_button("Download as CSV", df[show].to_csv(index=False),
                       file_name=f"{label.lower()}_implied_vols.csv", mime="text/csv")

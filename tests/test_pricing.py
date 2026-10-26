import numpy as np
import pytest

from pricing import bs_delta, bs_gamma, bs_price, bs_vega, implied_vol

S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.04, 0.01, 0.25


def test_textbook_value():
    # Hull's classic example: S=K=100, T=1, r=5%, sigma=20% -> call = 10.4506
    assert float(bs_price(100, 100, 1, 0.05, 0.20, "call")) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    call = float(bs_price(S, K, T, r, sigma, "call", q))
    put = float(bs_price(S, K, T, r, sigma, "put", q))
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(kind):
    for true_vol in (0.08, 0.2, 0.55, 1.2):
        price = float(bs_price(S, K, T, r, true_vol, kind, q))
        assert implied_vol(price, S, K, T, r, kind, q) == pytest.approx(true_vol, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(0.0, S, K, T, r, "call", q))
    assert np.isnan(implied_vol(S * 2, S, K, T, r, "call", q))  # above max possible price


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_finite_difference(kind):
    h = 1e-3
    fd = (bs_price(S + h, K, T, r, sigma, kind, q) - bs_price(S - h, K, T, r, sigma, kind, q)) / (2 * h)
    assert float(bs_delta(S, K, T, r, sigma, kind, q)) == pytest.approx(float(fd), abs=1e-6)


def test_gamma_and_vega_match_finite_difference():
    h = 1e-3
    fd_gamma = (bs_delta(S + h, K, T, r, sigma, "call", q) - bs_delta(S - h, K, T, r, sigma, "call", q)) / (2 * h)
    assert float(bs_gamma(S, K, T, r, sigma, q)) == pytest.approx(float(fd_gamma), abs=1e-6)
    fd_vega = (bs_price(S, K, T, r, sigma + 1e-4, "call", q) - bs_price(S, K, T, r, sigma - 1e-4, "call", q)) / 2e-4 / 100
    assert float(bs_vega(S, K, T, r, sigma, q)) == pytest.approx(float(fd_vega), abs=1e-6)


def test_vectorised_pricing():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_price(S, strikes, T, r, sigma, "call", q)
    assert prices.shape == (3,) and np.all(np.diff(prices) < 0)  # calls cheaper as strike rises

from hedge import frequency_experiment, simulate_delta_hedge

BASE = dict(S0=100, K=100, T=0.25, r=0.04, sigma_implied=0.20, n_paths=4000, seed=1)


def test_fair_vol_gives_near_zero_mean_pnl():
    res = simulate_delta_hedge(sigma_realized=0.20, n_rebalances=252, **BASE)
    assert abs(res["hedged_pnl"].mean()) < 0.05 * res["premium"]


def test_hedging_reduces_risk():
    res = simulate_delta_hedge(sigma_realized=0.20, n_rebalances=63, **BASE)
    assert res["hedged_pnl"].std() < 0.2 * res["unhedged_pnl"].std()


def test_more_frequent_hedging_reduces_error():
    df = frequency_experiment(freqs=(12, 52, 252), sigma_realized=0.20, **BASE)
    assert df["Std dev of P&L"].is_monotonic_decreasing


def test_short_vol_loses_when_realised_exceeds_implied():
    high = simulate_delta_hedge(sigma_realized=0.30, n_rebalances=63, **BASE)
    low = simulate_delta_hedge(sigma_realized=0.10, n_rebalances=63, **BASE)
    assert high["hedged_pnl"].mean() < 0 < low["hedged_pnl"].mean()


def test_transaction_costs_reduce_pnl():
    free = simulate_delta_hedge(n_rebalances=63, tc_bps=0, **BASE)
    costly = simulate_delta_hedge(n_rebalances=63, tc_bps=10, **BASE)
    assert costly["hedged_pnl"].mean() < free["hedged_pnl"].mean()

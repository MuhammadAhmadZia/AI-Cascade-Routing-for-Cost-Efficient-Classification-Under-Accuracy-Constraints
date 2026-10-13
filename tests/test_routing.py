"""Unit tests for the routing maths and the published results.

Run with:  pytest -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing import (aucac, at_retention, ece, error_structure,
                         interior_minimum_possible, mcnemar, oracle_frontier,
                         random_frontier, sweep)
from src.analysis import scaling_reanalysis

RESULTS = Path(__file__).resolve().parent.parent / "results"


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 400
    conf = rng.beta(2, 2, n)
    small_ok = rng.binomial(1, np.clip(conf * 0.9 + 0.05, 0, 1))
    large_ok = np.where(small_ok == 1, rng.binomial(1, 0.95, n),
                        rng.binomial(1, 0.6, n))
    cost_s = np.full(n, 1.7 * 800, dtype=float)
    cost_l = np.full(n, 14.0 * 800, dtype=float)
    return conf, small_ok, large_ok, cost_s, cost_l


# --------------------------------------------------------------------------
# frontier invariants
# --------------------------------------------------------------------------
def test_sweep_endpoints_are_always_small_and_always_large(toy):
    conf, s, l, cs, cl = toy
    f = sweep(conf, s, l, cs, cl)
    assert f.escalation_rate.iloc[0] == pytest.approx(0.0)
    assert f.escalation_rate.iloc[-1] == pytest.approx(1.0)
    assert f.accuracy.iloc[0] == pytest.approx(s.mean())
    assert f.accuracy.iloc[-1] == pytest.approx(l.mean())


def test_small_tier_cost_is_always_paid(toy):
    """Escalated items still cost a small-model pass. Omitting it would
    overstate savings, which is the accounting error the paper warns about."""
    conf, s, l, cs, cl = toy
    f = sweep(conf, s, l, cs, cl)
    assert f.cost.min() >= cs.sum() - 1e-9
    assert f.cost.iloc[-1] == pytest.approx(cs.sum() + cl.sum())


def test_cost_increases_monotonically_with_escalation(toy):
    conf, s, l, cs, cl = toy
    f = sweep(conf, s, l, cs, cl).sort_values("escalation_rate")
    assert (f.cost.diff().dropna() >= -1e-9).all()


def test_oracle_dominates_router_dominates_random(toy):
    conf, s, l, cs, cl = toy
    router = aucac(sweep(conf, s, l, cs, cl))
    rand = aucac(random_frontier(s, l, cs, cl))
    orac = aucac(oracle_frontier(s, l, cs, cl))
    assert orac >= router >= rand


def test_oracle_can_exceed_always_large(toy):
    """Where the small model is right and the large model wrong, the oracle
    curve rises above always-large accuracy. A confidence router cannot."""
    conf, s, l, cs, cl = toy
    o = oracle_frontier(s, l, cs, cl)
    assert o.accuracy.max() >= l.mean() - 1e-9


# --------------------------------------------------------------------------
# signal orientation
# --------------------------------------------------------------------------
def test_inverted_confidence_falls_below_random(toy):
    """Escalation fires below a threshold, so a sign error silently inverts the
    router. This test is the guard against that."""
    conf, s, l, cs, cl = toy
    good = aucac(sweep(conf, s, l, cs, cl))
    bad = aucac(sweep(-conf, s, l, cs, cl))
    rand = aucac(random_frontier(s, l, cs, cl))
    assert bad < rand < good


# --------------------------------------------------------------------------
# calibration and partitions
# --------------------------------------------------------------------------
def test_ece_is_zero_for_perfect_calibration():
    conf = np.repeat([0.1, 0.5, 0.9], 2000)
    rng = np.random.default_rng(0)
    correct = rng.binomial(1, conf)
    assert ece(conf, correct) < 0.02


def test_error_structure_partitions_the_workload(toy):
    _c, s, l, _cs, _cl = toy
    e = error_structure(s, l)
    both_right = int(((s == 1) & (l == 1)).sum())
    assert (e["escalation_can_help"] + e["escalation_would_harm"]
            + e["both_wrong"] + both_right) == e["n"]


def test_mcnemar_matches_hand_computation():
    s = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    l = np.array([0, 1, 1, 1, 1, 0, 1, 1])
    m = mcnemar(s, l)
    assert m["small_only"] == 1 and m["large_only"] == 4
    assert m["chi2"] == pytest.approx((abs(1 - 4) - 1) ** 2 / 5)


# --------------------------------------------------------------------------
# Section 3.7: when routing can pay
# --------------------------------------------------------------------------
def test_interior_minimum_condition_fails_for_every_published_config():
    """k < 1/a_s is necessary for cost per correct decision to fall at r = 0.
    It fails in all eight published configurations, which is why the paper uses
    an explicit retention floor instead."""
    S = pd.read_csv(RESULTS / "summary_full.csv")
    for _, row in S.iterrows():
        assert not interior_minimum_possible(row.cost_ratio_k, row.small_acc), \
            f"{row.dataset}/{row.small} unexpectedly admits an interior minimum"


def test_concavity_makes_the_condition_sufficient():
    """Under concave g, a non-negative derivative at the origin implies no
    interior minimum anywhere. Verified by direct search."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(4000):
        a_s = rng.uniform(0.15, 0.85)
        k = rng.uniform(1.01, 12.0)
        G = rng.uniform(0.01, 1 - a_s)
        b = rng.uniform(0.5, 25.0)
        g = lambda r: G * (1 - np.exp(-b * r)) / (1 - np.exp(-b))
        gp0 = G * b / (1 - np.exp(-b))
        if gp0 > 1 or k * a_s < gp0:      # enforce g'(0)<=1; skip F'(0)<0 cases
            continue
        r = np.linspace(0, 1, 300)
        F = (1 + k * r) / (a_s + g(r))
        assert F.min() >= F[0] - 1e-9, "interior minimum under concave g"
        checked += 1
    assert checked > 200, "too few cases exercised"


# --------------------------------------------------------------------------
# published results stay reproducible
# --------------------------------------------------------------------------
def test_published_tables_are_internally_consistent():
    S = pd.read_csv(RESULTS / "summary_full.csv")
    assert len(S) == 8
    assert (S.aucac_router > S.aucac_random).all(), "router must beat random"
    assert (S.aucac_oracle >= S.aucac_router).all(), "oracle must bound router"
    assert (S.large_acc > S.small_acc).all()
    assert S.crossfit99_retention.between(0.95, 1.0).all()
    assert S.save_at_95.between(0, 1).all()


def test_scaling_reanalysis_reproduces_published_values():
    S = pd.read_csv(RESULTS / "summary_full.csv")
    out = scaling_reanalysis(S, n_boot=5000)
    assert out["pooled_r"] == pytest.approx(0.956, abs=0.002)
    assert out["within_dataset_all_positive"] is True
    lo, hi = out["cluster_bootstrap_ci"]
    assert 0.80 < lo < 0.95 and hi > 0.95


def test_at_retention_never_returns_a_point_below_the_floor(toy):
    conf, s, l, cs, cl = toy
    f = sweep(conf, s, l, cs, cl)
    for target in (0.95, 0.99):
        r = at_retention(f, l.mean(), target)
        if not np.isnan(r["saving"]):
            hit = f[np.isclose(f.escalation_rate, r["escalation_rate"])]
            assert (hit.accuracy >= target * l.mean() - 1e-9).any()

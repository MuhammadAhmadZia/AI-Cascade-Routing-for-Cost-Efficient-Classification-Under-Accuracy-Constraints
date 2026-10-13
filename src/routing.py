"""Routing, baselines and metrics. Pure CPU, no model required.

Everything here reads the prediction tables written by `run_inference` and is
deterministic given `SEED`.

Cost model
----------
    C(E) = c_s * n + c_l * |E|

The small tier runs on every item, escalated ones included, so its cost is paid
in full. Charging only the prompt prefix and omitting the scored label
continuations would understate the cheap tier by roughly a quarter on a
77-label task, which is exactly the accounting error that inflates reported
savings.

The proxy is linear in processed tokens, so it omits the quadratic attention
term, key-value cache traffic, and everything that depends on hardware,
batching and kernel implementation. Both tiers run the same prompts through the
same procedure on the same accelerator, so those terms enter both sides, and the
analysis turns on the ratio rather than either absolute. `sensitivity_fixed_ratio`
varies that ratio directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (ECE_BINS, N_BOOTSTRAP, N_CROSSFIT_FOLDS, N_RANDOM_DRAWS,
                     N_THRESHOLD_STEPS, SEED)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error, equal-width binning.

    Reported alongside AUROC because the two come apart. A router needs the
    ordering of confidence, not its absolute level, since the threshold is
    fitted on data regardless.
    """
    edges = np.linspace(0, 1, bins + 1)
    total, n = 0.0, len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.sum():
            total += m.sum() / n * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


# --------------------------------------------------------------------------
# frontiers
# --------------------------------------------------------------------------
def sweep(conf, small_ok, large_ok, cost_s, cost_l, n_steps=N_THRESHOLD_STEPS):
    """Escalate item i iff confidence_i < theta; sweep theta over its quantiles."""
    rows = []
    qs = np.unique(np.quantile(conf, np.linspace(0, 1, n_steps)))
    for theta in np.concatenate([[-np.inf], qs, [np.inf]]):
        esc = conf < theta
        correct = np.where(esc, large_ok, small_ok)
        cost = cost_s.sum() + cost_l[esc].sum()
        rows.append({"theta": float(theta),
                     "escalation_rate": float(esc.mean()),
                     "accuracy": float(correct.mean()),
                     "cost": float(cost),
                     "cost_per_correct": float(cost / max(correct.sum(), 1))})
    return pd.DataFrame(rows)


def random_frontier(small_ok, large_ok, cost_s, cost_l,
                    draws=N_RANDOM_DRAWS, n_steps=41, seed=SEED):
    """Escalate a matched fraction without looking at confidence.

    This is the honest null. A router that fails to beat it is worthless, since
    any escalation at all buys some accuracy.
    """
    rng = np.random.default_rng(seed)
    n, rows = len(small_ok), []
    for r in np.linspace(0, 1, n_steps):
        accs, costs = [], []
        for _ in range(draws):
            esc = rng.random(n) < r
            accs.append(np.where(esc, large_ok, small_ok).mean())
            costs.append(cost_s.sum() + cost_l[esc].sum())
        rows.append({"escalation_rate": float(r),
                     "accuracy": float(np.mean(accs)),
                     "cost": float(np.mean(costs))})
    return pd.DataFrame(rows)


def oracle_frontier(small_ok, large_ok, cost_s, cost_l, n_steps=41):
    """Escalate in descending order of gain: wrong-then-right items go first.

    Unattainable by construction, since it uses both models' outcomes. It bounds
    what any escalation policy could reach and it rises above always-large
    accuracy wherever the small model is right and the large model wrong.
    """
    order = np.argsort(-(large_ok.astype(int) - small_ok.astype(int)), kind="stable")
    n, rows = len(small_ok), []
    for r in np.linspace(0, 1, n_steps):
        esc = np.zeros(n, dtype=bool)
        esc[order[:int(round(r * n))]] = True
        rows.append({"escalation_rate": float(r),
                     "accuracy": float(np.where(esc, large_ok, small_ok).mean()),
                     "cost": float(cost_s.sum() + cost_l[esc].sum())})
    return pd.DataFrame(rows)


def aucac(frontier: pd.DataFrame) -> float:
    """Normalised area under the accuracy-cost curve. Higher is better."""
    d = frontier.sort_values("cost")
    span = max(d.cost.max() - d.cost.min(), 1e-12)
    x = (d.cost - d.cost.min()) / span
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(d.accuracy, x))


# --------------------------------------------------------------------------
# operating points
# --------------------------------------------------------------------------
def at_retention(frontier: pd.DataFrame, large_acc: float, target: float) -> dict:
    """Cheapest point on the frontier that retains `target` of large accuracy."""
    ok = frontier[frontier.accuracy >= target * large_acc].sort_values("escalation_rate")
    if not len(ok):
        return {"escalation_rate": float("nan"), "cost_fraction": float("nan"),
                "saving": float("nan")}
    row = ok.iloc[0]
    frac = float(row.cost / frontier.cost.max())
    return {"escalation_rate": float(row.escalation_rate),
            "cost_fraction": frac, "saving": 1.0 - frac}


def crossfit(conf, small_ok, large_ok, cost_s, cost_l,
             target=0.99, k=N_CROSSFIT_FOLDS, seed=SEED) -> dict:
    """Fit the threshold on k-1 folds, evaluate on the held-out fold.

    The full-sample sweep describes the attainable frontier. This asks the
    different question of whether a threshold chosen without sight of an item
    still holds on that item. No reported saving uses held-out outcomes to pick
    its own threshold.
    """
    folds = np.array_split(np.random.default_rng(seed).permutation(len(conf)), k)
    accs, escs, thetas = [], [], []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        f = sweep(conf[train], small_ok[train], large_ok[train],
                  cost_s[train], cost_l[train])
        ok = f[f.accuracy >= target * large_ok[train].mean()].sort_values("escalation_rate")
        theta = float(ok.theta.iloc[0]) if len(ok) else np.inf
        esc = conf[test] < theta
        accs.append(np.where(esc, large_ok[test], small_ok[test]).mean())
        escs.append(esc.mean())
        thetas.append(theta)
    return {"target": target,
            "theta_mean": float(np.mean(thetas)),
            "escalation_mean": float(np.mean(escs)),
            "escalation_sd": float(np.std(escs)),
            "accuracy_mean": float(np.mean(accs)),
            "retention_mean": float(np.mean(accs) / large_ok.mean())}


# --------------------------------------------------------------------------
# inference on differences
# --------------------------------------------------------------------------
def bootstrap_accuracy_ci(ok: np.ndarray, n=N_BOOTSTRAP, seed=SEED) -> tuple:
    rng = np.random.default_rng(seed)
    n_items = len(ok)
    draws = [ok[rng.integers(0, n_items, n_items)].mean() for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mcnemar(small_ok: np.ndarray, large_ok: np.ndarray) -> dict:
    """Paired test on the discordant cells, with continuity correction.

    The two tiers answer the same items, so an unpaired comparison would throw
    away the pairing and badly understate the available power.
    """
    b = int(((small_ok == 1) & (large_ok == 0)).sum())   # small right, large wrong
    c = int(((small_ok == 0) & (large_ok == 1)).sum())   # small wrong, large right
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return {"small_only": b, "large_only": c, "chi2": float(chi2)}


def error_structure(small_ok: np.ndarray, large_ok: np.ndarray) -> dict:
    """Partition the workload by which tiers get each item right."""
    n = len(small_ok)
    both_wrong = int(((small_ok == 0) & (large_ok == 0)).sum())
    return {
        "n": n,
        "escalation_can_help": int(((small_ok == 0) & (large_ok == 1)).sum()),
        "escalation_would_harm": int(((small_ok == 1) & (large_ok == 0)).sum()),
        "both_wrong": both_wrong,
        "both_wrong_share": both_wrong / n,
    }


# --------------------------------------------------------------------------
# when routing can pay (Section 3.7)
# --------------------------------------------------------------------------
def interior_minimum_possible(k: float, small_acc: float) -> bool:
    """Necessary condition for cost-per-correct-decision to fall at r = 0.

    With F(r) = c_s n (1 + k r) / (a_s + g(r)), the derivative at the origin is
    negative only when k * a_s < g'(0). Each escalated item can flip at most one
    answer, so g'(0) <= 1, giving the necessary condition k < 1 / a_s.

    Necessary, not sufficient: the sharper requirement is k * a_s < g'(0), and
    g'(0) can sit well below 1. Going from a non-negative derivative at the
    origin to the absence of any interior minimum needs g concave, which is what
    a sensible escalation order produces because the items with most to gain go
    first. Under concavity the sign of the derivative is non-decreasing in r, so
    a non-negative value at the origin carries across the whole range.
    """
    return k < 1.0 / small_acc


def sensitivity_fixed_ratio(conf, small_ok, large_ok, prefix_s, label_s,
                            cost_l, params_small, params_fixed, target=0.95) -> dict:
    """Re-price a cascade at a different small-tier parameter count.

    Two things change between the cascades: the small model gets better, and the
    cost ratio falls. Pricing the larger small model as though it cost the same
    per token as the smaller one isolates the accuracy effect.
    """
    out = {}
    for name, params in (("true", params_small), ("fixed", params_fixed)):
        cost_s = (params * (prefix_s + label_s)).astype(float)
        f = sweep(conf, small_ok, large_ok, cost_s, cost_l)
        out[name] = at_retention(f, large_ok.mean(), target)["saving"]
    return out

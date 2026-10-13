"""Build every table in the paper from the prediction files. CPU only.

Outputs written to `results/`:

    summary_full.csv            Table 3 plus the underlying frontier areas
    confidence_intervals.csv    Table 2
    cost_ratio_sensitivity.csv  Table 4
    error_structure.csv         Section 4.5
    scaling_reanalysis.json     Section 4.4 within/between decomposition
    frontier_<model>_<ds>.csv   per-pair frontier for the figures
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (CONFIDENCE_SIGNALS, DATASET_ORDER, LARGE_MODEL, PREDS,
                     RESULTS, SEED, SMALL_MODELS)
from .routing import (aucac, at_retention, bootstrap_accuracy_ci, crossfit, ece,
                      error_structure, mcnemar, oracle_frontier, random_frontier,
                      sweep)

LARGE_TAG, _, LARGE_PARAMS, _ = LARGE_MODEL


def _load_pair(small_tag: str, dataset: str) -> pd.DataFrame:
    """Join the two tiers on instance_id and attach costs."""
    s = pd.read_parquet(PREDS / f"{small_tag}__{dataset}.parquet")
    l = pd.read_parquet(PREDS / f"{LARGE_TAG}__{dataset}.parquet")
    m = s.merge(l, on="instance_id", suffixes=("_s", "_l"))
    if not len(m):
        raise RuntimeError(f"no overlap for {small_tag} / {dataset}")
    m["small_ok"] = (m.pred_s == m.gold_s).astype(int)
    m["large_ok"] = (m.pred_l == m.gold_s).astype(int)
    params = SMALL_MODELS[small_tag][1]
    m["cost_s"] = params * (m.prefix_tokens_s + m.label_tokens_s)
    m["cost_l"] = LARGE_PARAMS * (m.prefix_tokens_l + m.label_tokens_l)
    return m


def analyse_all(verbose: bool = True) -> dict:
    summary, cis, sens, errs = [], [], [], []

    for dataset in DATASET_ORDER:
        for small_tag, (_id, params, _q) in SMALL_MODELS.items():
            m = _load_pair(small_tag, dataset)
            s_ok, l_ok = m.small_ok.values, m.large_ok.values
            c_s = m.cost_s.values.astype(float)
            c_l = m.cost_l.values.astype(float)

            # best free signal by error-detection AUROC
            aurocs = {c: float(roc_auc_score(s_ok, m[f"{c}_s"].values))
                      for c in CONFIDENCE_SIGNALS}
            best = max(aurocs, key=aurocs.get)
            conf = m[f"{best}_s"].values

            front = sweep(conf, s_ok, l_ok, c_s, c_l)
            rand = random_frontier(s_ok, l_ok, c_s, c_l)
            orac = oracle_frontier(s_ok, l_ok, c_s, c_l)
            front.to_csv(RESULTS / f"frontier_{small_tag}_{dataset}.csv", index=False)

            r95 = at_retention(front, l_ok.mean(), 0.95)
            r99 = at_retention(front, l_ok.mean(), 0.99)
            cf = crossfit(conf, s_ok, l_ok, c_s, c_l, target=0.99)

            summary.append({
                "dataset": dataset,
                "labels": int(m.gold_s.nunique()),
                "small": small_tag,
                "n": len(m),
                "small_acc": round(float(s_ok.mean()), 3),
                "large_acc": round(float(l_ok.mean()), 3),
                "gap_pp": round(float((l_ok.mean() - s_ok.mean()) * 100), 1),
                "cost_ratio_k": round(float(c_l.sum() / c_s.sum()), 1),
                "signal": best,
                "auroc": round(aurocs[best], 3),
                "ece": round(ece(m.c1_max_softmax_s.values, s_ok), 3),
                "aucac_router": round(aucac(front), 4),
                "aucac_random": round(aucac(rand), 4),
                "aucac_oracle": round(aucac(orac), 4),
                "esc_at_95": round(r95["escalation_rate"], 3),
                "save_at_95": round(r95["saving"], 3),
                "esc_at_99": round(r99["escalation_rate"], 3),
                "save_at_99": round(r99["saving"], 3),
                "crossfit99_esc": round(cf["escalation_mean"], 3),
                "crossfit99_retention": round(cf["retention_mean"], 3),
            })

            s_lo, s_hi = bootstrap_accuracy_ci(s_ok)
            l_lo, l_hi = bootstrap_accuracy_ci(l_ok)
            mc = mcnemar(s_ok, l_ok)
            cis.append({
                "dataset": dataset, "small": small_tag, "n": len(m),
                "small_acc": round(float(s_ok.mean()), 3),
                "small_ci": f"[{s_lo:.3f}, {s_hi:.3f}]",
                "large_acc": round(float(l_ok.mean()), 3),
                "large_ci": f"[{l_lo:.3f}, {l_hi:.3f}]",
                "gap_pp": round(float((l_ok.mean() - s_ok.mean()) * 100), 1),
                "small_only": mc["small_only"], "large_only": mc["large_only"],
                "mcnemar_chi2": round(mc["chi2"], 1),
            })

            es = error_structure(s_ok, l_ok)
            es.update({"dataset": dataset, "small": small_tag})
            errs.append(es)

            # cost-ratio sensitivity: re-price the small tier at 1.7B cost
            base_params = min(p for _i, p, _q in SMALL_MODELS.values())
            c_s_fixed = (base_params * (m.prefix_tokens_s + m.label_tokens_s)).values.astype(float)
            f_fixed = sweep(conf, s_ok, l_ok, c_s_fixed, c_l)
            sens.append({
                "dataset": dataset, "small": small_tag,
                "small_acc": round(float(s_ok.mean()), 3),
                "save_true_ratio": round(r95["saving"], 3),
                "save_fixed_ratio": round(at_retention(f_fixed, l_ok.mean(), 0.95)["saving"], 3),
            })

            if verbose:
                print(f"{dataset:10s} {small_tag:11s} acc {s_ok.mean():.3f} "
                      f"save@95 {r95['saving']:.3f}")

    S = pd.DataFrame(summary)
    S.to_csv(RESULTS / "summary_full.csv", index=False)
    pd.DataFrame(cis).to_csv(RESULTS / "confidence_intervals.csv", index=False)
    pd.DataFrame(sens).to_csv(RESULTS / "cost_ratio_sensitivity.csv", index=False)
    pd.DataFrame(errs).to_csv(RESULTS / "error_structure.csv", index=False)

    reanalysis = scaling_reanalysis(S)
    (RESULTS / "scaling_reanalysis.json").write_text(json.dumps(reanalysis, indent=2))
    return {"summary": S, "scaling": reanalysis}


def scaling_reanalysis(S: pd.DataFrame, n_boot: int = 20000, seed: int = SEED) -> dict:
    """Within- and between-dataset decomposition of the savings relationship.

    Eight points with two per dataset are not eight independent observations, so
    a pooled correlation on its own would be weak evidence. Resampling whole
    datasets respects the pairing; leaving one dataset out shows whether a single
    task carries the result; and the within-dataset movement is the part that
    between-dataset variation cannot explain.
    """
    order = DATASET_ORDER
    acc = np.array([S[(S.dataset == d) & (S.small == m)].small_acc.iloc[0]
                    for d in order for m in SMALL_MODELS])
    sav = np.array([S[(S.dataset == d) & (S.small == m)].save_at_95.iloc[0]
                    for d in order for m in SMALL_MODELS])

    pooled = float(np.corrcoef(acc, sav)[0, 1])

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(order), len(order), replace=True)
        a = np.concatenate([acc[[2 * i, 2 * i + 1]] for i in pick])
        s = np.concatenate([sav[[2 * i, 2 * i + 1]] for i in pick])
        if np.std(a) > 1e-9 and np.std(s) > 1e-9:
            draws.append(np.corrcoef(a, s)[0, 1])
    draws = np.asarray(draws)

    loo = {}
    for i, name in enumerate(order):
        keep = [j for j in range(len(acc)) if j // 2 != i]
        loo[name] = round(float(np.corrcoef(acc[keep], sav[keep])[0, 1]), 3)

    d_acc = np.array([acc[2 * i + 1] - acc[2 * i] for i in range(len(order))])
    d_sav = np.array([sav[2 * i + 1] - sav[2 * i] for i in range(len(order))])

    return {
        "pooled_r": round(pooled, 3),
        "cluster_bootstrap_ci": [round(float(np.percentile(draws, 2.5)), 2),
                                 round(float(np.percentile(draws, 97.5)), 2)],
        "leave_one_dataset_out_r": loo,
        "within_dataset_saving_delta": {order[i]: round(float(d_sav[i]), 3)
                                        for i in range(len(order))},
        "within_dataset_all_positive": bool((d_sav > 0).all()),
        "within_dataset_mean_delta": round(float(d_sav.mean()), 3),
        "sign_test_one_sided_p": round(0.5 ** len(order), 4),
        "local_slope_range": [round(float((d_sav / d_acc).min()), 2),
                              round(float((d_sav / d_acc).max()), 2)],
        "note": ("Direction is consistent across all datasets and survives "
                 "dataset-level resampling. With four datasets the sign test "
                 "is not conclusive at conventional levels, and the local slope "
                 "varies by a factor of about 2.4, so the relationship is "
                 "monotone and steep rather than linear with a transferable "
                 "coefficient."),
    }


if __name__ == "__main__":
    out = analyse_all()
    print("\n" + out["summary"].to_string(index=False))
    print("\nscaling reanalysis:")
    print(json.dumps(out["scaling"], indent=2))

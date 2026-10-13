"""Resumable inference driver.

One model is loaded once and run over every dataset. Predictions and the full
(N, K) score matrix are checkpointed to disk every `CHECKPOINT_EVERY` instances,
so an interrupted run resumes at the last checkpoint instead of restarting.

Storing the whole score matrix, not just the three scalar signals, is what makes
the analysis reproducible without a GPU. Any later question about scoring rules,
calibration or alternative confidence signals is answered from these files.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (CHECKPOINT_EVERY, DATASETS, DATASET_ORDER, LARGE_MODEL,
                     PREDS, SMALL_MODELS)
from .datasets import labels_of, load_frozen
from .scoring import (build_prefix, confidence_signals, load_model, score_labels,
                      truncate)


def run_combo(tag: str, model, tok, dataset: str, n_eval: int) -> pd.DataFrame:
    """Score one (model, dataset) pair, resuming from any existing checkpoint."""
    pred_path = PREDS / f"{tag}__{dataset}.parquet"
    score_path = PREDS / f"{tag}__{dataset}__scores.npz"

    df = load_frozen(dataset, n_eval)
    labels = labels_of(dataset)
    label_ids = [tok(l, add_special_tokens=False, return_tensors="pt").input_ids
                 for l in labels]
    max_tokens = DATASETS[dataset]["max_tokens"]

    done, rows, mat = set(), [], []
    if pred_path.exists():
        prev = pd.read_parquet(pred_path)
        prev = prev[prev.instance_id < len(df)]
        done, rows = set(prev.instance_id), prev.to_dict("records")
        if score_path.exists():
            mat = list(np.load(score_path)["norms"])[:len(rows)]
        if len(done) >= len(df):
            print(f"  {dataset}: complete ({len(done)})")
            return prev
        print(f"  {dataset}: resuming at {len(done)}/{len(df)}")

    t0, since = time.time(), 0
    for i, r in df.iterrows():
        if i in done:
            continue
        prefix = build_prefix(tok, truncate(tok, r.text, max_tokens), labels)
        p, norms, _sums, n_prefix, n_label = score_labels(
            model, tok, prefix, label_ids, model.device)
        assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-6, "invalid distribution"

        sig = confidence_signals(p)
        rows.append({
            "instance_id": int(i),
            "gold": r.label,
            "pred": labels[sig["pred_idx"]],
            "c1_max_softmax": sig["c1_max_softmax"],
            "c2_margin": sig["c2_margin"],
            "c3_norm_entropy": sig["c3_norm_entropy"],
            "prefix_tokens": n_prefix,
            "label_tokens": n_label,
        })
        mat.append(norms.astype(np.float32))
        since += 1

        if since % CHECKPOINT_EVERY == 0:
            _checkpoint(rows, mat, labels, pred_path, score_path)
            el = time.time() - t0
            print(f"    {len(rows)}/{len(df)}  {el/since:.1f}s/item  "
                  f"eta {el/since*(len(df)-len(rows))/60:.0f} min", flush=True)

    out = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
    _checkpoint(out.to_dict("records"), mat, labels, pred_path, score_path)
    acc = (out.pred == out.gold).mean()
    print(f"  {dataset}: accuracy {acc:.3f}  ({(time.time()-t0)/60:.0f} min)")
    return out


def _checkpoint(rows, mat, labels, pred_path: Path, score_path: Path) -> None:
    pd.DataFrame(rows).to_parquet(pred_path, index=False)
    np.savez_compressed(score_path,
                        norms=np.asarray(mat, dtype=np.float32),
                        labels=np.asarray(labels))


def run_model(tag: str, model_id: str, four_bit: bool, datasets: list[str]) -> None:
    print(f"\n=== {tag} ({model_id})")
    tok, model = load_model(model_id, four_bit)
    try:
        for ds in datasets:
            run_combo(tag, model, tok, ds, DATASETS[ds]["n_eval"])
    finally:
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"freed | allocated {torch.cuda.memory_allocated()/1e9:.2f} GB"
              if torch.cuda.is_available() else "freed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=list(SMALL_MODELS) + [LARGE_MODEL[0]],
                    help="model tags to run (default: all)")
    ap.add_argument("--datasets", nargs="+", default=DATASET_ORDER,
                    help="dataset names to run (default: all, cheapest first)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU detected. Inference needs a CUDA device.")
    print("GPU:", torch.cuda.get_device_name(0))

    registry = dict(SMALL_MODELS)
    registry[LARGE_MODEL[0]] = (LARGE_MODEL[1], LARGE_MODEL[2], LARGE_MODEL[3])

    for tag in args.models:
        if tag not in registry:
            raise SystemExit(f"unknown model tag {tag!r}; known: {list(registry)}")
        model_id, _params, four_bit = registry[tag]
        run_model(tag, model_id, four_bit, args.datasets)
    print("\nALL DONE")


if __name__ == "__main__":
    main()

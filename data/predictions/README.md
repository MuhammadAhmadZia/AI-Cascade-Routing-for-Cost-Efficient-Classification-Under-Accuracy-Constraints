# Prediction files

`src/run_inference.py` writes two files per (model, dataset) pair here:

    <model>__<dataset>.parquet          per-item prediction, gold label, three
                                        confidence signals, token counts
    <model>__<dataset>__scores.npz      the full (N, K) matrix of
                                        length-normalised label scores, plus the
                                        label list

Eight pairs in total: two small models times four datasets, plus the large model
on all four.

These are not in git because of their size. Regenerate them with
`scripts/01_run_small_models.sh` and `scripts/02_run_large_model.sh`, or drop
the released archive into this folder and run `scripts/03_analyse.sh`.

The score matrices are the reason the analysis needs no GPU. Any later question
about scoring rules, calibration or alternative confidence signals is answerable
from these files alone.

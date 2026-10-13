"""Central configuration. Every number that defines the experiment lives here."""
from pathlib import Path

# ---------------------------------------------------------------- paths
# Override with the CASCADE_ROOT environment variable (e.g. a Google Drive folder).
import os
ROOT = Path(os.environ.get("CASCADE_ROOT", Path(__file__).resolve().parent.parent))
FROZEN = ROOT / "data" / "frozen"
PREDS = ROOT / "data" / "predictions"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
for _p in (FROZEN, PREDS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 0

# ---------------------------------------------------------------- models
# name -> (HuggingFace id, parameters in billions, load in 4-bit)
SMALL_MODELS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", 1.7, False),
    "qwen3-4b":   ("Qwen/Qwen3-4B",   4.0, False),
}
LARGE_MODEL = ("qwen3-14b", "Qwen/Qwen3-14B", 14.0, True)

# Cost ratio k = c_large / c_small, implied by the parameter counts above.
#   qwen3-1.7b cascade: 14.0 / 1.7 = 8.2
#   qwen3-4b   cascade: 14.0 / 4.0 = 3.5

# ---------------------------------------------------------------- datasets
# Per-dataset truncation is deliberate; see DATASETS_FROZEN.md for the rationale.
DATASETS = {
    "cfpb":      {"max_tokens": 512, "n_frozen": 2000, "n_eval": 1000},
    "banking77": {"max_tokens": 256, "n_frozen": 1925, "n_eval": 1000},
    "ledgar":    {"max_tokens": 512, "n_frozen": 2000, "n_eval": 500},
    "clinc150":  {"max_tokens": 256, "n_frozen": 1950, "n_eval": 500},
}
DATASET_ORDER = ["cfpb", "banking77", "ledgar", "clinc150"]

# ---------------------------------------------------------------- prompt
# Identical across both tiers. Changing it invalidates the comparison.
PROMPT = """Classify the input into exactly one category.

Categories:
{labels}

Input: {text}

Answer with the category name only."""

# ---------------------------------------------------------------- analysis
CONFIDENCE_SIGNALS = ["c1_max_softmax", "c2_margin", "c3_norm_entropy"]
RETENTION_TARGETS = [0.95, 0.99]
N_THRESHOLD_STEPS = 200      # quantiles swept over the confidence distribution
N_RANDOM_DRAWS = 25          # draws averaged for the random-escalation baseline
N_BOOTSTRAP = 2000           # resamples for accuracy intervals
N_CROSSFIT_FOLDS = 5
ECE_BINS = 15
CHECKPOINT_EVERY = 50        # instances between parquet writes during inference

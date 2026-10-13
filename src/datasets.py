"""Dataset loading, stratified sampling and freezing.

Each loader returns a DataFrame with exactly two columns, `text` and `label`,
with string labels. Integer label codes are never carried forward: a mapping
that silently loses its names would score the strings "0", "1", "2" as if they
were category names and produce a complete run of meaningless numbers.

Run `scripts/00_freeze_datasets.py` once. After that every stage reads the
frozen parquet files and the loaders are not touched again.
"""
from __future__ import annotations

import hashlib
import io
import json
import datetime
import numpy as np
import pandas as pd

from .config import DATASETS, FROZEN, SEED

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}


# --------------------------------------------------------------------------
# individual loaders
# --------------------------------------------------------------------------
def load_banking77() -> pd.DataFrame:
    """Canonical Banking77 test split from PolyAI's GitHub repository.

    The HuggingFace mirror ships a loading script that `datasets` 4.x refuses to
    execute, and that same script blocked the hub's parquet auto-conversion, so
    `refs/convert/parquet` does not exist for it either. These CSVs carry string
    labels directly, so there is no integer mapping to get wrong.
    """
    url = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
           "master/banking_data/test.csv")
    d = pd.read_csv(url)
    assert list(d.columns) == ["text", "category"], d.columns.tolist()
    assert len(d) == 3080 and d.category.nunique() == 77, (len(d), d.category.nunique())
    return pd.DataFrame({"text": d.text, "label": d.category.map(normalise_label)})


def load_clinc150() -> pd.DataFrame:
    """CLINC150 'plus' test split, out-of-scope class removed.

    The out-of-scope label carries no semantic content for a closed-set scorer,
    so it is dropped and the task stays closed over 150 intents. The `label_text`
    column supplies names per row, which removes the integer-mapping risk.
    """
    from datasets import load_dataset
    ds = load_dataset("DeepPavlov/clinc_oos", "plus", split="test").to_pandas()
    ds = ds[ds.label_text != "oos"]
    assert ds.label_text.nunique() == 150, ds.label_text.nunique()
    return pd.DataFrame({"text": ds.text, "label": ds.label_text.map(normalise_label)})


def load_ledgar() -> pd.DataFrame:
    """LEDGAR contract-provision test split, via the LexGLUE distribution."""
    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="test")
    names = ds.features["label"].names
    assert len(names) == 100, len(names)
    d = ds.to_pandas()
    return pd.DataFrame({"text": d.text, "label": d.label.map(lambda i: names[i])})


def load_cfpb(date_min: str = "2024-01-01", date_max: str = "2024-02-01") -> pd.DataFrame:
    """CFPB consumer complaints with narratives, one month, via the public API.

    A single month keeps the product taxonomy internally consistent. The CFPB
    revised its product categories over time and mixing eras produces near
    duplicate labels for the same concept, which is unfixable after sampling.

    Two API quirks are handled here: `format` only accepts `csv` (omitting it
    yields JSON), and the CSV export uses title-case column names while the JSON
    search endpoint uses snake_case for the same fields.
    """
    import requests
    base = ("https://www.consumerfinance.gov/data-research/consumer-complaints/"
            "search/api/v1/")
    url = (f"{base}?format=csv&no_aggs=true&has_narrative=true"
           f"&date_received_min={date_min}&date_received_max={date_max}")
    r = requests.get(url, headers=HTTP_HEADERS, timeout=600)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    prod = next(c for c in d.columns if c.strip().lower() == "product")
    narr = next(c for c in d.columns if "narrative" in c.lower())
    d = d[[narr, prod]].dropna()
    d.columns = ["text", "label"]
    # One product has too few narratives in a single month to fill an equal quota.
    thin = d.label.value_counts()
    d = d[d.label.isin(thin[thin >= 200].index)]
    return d.reset_index(drop=True)


LOADERS = {
    "banking77": load_banking77,
    "clinc150": load_clinc150,
    "ledgar": load_ledgar,
    "cfpb": load_cfpb,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def normalise_label(s: str) -> str:
    """Underscored identifiers become natural language. LEDGAR labels are
    already title-cased prose and pass through a no-op."""
    return s.replace("_", " ").lower() if "_" in s else s


def stratified_capped(df: pd.DataFrame, n_total: int, seed: int = SEED):
    """Up to n_total/K per class, then top up from the remainder.

    LEDGAR's test split is naturally imbalanced and six provision types hold
    fewer instances than an equal quota needs. Capping rather than requiring
    keeps all 100 classes present, which matters because the label-space size is
    one of the axes the study varies. Classes that fall short are reported.
    """
    classes = sorted(df.label.unique())
    quota = n_total // len(classes)
    parts = [g.sample(min(len(g), quota), random_state=seed)
             for _, g in df.groupby("label")]
    out = pd.concat(parts)
    counts = df.label.value_counts()
    short = {c: int(n) for c, n in counts.items() if n < quota}
    if len(out) < n_total:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(n_total - len(out), len(rest)),
                                          random_state=seed)])
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, short


def sha256_of(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# --------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------
def freeze_all(verbose: bool = True) -> dict:
    """Build every frozen sample and write `manifest.json` with checksums.

    Selecting datasets after seeing results is the easiest way to make an honest
    study unpublishable, and it is invisible without a pre-commitment. The
    manifest is that commitment: later stages verify their inputs against it and
    refuse to run if a file has changed.
    """
    manifest = {
        "seed": SEED,
        "frozen_on": datetime.date.today().isoformat(),
        "datasets": {},
    }
    for name in DATASETS:
        df = LOADERS[name]()
        n = DATASETS[name]["n_frozen"]
        sample, short = stratified_capped(df, n)
        path = FROZEN / f"{name}.parquet"
        sample.to_parquet(path, index=False)
        entry = {
            "n": len(sample),
            "n_labels": int(sample.label.nunique()),
            "max_text_tokens": DATASETS[name]["max_tokens"],
            "n_eval": DATASETS[name]["n_eval"],
            "sampling": ("capped per class, topped up from the remainder"
                         if short else "equal quota per class"),
            "sha256": sha256_of(path),
        }
        if short:
            entry["short_classes"] = short
        manifest["datasets"][name] = entry
        if verbose:
            print(f"{name:10s} n={len(sample):5d} labels={entry['n_labels']:4d} "
                  f"sha={entry['sha256']}"
                  + (f"  short: {len(short)} classes" if short else ""))

    (FROZEN / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_frozen(name: str, n: int | None = None, verify: bool = True) -> pd.DataFrame:
    """Read a frozen sample, checking it against the manifest checksum."""
    path = FROZEN / f"{name}.parquet"
    if verify:
        manifest = json.loads((FROZEN / "manifest.json").read_text())
        expected = manifest["datasets"][name]["sha256"]
        actual = sha256_of(path)
        if actual != expected:
            raise RuntimeError(
                f"{name}.parquet changed since freezing "
                f"(manifest {expected}, file {actual}). Re-freezing would "
                f"invalidate every result already computed against it.")
    df = pd.read_parquet(path)
    assert {"text", "label"} <= set(df.columns), df.columns.tolist()
    return df.head(n).reset_index(drop=True) if n else df


def labels_of(name: str) -> list[str]:
    """Full label set for a dataset, taken from the frozen sample."""
    return sorted(load_frozen(name).label.unique())

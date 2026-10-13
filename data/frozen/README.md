# Frozen evaluation samples

`scripts/00_freeze_datasets.py` writes four parquet files here plus
`manifest.json`, which records row counts, label counts, truncation limits,
sampling rules and a sha256 prefix per file.

`load_frozen()` recomputes the checksum on every read and refuses to proceed if
it has changed. Re-freezing would silently invalidate every result already
computed against the old sample.

See `DATASETS_FROZEN.md` in the repository root for provenance and for the
reasoning behind each sampling and truncation decision.

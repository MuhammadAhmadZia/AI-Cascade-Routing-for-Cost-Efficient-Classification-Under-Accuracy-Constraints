#!/usr/bin/env python3
"""Build the frozen evaluation samples and write the manifest. Run once.

Freezing before any model runs is what makes the study auditable: the manifest
records the exact samples, their label spaces and their checksums, and later
stages refuse to run against a file that has changed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets import freeze_all

if __name__ == "__main__":
    m = freeze_all()
    print(f"\nfrozen on {m['frozen_on']}, seed {m['seed']}")
    print("manifest written to data/frozen/manifest.json")

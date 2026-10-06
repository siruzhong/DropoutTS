#!/usr/bin/env python
"""
Resplit the already-prepared ExchangeRate dataset to support long horizons (e.g., pred_len=720).

Why:
  BasicTSForecastingDataset.__len__ = T - input_len - output_len + 1
  With the default 0.7/0.1/0.2 split, ExchangeRate val_T=758 < 96+720=816, which crashes validation.

What this script does:
  - Reconstruct full series by concatenating existing train/val/test in order
  - Resplit into 0.6/0.2/0.2 (common in LTSF codebases and ensures val/test >= 816 here)
  - Back up the original split files before overwriting
  - Update datasets/ExchangeRate/meta.json train_val_test_ratio accordingly
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DS_DIR = ROOT / "datasets" / "ExchangeRate"

SPLIT = (0.6, 0.2, 0.2)
MIN_REQUIRED = 96 + 720  # for our rebuttal setting; adjust if needed


def _load(split: str):
    x = np.load(DS_DIR / f"{split}_data.npy")
    ts = np.load(DS_DIR / f"{split}_timestamps.npy")
    return x, ts


def main():
    DS_DIR.mkdir(parents=True, exist_ok=True)

    train_x, train_ts = _load("train")
    val_x, val_ts = _load("val")
    test_x, test_ts = _load("test")

    full_x = np.concatenate([train_x, val_x, test_x], axis=0)
    full_ts = np.concatenate([train_ts, val_ts, test_ts], axis=0)

    T = full_x.shape[0]
    t1 = int(T * SPLIT[0])
    t2 = int(T * (SPLIT[0] + SPLIT[1]))

    new = {
        "train": (full_x[:t1], full_ts[:t1]),
        "val": (full_x[t1:t2], full_ts[t1:t2]),
        "test": (full_x[t2:], full_ts[t2:]),
    }

    print(f"Full series: T={T}, C={full_x.shape[1]}")
    for k, (x, ts) in new.items():
        print(f"New split {k:>5s}: data={x.shape}, ts={ts.shape}")

    # Sanity check for long-horizon feasibility
    bad = [k for k, (x, _) in new.items() if x.shape[0] < MIN_REQUIRED]
    if bad:
        raise RuntimeError(
            f"Split(s) too short for input_len+pred_len={MIN_REQUIRED}: {bad}. "
            f"Consider a different ratio or dropping pred_len=720."
        )

    # Backup originals
    backup_dir = DS_DIR.parent / f"ExchangeRate_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fn in [
        "train_data.npy", "train_timestamps.npy",
        "val_data.npy", "val_timestamps.npy",
        "test_data.npy", "test_timestamps.npy",
        "meta.json",
    ]:
        src = DS_DIR / fn
        if src.exists():
            shutil.copy2(src, backup_dir / fn)
    print(f"Backed up original files to: {backup_dir}")

    # Write new split (overwrite)
    for split, (x, ts) in new.items():
        np.save(DS_DIR / f"{split}_data.npy", x)
        np.save(DS_DIR / f"{split}_timestamps.npy", ts)

    # Update meta.json ratio
    meta_path = DS_DIR / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    meta.setdefault("regular_settings", {})
    meta["regular_settings"]["train_val_test_ratio"] = list(SPLIT)
    meta["num_time_steps"] = int(T)
    meta["shape"] = [int(T), int(full_x.shape[1])]
    meta["timestamps_shape"] = [int(T), int(full_ts.shape[1])]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Updated meta.json ratio to {SPLIT}")

    print("Done.")


if __name__ == "__main__":
    main()


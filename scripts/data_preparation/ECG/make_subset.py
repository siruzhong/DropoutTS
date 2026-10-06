#!/usr/bin/env python
"""
Create a smaller ECG dataset subset for fast rebuttal experiments.

This script takes the existing processed ECG dataset in `datasets/ECG/`
and creates a contiguous prefix subset (preserves temporal continuity).

Output: `datasets/ECG_10k/` by default.
"""

import os
import json
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))


def save_desc(out_dir: str, total_steps: int, num_features: int, fs_hz: int = 36):
    desc = {
        "name": os.path.basename(out_dir.rstrip("/")),
        "domain": "ECG (PhysioNet MIT-BIH Arrhythmia; downsampled)",
        "num_time_steps": int(total_steps),
        "num_nodes": int(num_features),
        "num_features": 1,
        "has_graph": False,
        "sampling_rate_hz": int(fs_hz),
    }
    with open(os.path.join(out_dir, "desc.json"), "w") as f:
        json.dump(desc, f, indent=2)


def main(total_steps: int = 10_000, out_name: str = "ECG_10k"):
    src_dir = os.path.join(ROOT_DIR, "datasets", "ECG")
    out_dir = os.path.join(ROOT_DIR, "datasets", out_name)
    os.makedirs(out_dir, exist_ok=True)

    splits = [("train", "train"), ("val", "val"), ("test", "test")]
    data_all = []
    ts_all = []
    for split, prefix in splits:
        data_p = os.path.join(src_dir, f"{prefix}_data.npy")
        ts_p = os.path.join(src_dir, f"{prefix}_ts.npy")
        if not (os.path.exists(data_p) and os.path.exists(ts_p)):
            raise FileNotFoundError(f"Missing {data_p} or {ts_p}. Run ECG preprocessing first.")
        data_all.append(np.load(data_p))
        ts_all.append(np.load(ts_p))

    data = np.concatenate(data_all, axis=0)
    ts = np.concatenate(ts_all, axis=0)

    if total_steps <= 0 or total_steps > data.shape[0]:
        raise ValueError(f"total_steps must be in [1, {data.shape[0]}], got {total_steps}")

    data = data[:total_steps]
    ts = ts[:total_steps]

    # Keep the original split ratios: 0.7 / 0.1 / 0.2
    t1 = int(total_steps * 0.7)
    t2 = int(total_steps * 0.8)
    parts = {
        "train": (data[:t1], ts[:t1]),
        "val": (data[t1:t2], ts[t1:t2]),
        "test": (data[t2:], ts[t2:]),
    }

    for split, (d, t) in parts.items():
        np.save(os.path.join(out_dir, f"{split}_data.npy"), d)
        # Keep both names for compatibility with older scripts.
        np.save(os.path.join(out_dir, f"{split}_ts.npy"), t)
        np.save(os.path.join(out_dir, f"{split}_timestamps.npy"), t)

    # Copy normalization stats if present (helps keep configs comparable)
    for name in ["normalize_mean.npy", "normalize_std.npy"]:
        p = os.path.join(src_dir, name)
        if os.path.exists(p):
            np.save(os.path.join(out_dir, name), np.load(p))

    save_desc(out_dir, total_steps=total_steps, num_features=data.shape[1], fs_hz=36)

    print(f"Saved subset dataset to: {out_dir}")
    print(f"  total_steps={total_steps}, num_features={data.shape[1]}")
    print(f"  train={parts['train'][0].shape}, val={parts['val'][0].shape}, test={parts['test'][0].shape}")


if __name__ == "__main__":
    main()


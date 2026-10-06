"""ECG Dataset Preparation (MIT-BIH Arrhythmia Database).

Downloads 6 records from the MIT-BIH Arrhythmia Database via PhysioNet,
extracts both ECG leads (2 channels per record → 12 channels total),
downsamples from 360 Hz to 36 Hz (factor 10), and saves as BasicTS-compatible
.npy files under datasets/ECG/.

Dataset statistics after processing:
  - Records   : 100, 101, 102, 103, 104, 105
  - Channels  : 12 (2 leads × 6 records)
  - Time steps: ~65,000 (≈ 30 min of ECG @ 36 Hz)
  - Split     : 70 / 10 / 20

Usage:
    python scripts/data_preparation/ECG/generate_training_data.py
"""

import os
import sys
import numpy as np
import wfdb
from scipy.signal import decimate

# ── Configuration ────────────────────────────────────────────────────────────
RECORDS      = ['100', '101', '102', '103', '104', '105']   # MIT-BIH records
PHYSIONET_DB = 'mitdb'          # PhysioNet database identifier
DOWNSAMPLE   = 10               # 360 Hz → 36 Hz
SPLIT        = (0.7, 0.1, 0.2)  # train / val / test

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
RAW_DIR     = os.path.join(ROOT_DIR, 'datasets', 'raw_data', 'ECG')
OUTPUT_DIR  = os.path.join(ROOT_DIR, 'datasets', 'ECG')
os.makedirs(RAW_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def download_record(record_id: str) -> np.ndarray:
    """Download one MIT-BIH record and return its signal as (T, 2) ndarray."""
    local_path = os.path.join(RAW_DIR, record_id)
    if os.path.exists(local_path + '.hea'):
        print(f"  [{record_id}] already cached, loading from disk ...")
        rec = wfdb.rdrecord(local_path)
    else:
        print(f"  [{record_id}] downloading from PhysioNet ...")
        rec = wfdb.rdrecord(record_id, pn_dir=PHYSIONET_DB)
        wfdb.wrsamp(
            record_id,
            fs=rec.fs,
            units=rec.units,
            sig_name=rec.sig_name,
            p_signal=rec.p_signal,
            write_dir=RAW_DIR,
        )
    signal = rec.p_signal.astype(np.float32)       # (T, 2)
    # Replace NaN (missing samples) with linear interpolation
    for ch in range(signal.shape[1]):
        nans = np.isnan(signal[:, ch])
        if nans.any():
            idx = np.arange(len(signal))
            signal[nans, ch] = np.interp(idx[nans], idx[~nans], signal[~nans, ch])
    return signal


def downsample_signal(signal: np.ndarray, factor: int) -> np.ndarray:
    """Anti-alias decimate along time axis, column by column."""
    cols = []
    for ch in range(signal.shape[1]):
        cols.append(decimate(signal[:, ch], factor, zero_phase=True))
    return np.stack(cols, axis=1).astype(np.float32)


def main():
    print("=" * 60)
    print("ECG Dataset Preparation (MIT-BIH Arrhythmia Database)")
    print("=" * 60)

    all_signals = []
    for rid in RECORDS:
        raw  = download_record(rid)                     # (T_raw, 2)
        down = downsample_signal(raw, DOWNSAMPLE)       # (T_ds, 2)
        print(f"  [{rid}] raw {raw.shape} → downsampled {down.shape}")
        all_signals.append(down)

    # Truncate all to minimum length then stack as channels
    min_len = min(s.shape[0] for s in all_signals)
    all_signals = [s[:min_len] for s in all_signals]
    data = np.concatenate(all_signals, axis=1)          # (T, 12)
    print(f"\nCombined shape: {data.shape}  (T={data.shape[0]}, C={data.shape[1]})")

    # Global z-score normalisation (per channel)
    mu  = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0,  keepdims=True) + 1e-6
    data = (data - mu) / std

    # Train / val / test split
    T = data.shape[0]
    t1 = int(T * SPLIT[0])
    t2 = int(T * (SPLIT[0] + SPLIT[1]))
    train, val, test = data[:t1], data[t1:t2], data[t2:]
    print(f"Split  → train {train.shape}, val {val.shape}, test {test.shape}")

    # Dummy timestamp arrays (required by BasicTS but unused for ECG).
    # Must be in [0, 1) to avoid out-of-bounds in TimestampEmbedding.
    def make_ts(length):
        ts = np.zeros((length, 4), dtype=np.float32)
        ts[:, 0] = (np.arange(length) % 86400) / 86400.0  # normalized to [0,1)
        return ts

    splits = {'train': train, 'val': val, 'test': test}
    for name, arr in splits.items():
        np.save(os.path.join(OUTPUT_DIR, f'{name}_data.npy'), arr)
        np.save(os.path.join(OUTPUT_DIR, f'{name}_ts.npy'),   make_ts(arr.shape[0]))
        print(f"Saved  {name}_data.npy  {arr.shape}")

    # Save normalisation stats for reference
    np.save(os.path.join(OUTPUT_DIR, 'normalize_mean.npy'), mu)
    np.save(os.path.join(OUTPUT_DIR, 'normalize_std.npy'),  std)

    # Save full dataset as CSV for visualization compatibility
    import pandas as pd
    col_names = [f'lead_{i}' for i in range(data.shape[1])]
    fs_ds = 360 // DOWNSAMPLE
    time_sec = np.arange(T) / fs_ds
    df_full = pd.DataFrame(data, columns=col_names)
    df_full.insert(0, 'time_sec', time_sec)
    csv_raw_dir = os.path.join(ROOT_DIR, 'datasets', 'raw_data', 'ECG')
    os.makedirs(csv_raw_dir, exist_ok=True)
    csv_path = os.path.join(csv_raw_dir, 'ECG.csv')
    df_full.to_csv(csv_path, index=False)
    print(f"Saved  ECG.csv  {df_full.shape}  →  {csv_path}")

    print("\nDone! ECG dataset ready at:", OUTPUT_DIR)
    print(f"  num_features = {data.shape[1]}")
    print(f"  total_steps  = {T}")
    print(f"  fs_after_ds  = {360 // DOWNSAMPLE} Hz")


if __name__ == '__main__':
    main()

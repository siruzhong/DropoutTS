import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tkde_style import PALETTE, apply_tkde_style, save_figure, style_axis


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "dropoutts_visualization_source.npz"
# Selected from the full paired test set using moderate baseline errors and
# improvements spread across the test distribution, rather than largest gains.
DEFAULT_INDICES = (3768, 517, 4145, 5331)
EXPORT_FORMATS = (".pdf",)

# Preserve the original figure's method colors. They are clear, familiar, and
# distinct in both the time and frequency portions of each forecast.
COLORS = {
    "input": "#7F8C8D",
    "ground_truth": "#27AE60",
    "baseline": "#D62728",
    "dropoutts": "#1F77B4",
    "boundary": "#34495E",
}


def set_academic_style():
    """Use the shared TKDE typography without changing the original palette."""
    apply_tkde_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
    })


def load_test_results(result_dir):
    result_dir = Path(result_dir)
    test_results_dir = result_dir / "test_results"
    cfg_file = result_dir / "cfg.json"
    input_len, output_len, n_channels = 96, 720, 1

    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as handle:
            model_cfg = json.load(handle).get("model_config", {})
        input_len = model_cfg.get("input_len", input_len)
        output_len = model_cfg.get("output_len", output_len)
        n_channels = model_cfg.get("num_features", n_channels)

    def load_npy(name):
        path = test_results_dir / f"{name}.npy"
        length = input_len if name == "inputs" else output_len
        try:
            data = np.load(path)
            if data.ndim == 2:
                data = data.reshape(-1, length, n_channels)
            return data
        except ValueError:
            data = np.memmap(path, mode="r", dtype="float32")
            return data.reshape(-1, length, n_channels)

    return load_npy("inputs"), load_npy("prediction"), load_npy("targets")


def load_checkpoint_samples(before_dir, after_dir, feat_idx, indices):
    """Load the original four samples from matched 96-to-720 checkpoints."""
    inputs_b, pred_b, targets_b = load_test_results(before_dir)
    inputs_a, pred_a, targets_a = load_test_results(after_dir)

    if inputs_b.shape[1] != 96 or targets_b.shape[1] != 720:
        raise ValueError(
            f"Expected a 96-to-720 experiment, got {inputs_b.shape[1]}-to-{targets_b.shape[1]}."
        )
    if inputs_b.shape != inputs_a.shape or targets_b.shape != targets_a.shape:
        raise ValueError("Baseline and DropoutTS result shapes do not match.")
    if not np.allclose(targets_b, targets_a):
        raise ValueError("Baseline and DropoutTS targets are not identical.")
    if max(indices) >= len(targets_b):
        raise IndexError("One or more requested sample indices are outside the test set.")

    base_mae = np.mean(np.abs(pred_b[:, :, feat_idx] - targets_b[:, :, feat_idx]), axis=1)
    ours_mae = np.mean(np.abs(pred_a[:, :, feat_idx] - targets_a[:, :, feat_idx]), axis=1)
    improvements = (base_mae - ours_mae) / (base_mae + 1e-9) * 100
    t_in = np.arange(96)
    t_out = np.arange(96, 816)

    samples = []
    for index in indices:
        samples.append({
            "sample_id": index,
            "improvement": float(improvements[index]),
            "input_t": t_in,
            "input_y": inputs_b[index, :, feat_idx],
            "ground_truth_t": t_out,
            "ground_truth_y": targets_b[index, :, feat_idx],
            "baseline_t": t_out,
            "baseline_y": pred_b[index, :, feat_idx],
            "dropoutts_t": t_out,
            "dropoutts_y": pred_a[index, :, feat_idx],
        })
    return samples


def load_cached_samples(source_path):
    """Load the visible paths recovered from the original editable PDF."""
    with np.load(source_path) as source:
        if int(source["input_len"]) != 96 or int(source["output_len"]) != 720:
            raise ValueError("Bundled source must describe a 96-to-720 forecast.")
        samples = []
        for panel, (sample_id, improvement) in enumerate(
            zip(source["sample_ids"], source["improvements"])
        ):
            sample = {
                "sample_id": int(sample_id),
                "improvement": float(improvement),
            }
            for series in ("input", "ground_truth", "baseline", "dropoutts"):
                sample[f"{series}_t"] = source[f"p{panel}_{series}_t"]
                sample[f"{series}_y"] = source[f"p{panel}_{series}_y"]
            samples.append(sample)
    return samples


def plot_samples(samples, output_path):
    set_academic_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(7.16, 5.65), sharex=True, constrained_layout=True
    )

    for panel, (ax, sample) in enumerate(zip(axes.flat, samples)):
        ax.plot(
            sample["input_t"], sample["input_y"], color=COLORS["input"],
            label="Input", lw=1.7, alpha=0.5,
        )
        ax.plot(
            sample["ground_truth_t"], sample["ground_truth_y"],
            color=COLORS["ground_truth"], label="Ground Truth", lw=2.1,
        )
        ax.plot(
            sample["baseline_t"], sample["baseline_y"],
            color=COLORS["baseline"], label="Baseline", lw=2.1, ls="--",
        )
        ax.plot(
            sample["dropoutts_t"], sample["dropoutts_y"],
            color=COLORS["dropoutts"], label="Ours (+ SACM)", lw=2.2,
        )
        ax.axvline(95, color=COLORS["boundary"], linestyle=":", lw=1.25)
        ax.set_title(
            f"({chr(97 + panel)})  Sample #{sample['sample_id']}  |  "
            f"Imp: +{sample['improvement']:.1f}%",
            loc="center", pad=4, fontsize=10.5, fontweight="semibold",
        )
        style_axis(ax, None)
        ax.tick_params(axis="both", labelsize=10.0)
        ax.grid(
            True, color=PALETTE["grid"], linestyle="--", linewidth=0.65,
            alpha=0.7,
        )
        if panel >= 2:
            ax.set_xlabel("Time Steps", fontsize=11.0)
        if panel % 2 == 0:
            ax.set_ylabel("Normalized Value", fontsize=11.0)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=4, frameon=False,
        fontsize=9.5, handlelength=2.2, columnspacing=1.3, handletextpad=0.45,
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig, (output_dir / "dropoutts_visualization.pdf").resolve(),
        formats=EXPORT_FORMATS, dpi=600,
    )


def parse_indices(value):
    indices = tuple(int(item.strip()) for item in value.split(","))
    if len(indices) != 4:
        raise argparse.ArgumentTypeError("Exactly four comma-separated indices are required.")
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--feat", type=int, default=0)
    parser.add_argument("--indices", type=parse_indices, default=DEFAULT_INDICES)
    args = parser.parse_args()

    if bool(args.before) != bool(args.after):
        parser.error("--before and --after must be provided together.")
    if args.before:
        figure_samples = load_checkpoint_samples(
            args.before, args.after, args.feat, args.indices
        )
    else:
        figure_samples = load_cached_samples(args.source)

    plot_samples(figure_samples, args.output)

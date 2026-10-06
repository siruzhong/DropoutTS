#!/usr/bin/env python3
"""Task generality: classification accuracy gains and anomaly-detection F1 gains."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path.cwd()
sys.path.insert(0, str(ROOT / "vis"))

from tkde_style import PALETTE, apply_tkde_style, panel_title, save_figure, style_axis


apply_tkde_style()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

MODELS_CLASSIFICATION = [
    "iTransformer",
    "PatchTST",
    "NSFormer",
    "TimesNet",
    "InceptionTime",
    "MiniROCKET",
]
MODELS_ANOMALY = [
    "PatchTST",
    "iTransformer",
    "NSFormer",
    "AnomTrans",
    "TranAD",
    "DCdetector",
    "TimesNet",
]

CLASSIFICATION_LABELS = {
    "ArticularyWordRecognition": "AWR",
    "AtrialFibrillation": "AF",
    "BasicMotions": "BM",
    "CharacterTrajectories": "CT",
    "Cricket": "Cricket",
    "DuckDuckGeese": "DDG",
    "EigenWorms": "EW",
    "Epilepsy": "Epil.",
    "EthanolConcentration": "EC",
    "ERing": "ER",
    "FaceDetection": "FD",
    "FingerMovements": "FM",
    "HandMovementDirection": "HMD",
    "Handwriting": "HW",
    "Heartbeat": "HB",
    "InsectWingbeat": "IW",
    "JapaneseVowels": "JV",
    "Libras": "Libras",
    "LSST": "LSST",
    "MotorImagery": "MI",
    "NATOPS": "NATOPS",
    "PenDigits": "PD",
    "PEMS-SF": "PEMS",
    "PhonemeSpectra": "PS",
    "RacketSports": "RS",
    "SelfRegulationSCP1": "SCP1",
    "SelfRegulationSCP2": "SCP2",
    "SpokenArabicDigits": "SAD",
    "StandWalkJump": "SWJ",
    "UWaveGestureLibrary": "UWave",
    "HAR": "HAR",
    "Sleep-EDF": "Sleep",
}


def export_figure(fig: plt.Figure, stem: str) -> None:
    save_figure(
        fig,
        FIG_DIR / f"{stem}.pdf",
        formats=(".pdf",),
        dpi=600,
    )


def parse_classification_changes() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    source = (Path.cwd() / "task_generality_classification_anomaly_classification.tex").read_text(encoding="utf-8")
    row_pattern = re.compile(r"\\textit\{([^}]+)\}\s*&([^\n]+)")
    pair_pattern = re.compile(r"\\clsraw\{([0-9.]+)\}\{([0-9.]+)\}")
    datasets: list[str] = []
    changes = []
    raw_means = []
    dt_means = []
    for dataset, body in row_pattern.findall(source):
        pairs = [(float(raw), float(dt)) for raw, dt in pair_pattern.findall(body)]
        if pairs:
            if len(pairs) != len(MODELS_CLASSIFICATION):
                raise ValueError("Classification row does not contain six backbone pairs.")
            datasets.append(dataset)
            changes.append([dt - raw for raw, dt in pairs])
            raw_means.append(float(np.mean([raw for raw, _ in pairs])))
            dt_means.append(float(np.mean([dt for _, dt in pairs])))
    matrix = np.asarray(changes)
    if matrix.shape != (32, 6):
        raise ValueError(f"Expected 32 x 6 classification pairs, found {matrix.shape}.")
    return datasets, matrix, np.asarray(raw_means), np.asarray(dt_means)


def parse_anomaly_f1_changes() -> tuple[list[str], np.ndarray]:
    source = (Path.cwd() / "task_generality_classification_anomaly_anomaly.tex").read_text(encoding="utf-8")
    pair_pattern = re.compile(r"\\adfraw\{([0-9.]+)\}\{([0-9.]+)\}")
    dataset_pattern = re.compile(r"\\textit\{(SMD|MSL|SMAP|PSM)\}")
    datasets: list[str] = []
    changes: list[list[float]] = []
    current_dataset = ""
    for line in source.splitlines():
        match = dataset_pattern.search(line)
        if match:
            current_dataset = match.group(1)
        if "& F1" not in line:
            continue
        pairs = [(float(raw), float(dt)) for raw, dt in pair_pattern.findall(line)]
        if len(pairs) != len(MODELS_ANOMALY):
            raise ValueError("Anomaly F1 row does not contain seven backbone pairs.")
        datasets.append(current_dataset)
        changes.append([dt - raw for raw, dt in pairs])
    matrix = np.asarray(changes)
    if matrix.shape != (4, 7):
        raise ValueError(f"Expected 4 x 7 anomaly pairs, found {matrix.shape}.")
    return datasets, matrix


def _luminance(rgba) -> float:
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]


def plot_task_generality() -> None:
    classification_datasets, classification, cls_raw, cls_dt = parse_classification_changes()
    anomaly_datasets, anomaly = parse_anomaly_f1_changes()

    # Wider canvas: (a) gain bars + side dual-bars; (b) heatmap + row-mean bars
    fig = plt.figure(figsize=(8.6, 4.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.10)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[2.7, 1.0], wspace=0.07)
    gs_bot = gs[1].subgridspec(1, 3, width_ratios=[3.55, 0.95, 0.038], wspace=0.05)
    ax = fig.add_subplot(gs_top[0, 0])
    ax_side = fig.add_subplot(gs_top[0, 1])
    ax_hm = fig.add_subplot(gs_bot[0, 0])
    ax_mean = fig.add_subplot(gs_bot[0, 1], sharey=ax_hm)
    cax = fig.add_subplot(gs_bot[0, 2])

    # ── (a) Classification gains (sqrt height for dynamic range) ───────────
    positions = np.arange(len(classification_datasets))
    mean_changes = classification.mean(axis=1)
    overall = float(classification.mean())
    # sqrt display compresses a few large gains so mid/low bars stay visible
    heights = np.sqrt(np.maximum(mean_changes, 0.0))
    overall_h = float(np.sqrt(max(overall, 0.0)))

    vmax_a = float(mean_changes.max())
    cmap_a = LinearSegmentedColormap.from_list(
        "cls_gain",
        ["#D6E4F0", PALETTE["ours_light"], PALETTE["ours"], "#1F4E79"],
    )
    norm_a = Normalize(vmin=0.0, vmax=vmax_a)
    bar_colors = [cmap_a(norm_a(v)) for v in mean_changes]

    ax.bar(
        positions,
        heights,
        width=0.78,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    # Light band under the mean line: all bars are positive gains
    ax.axhspan(0, overall_h, color=PALETTE["ours"], alpha=0.06, zorder=0, linewidth=0)
    ax.axhline(overall_h, color=PALETTE["warning"], linewidth=1.15, linestyle="--", zorder=4)
    ax.annotate(
        f"mean +{overall:.2f}",
        xy=(len(positions) - 0.55, overall_h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["warning"],
        fontweight="semibold",
        zorder=5,
    )

    for x, h, v in zip(positions, heights, mean_changes):
        ax.text(
            x,
            h + 0.04,
            f"+{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.4,
            color=PALETTE["dark"],
            rotation=90,
            clip_on=False,
            zorder=5,
        )

    short_labels = [CLASSIFICATION_LABELS[name] for name in classification_datasets]
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, rotation=48, ha="right", rotation_mode="anchor", fontsize=6.6)
    ax.set_ylabel("Mean accuracy gain (pp)")
    # True-value ticks on a sqrt-mapped axis
    tick_vals = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    tick_vals = tick_vals[tick_vals <= vmax_a * 1.02]
    ax.set_yticks(np.sqrt(tick_vals))
    ax.set_yticklabels([f"{t:g}" for t in tick_vals])
    ax.set_ylim(0, float(heights.max()) * 1.06)
    ax.set_xlim(-0.55, len(positions) - 0.45)
    panel_title(ax, "(a)", f"Classification gains across 32 datasets  (mean +{overall:.2f} pp)")
    style_axis(ax, grid_axis="y")

    # Side panel: Raw vs + SACM on selected datasets (absolute accuracy; no occlusion)
    side_names = [
        "StandWalkJump",
        "EigenWorms",
        "DuckDuckGeese",
        "FingerMovements",
        "ERing",
    ]
    name_to_idx = {n: i for i, n in enumerate(classification_datasets)}
    side_idx = [name_to_idx[n] for n in side_names]
    side_raw = cls_raw[side_idx]
    side_dt = cls_dt[side_idx]
    side_labels = [CLASSIFICATION_LABELS[n] for n in side_names]

    x_in = np.arange(len(side_names))
    w = 0.36
    ax_side.bar(
        x_in - w / 2,
        side_raw,
        width=w,
        color=PALETTE["baseline"],
        edgecolor="white",
        linewidth=0.3,
        label="Raw",
        zorder=3,
    )
    ax_side.bar(
        x_in + w / 2,
        side_dt,
        width=w,
        color=PALETTE["ours"],
        edgecolor="white",
        linewidth=0.3,
        label="+ SACM",
        zorder=3,
    )
    for i, (r, d) in enumerate(zip(side_raw, side_dt)):
        ax_side.annotate(
            "",
            xy=(i + w / 2, d),
            xytext=(i - w / 2, r),
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["dark"],
                lw=0.7,
                connectionstyle="arc3,rad=0",
            ),
            zorder=4,
        )
        ax_side.text(
            i,
            max(r, d) + 1.8,
            f"+{d - r:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=PALETTE["ours"],
            fontweight="semibold",
            zorder=5,
        )
    ax_side.set_xticks(x_in)
    ax_side.set_xticklabels(side_labels, fontsize=6.4, rotation=22, ha="right", rotation_mode="anchor")
    ax_side.set_ylabel("Acc. (%)", fontsize=7.2, labelpad=1)
    y_lo = float(min(side_raw.min(), side_dt.min()) - 4)
    y_hi = float(max(side_raw.max(), side_dt.max()) + 9)
    ax_side.set_ylim(y_lo, y_hi)
    ax_side.tick_params(axis="both", labelsize=6.0, length=2.0, width=0.5, pad=1)
    ax_side.legend(
        loc="upper left",
        fontsize=5.8,
        frameon=True,
        fancybox=False,
        edgecolor=PALETTE["light"],
        framealpha=0.95,
        borderpad=0.3,
        handlelength=0.95,
        handletextpad=0.35,
        labelspacing=0.18,
    )
    panel_title(ax_side, "", "Raw vs + SACM (selected)")
    style_axis(ax_side, grid_axis="y")

    # ── (b) Anomaly F1: heatmap (dataset × model) + right-side mean bars ──
    display = anomaly  # 4 × 7 only; mean shown as side bars (cleaner than extra row)
    color_cap = float(np.percentile(anomaly, 95))
    cmap = LinearSegmentedColormap.from_list(
        "anomaly_gain",
        ["#F7FAFC", "#B7D0E4", PALETTE["ours_light"], PALETTE["ours"], "#1F4E79"],
    )
    norm = Normalize(vmin=0.0, vmax=max(color_cap, 1.0))

    image = ax_hm.imshow(display, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.set_xticks(np.arange(display.shape[1] + 1) - 0.5, minor=True)
    ax_hm.set_yticks(np.arange(display.shape[0] + 1) - 0.5, minor=True)
    ax_hm.grid(which="minor", color="white", linestyle="-", linewidth=1.15)
    ax_hm.tick_params(which="minor", bottom=False, left=False, length=0)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display[i, j]
            rgba = cmap(norm(min(value, color_cap)))
            text_color = "white" if _luminance(rgba) < 0.52 else PALETTE["dark"]
            ax_hm.text(
                j,
                i,
                f"+{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
                fontweight="normal",
            )

    model_labels = [
        m.replace("iTransformer", "iTrans.")
        .replace("DCdetector", "DCdet.")
        .replace("AnomTrans", "AnomT.")
        .replace("PatchTST", "PatchTST")
        .replace("NSFormer", "NSFormer")
        .replace("TimesNet", "TimesNet")
        .replace("TranAD", "TranAD")
        for m in MODELS_ANOMALY
    ]
    ax_hm.set_xticks(range(len(MODELS_ANOMALY)))
    ax_hm.set_xticklabels(model_labels, rotation=0, ha="center", fontsize=7.4)
    ax_hm.set_yticks(range(len(anomaly_datasets)))
    ax_hm.set_yticklabels(anomaly_datasets, fontsize=8.0)
    ax_hm.tick_params(axis="both", length=0)

    overall_ad = float(anomaly.mean())
    panel_title(
        ax_hm,
        "(b)",
        f"Anomaly F1 gains  (mean +{overall_ad:.2f} pp; W/T/L = 28/0/0)",
    )

    # Per-dataset mean F1 gain as horizontal bars (summary of each row)
    row_means = anomaly.mean(axis=1)
    y_pos = np.arange(len(anomaly_datasets))
    ax_mean.barh(
        y_pos,
        row_means,
        height=0.62,
        color=PALETTE["ours"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    for y, v in zip(y_pos, row_means):
        ax_mean.text(
            v + 0.12,
            y,
            f"+{v:.1f}",
            ha="left",
            va="center",
            fontsize=6.6,
            color=PALETTE["dark"],
            fontweight="semibold",
            zorder=4,
        )
    ax_mean.axvline(overall_ad, color=PALETTE["warning"], linewidth=1.05, linestyle="--", zorder=2)
    ax_mean.set_xlabel("Mean gain (pp)", fontsize=7.0, labelpad=1)
    ax_mean.set_xlim(0, float(row_means.max()) * 1.28)
    ax_mean.tick_params(axis="y", labelleft=False, length=0)
    ax_mean.tick_params(axis="x", labelsize=6.4, length=2.0, width=0.5)
    style_axis(ax_mean, grid_axis="x")
    ax_mean.set_title("by dataset", fontsize=6.6, pad=2, color=PALETTE["dark"], loc="left")

    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_label("F1 gain (pp)", labelpad=2, fontsize=7.0)
    cbar.ax.tick_params(labelsize=7.0, width=0.5, length=1.8)
    cbar.outline.set_linewidth(0.45)

    export_figure(fig, "task_generality_classification_anomaly")


def main() -> None:
    plot_task_generality()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adaptive-dropout strategy comparison and component-ablation analysis."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.cwd()
FIG_DIR = Path.cwd()
sys.path.insert(0, str(ROOT / "vis"))

from tkde_style import PALETTE, apply_tkde_style, panel_title, save_figure, style_axis


apply_tkde_style()
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_rows(filename: str) -> list[dict[str, str]]:
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def export_figure(fig: plt.Figure, stem: str) -> None:
    save_figure(
        fig,
        FIG_DIR / f"{stem}.pdf",
        formats=(".pdf",),
        dpi=600,
    )


def percent_reduction(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference


def plot_adaptation_and_ablation() -> None:
    strategy = read_rows("adaptation_component_analysis_strategy.csv")
    ablation = read_rows("adaptation_component_analysis_ablation.csv")

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.16, 3.15), constrained_layout=True)

    columns = ["best_fixed_mse", "learned_global_mse", "dropoutts_mse"]
    values = np.array(
        [
            [percent_reduction(float(row["original_mse"]), float(row[column])) for column in columns]
            for row in strategy
        ]
    )
    backbone_labels = {"TimeFilter": "TF", "PatchTST": "PT", "TimeMixer": "TM", "Informer": "Inf"}
    row_labels = [
        f'{"ILI" if row["dataset"] == "ILI" else "EX"} {backbone_labels[row["backbone"]]}'
        for row in strategy
    ]
    cmap = LinearSegmentedColormap.from_list(
        "adaptation",
        ["#F3C9BE", "#FFFFFF", PALETTE["ours"]],
    )
    norm = TwoSlopeNorm(vmin=min(-2.5, float(values.min())), vcenter=0, vmax=55)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False, length=0)
    ax.axhline(3.5, color="white", linewidth=2.4, zorder=4)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=10.0,
                color="white" if luminance < 0.52 else PALETTE["dark"],
                fontweight="semibold" if j == 2 else "normal",
            )
    ax.set_xticks(range(3), labels=["Fixed", "Global", "+ SACM"])
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="both", length=0, labelsize=10.0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("MSE improvement (%)")
    panel_title(ax, "(a)", "Adaptive dropout")
    cbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.048, pad=0.025, aspect=16)
    cbar.set_ticks([0, 15, 30, 45])
    cbar.ax.tick_params(labelsize=9.5, width=0.6, length=2.2)
    cbar.outline.set_linewidth(0.5)

    ypos = np.arange(len(ablation))[::-1]
    mse = np.array([float(row["mse_gain_pct"]) for row in ablation])
    mae = np.array([float(row["mae_gain_pct"]) for row in ablation])
    colors = [PALETTE["ours"] if row["variant"] == "DropoutTS" else PALETTE["baseline"] for row in ablation]
    axb.barh(ypos, mse, height=0.58, color=colors, edgecolor="white", linewidth=0.5, zorder=2)
    axb.scatter(
        mae,
        ypos,
        s=42,
        marker="D",
        facecolor="white",
        edgecolor=PALETTE["dark"],
        linewidth=1.05,
        zorder=3,
    )
    for value, y in zip(mse, ypos):
        axb.text(
            9.2 if value >= 0 else -46.5,
            y,
            f"{value:+.1f}",
            va="center",
            ha="left",
            fontsize=9.8,
            color=PALETTE["dark"],
            fontweight="semibold",
        )
    axb.axvline(0, color=PALETTE["dark"], linewidth=0.9, zorder=1)
    axb.set_xlim(-48, 15)
    ablation_labels = {
        "Minimal model": "Minimal",
        "w/o detrend + norm": "No D+N",
        "w/o detrend": "No D",
        "Simple detrend": "Simple D",
        "w/o spectral norm": "No norm",
        "w/o log-SFM anchor": "No SFM",
        "DropoutTS": "Full",
    }
    axb.set_yticks(ypos, labels=[ablation_labels[row["variant"]] for row in ablation])
    axb.tick_params(axis="both", labelsize=10.0)
    axb.set_xlabel("Change vs. fixed (%)")
    panel_title(axb, "(b)", "Ablation")
    handles = [
        mpl.patches.Patch(facecolor=PALETTE["baseline"], edgecolor="white", linewidth=0.5, label="MSE"),
        mpl.lines.Line2D(
            [],
            [],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=PALETTE["dark"],
            markeredgewidth=1.0,
            markersize=6.5,
            label="MAE",
        ),
    ]
    axb.legend(
        handles=handles,
        loc="upper right",
        ncol=1,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.2,
        fontsize=8.0,
        frameon=True,
        fancybox=False,
        edgecolor=PALETTE["light"],
        framealpha=0.92,
    )
    style_axis(axb, grid_axis="x")

    export_figure(fig, "adaptation_component_analysis")


def main() -> None:
    plot_adaptation_and_ablation()


if __name__ == "__main__":
    main()

"""Recover the four qualitative time-series panels from the editable PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pdfplumber


SERIES_BY_COLOR = {
    (0.4980392157, 0.5490196078, 0.5529411765): "input",
    (0.1529411765, 0.6823529412, 0.3764705882): "ground_truth",
    (0.8392156863, 0.1529411765, 0.1568627451): "baseline",
    (0.1215686275, 0.4666666667, 0.7058823529): "dropoutts",
}
SAMPLE_IDS = np.asarray([4208, 5705, 3022, 2580])
IMPROVEMENTS = np.asarray([66.9, 68.9, 63.0, 38.4])

# Zero-line position and vertical points per normalized unit in each source panel.
ZERO_TOP = np.asarray([95.63, 103.29, 257.05, 255.19])
POINTS_PER_UNIT = np.asarray([45.02, 36.62, 48.32, 46.79])


def color_key(value):
    return tuple(round(float(channel), 10) for channel in value)


def panel_index(curve):
    row = 0 if curve["top"] < 180 else 1
    col = 0 if curve["x0"] < 270 else 1
    return row * 2 + col


def recover(source_pdf: Path) -> dict[str, np.ndarray]:
    with pdfplumber.open(source_pdf) as document:
        if len(document.pages) != 1:
            raise ValueError("Expected a one-page qualitative figure PDF.")
        page = document.pages[0]
        curves = [
            curve
            for curve in page.curves
            if len(curve.get("pts", ())) > 50
            and color_key(curve.get("stroking_color", ())) in SERIES_BY_COLOR
        ]

    if len(curves) != 16:
        raise ValueError(f"Expected 16 data curves, recovered {len(curves)}.")

    panels: list[dict[str, object]] = [dict() for _ in range(4)]
    for curve in curves:
        panels[panel_index(curve)][SERIES_BY_COLOR[color_key(curve["stroking_color"])]] = curve

    recovered: dict[str, np.ndarray] = {
        "input_len": np.asarray(96),
        "output_len": np.asarray(720),
        "sample_ids": SAMPLE_IDS,
        "improvements": IMPROVEMENTS,
    }
    for panel, series_curves in enumerate(panels):
        if set(series_curves) != set(SERIES_BY_COLOR.values()):
            raise ValueError(f"Panel {panel} does not contain all four expected series.")
        input_x0 = series_curves["input"]["x0"]
        output_x0 = series_curves["ground_truth"]["x0"]
        points_per_step = (output_x0 - input_x0) / 96.0

        for series, curve in series_curves.items():
            points = np.asarray(curve["pts"], dtype=float)
            time = (points[:, 0] - input_x0) / points_per_step
            values = (ZERO_TOP[panel] - points[:, 1]) / POINTS_PER_UNIT[panel]
            keep = np.r_[True, np.diff(time) > 1e-8]
            recovered[f"p{panel}_{series}_t"] = time[keep]
            recovered[f"p{panel}_{series}_y"] = values[keep]

    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("output_npz", type=Path)
    args = parser.parse_args()
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **recover(args.source_pdf))


if __name__ == "__main__":
    main()

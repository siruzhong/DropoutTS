"""Shared IEEE TKDE publication styling for the figures in this directory."""

from pathlib import Path

import matplotlib as mpl


PALETTE = {
    "ours": "#356D9F",
    "ours_light": "#7EA6C8",
    "baseline": "#D77968",
    "signal": "#4E9A8D",
    "accent": "#8A78B3",
    "warning": "#D9A441",
    "neutral": "#7B8490",
    "dark": "#30343B",
    "light": "#D2D7DD",
    "grid": "#E6E9EC",
    "positive": "#4F8A70",
}

SERIES_COLORS = [
    PALETTE["ours"],
    PALETTE["warning"],
    PALETTE["positive"],
    PALETTE["accent"],
    PALETTE["signal"],
]

SERIES_MARKERS = ["o", "s", "D", "^", "v"]
SERIES_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1))]


def apply_tkde_style() -> None:
    """Apply a compact, editable, grayscale-safe IEEE journal style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["dark"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax, grid_axis: str | None = "y") -> None:
    """Apply consistent spine, tick, and optional light-grid treatment."""
    ax.spines["left"].set_color(PALETTE["dark"])
    ax.spines["bottom"].set_color(PALETTE["dark"])
    ax.tick_params(colors=PALETTE["dark"])
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=PALETTE["grid"],
            linewidth=0.55,
            alpha=0.65,
        )


def panel_title(ax, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", pad=4)


def save_figure(
    fig,
    filename: str | Path,
    *,
    formats: tuple[str, ...] = (".pdf",),
    dpi: int = 600,
    close: bool = True,
) -> list[Path]:
    """Save the publication PDF at a deterministic location."""
    import matplotlib.pyplot as plt

    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.with_suffix("")
    saved = []
    for fmt in formats:
        fmt = fmt.lstrip(".")
        output = base.with_suffix(f".{fmt}")
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(output)
    if close:
        plt.close(fig)
    return saved

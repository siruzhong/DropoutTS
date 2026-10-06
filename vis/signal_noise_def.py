import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from tkde_style import PALETTE, apply_tkde_style, panel_title, save_figure, style_axis

apply_tkde_style()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.labelsize": 11.0,
    "axes.titlesize": 11.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
EXPORT_FORMATS = (".pdf",)

# ==========================================
# 2. Data generation
# ==========================================
np.random.seed(42)  # fix the seed for reproducibility
t = np.linspace(0, 3, 1000)  # time axis in seconds

# --- Base signal parameters ---
freq = 1.0  # base frequency, 1 Hz

# --- Four signal regimes ---
# 1. Stationary (Periodic)
y_stationary = 4 * np.sin(2 * np.pi * freq * t)

# 2. Non-stationary Mean (Trend)
# Linear trend + sinusoid
y_trend = 0.5 * t * 2 + 2 * np.sin(2 * np.pi * freq * t) - 1

# 3. Non-stationary Frequency (Chirp)
# Frequency increases linearly with time: f(t) = f0 + k*t
f0 = 0.5
k = 2.0
phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
y_chirp = 4 * np.sin(phase)

# 4. Non-stationary Variance (AM - Amplitude Modulation)
# Carrier times envelope
carrier_freq = 10.0
mod_freq = 0.5
envelope = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
y_am = 2.5 * envelope * np.sin(2 * np.pi * carrier_freq * t)

# --- Three noise profiles ---
# Base signal used to illustrate the noise
y_base = y_stationary.copy()

# 1. Gaussian Noise (Aleatoric)
noise_gaussian = np.random.normal(0, 0.5, size=t.shape)
y_gaussian = y_base + noise_gaussian

# 2. Heavy-tail Noise (Student-t)
# Student-t distribution (df=2.5) to simulate extreme values
noise_heavy = np.random.standard_t(df=2.5, size=t.shape) * 0.3
y_heavy = y_base + noise_heavy

# 3. Missing Values (Failures)
# Randomly mask 40% of the data
mask = np.random.choice([0, 1], size=t.shape, p=[0.4, 0.6])
y_missing = y_base.copy()
y_missing[mask == 0] = np.nan  # set to NaN so matplotlib breaks the line

# ==========================================
# 3. Plotting
# ==========================================
fig = plt.figure(figsize=(7.16, 4.65), constrained_layout=True)

# Use GridSpec for a two-row layout
# Height ratio 1:1; hspace slightly increased
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.28, figure=fig)

# --- Row 1: four signal regimes ---
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.08)
regime_titles = ["Stationary", "Mean shift", "Frequency shift", "Variance shift"]
regime_data = [y_stationary, y_trend, y_chirp, y_am]

for i in range(4):
    ax = fig.add_subplot(gs_row1[0, i])
    ax.plot(t, regime_data[i], color=PALETTE['ours'], lw=1.85)
    panel_title(ax, f"({chr(97+i)})", regime_titles[i])
    ax.set_xlabel("Time", fontsize=10.5)
    if i == 0:
        ax.set_ylabel("Amplitude", fontsize=10.5)
    ax.tick_params(axis='both', labelsize=9.5)
    style_axis(ax)
    ax.set_ylim(-4.5, 4.5)

# --- Row 2: three noise profiles ---
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.08)
noise_titles = ["Gaussian noise", "Heavy-tailed noise", "Missing observations"]

# Prepare plotting data (clean, noisy)
noise_plot_data = [
    (y_base, y_gaussian),
    (y_base, y_heavy),
    (y_base, y_missing)
]

for i in range(3):
    ax = fig.add_subplot(gs_row2[0, i])
    clean_sig, noisy_sig = noise_plot_data[i]
    
    # Special handling for the legend and rendering of missing values
    if i == 2:
        # Light line: complete original signal
        ax.plot(t, y_base, color=PALETTE['dark'], alpha=0.48, lw=1.85, label='Clean signal')
        # Dark line: observed values
        ax.plot(t, noisy_sig, color=PALETTE['ours'], lw=1.85, label='Observed signal')
    else:
        # For the first two, plot the clean and the corrupted signal
        ax.plot(t, clean_sig, color=PALETTE['dark'], alpha=0.58, lw=1.85, label='Clean signal')
        ax.plot(t, noisy_sig, color=PALETTE['baseline'], alpha=0.82, lw=1.15, label='Corrupted signal')
    
    panel_title(ax, f"({chr(101+i)})", noise_titles[i])
    ax.set_xlabel("Time", fontsize=10.5)
    if i == 0:
        ax.set_ylabel("Amplitude", fontsize=10.5)
    ax.tick_params(axis='both', labelsize=9.5)
    style_axis(ax)
    
    # Widen the y-range for heavy tails so spikes are visible without overstating them
    if i == 1:
        ax.set_ylim(-6, 6)
    else:
        ax.set_ylim(-4.5, 4.5)

legend_handles = [
    Line2D([0], [0], color=PALETTE['dark'], lw=1.8, label='Clean signal'),
    Line2D([0], [0], color=PALETTE['baseline'], lw=1.5, label='Corrupted signal'),
    Line2D([0], [0], color=PALETTE['ours'], lw=1.8, label='Observed samples'),
]
fig.legend(
    handles=legend_handles,
    loc='outside upper center',
    ncol=3,
    fontsize=9.5,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
    handletextpad=0.45,
)
save_figure(fig, 'signal_noise_def.pdf', formats=EXPORT_FORMATS, dpi=600)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tkde_style import SERIES_COLORS, SERIES_LINESTYLES, SERIES_MARKERS, apply_tkde_style, panel_title, save_figure, style_axis

apply_tkde_style()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.labelsize": 11.0,
    "axes.titlesize": 11.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.2,
    "legend.title_fontsize": 9.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
EXPORT_FORMATS = (".pdf",)
SCRIPT_DIR = Path(__file__).resolve().parent

# Source points recovered from the original editable vector figure. Grouped
# means reproduce the original curves to the displayed 0.001 MSE precision.
df = pd.read_csv(SCRIPT_DIR / 'parameter_sensitivity_analysis.csv')

# 3. Data filtering and preprocessing
df_synth = df[df['Dataset'].str.contains('SyntheticTS') &
              (df['Has_DropoutTS'] == 'Yes') &
              (df['sparsity_weight'] == 0.00)].copy()

# Extract the noise level
df_synth['Noise Level'] = df_synth['Dataset'].str.extract(r'noise(\d+\.\d+)').astype(float)

# Build string-format labels
df_synth['Noise Label'] = df_synth['Noise Level'].apply(lambda x: f'{x:.1f}')
df_synth['Sensitivity Label'] = df_synth['init_sensitivity'].apply(lambda x: f'{x:.1f}')

# 4. IEEE two-column canvas (slightly taller to fit large fonts)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), constrained_layout=True)

# Define the color scheme
unique_ws = sorted(df_synth['init_sensitivity'].unique())
unique_noises = sorted(df_synth['Noise Level'].unique())

# Left panel: colors for the three sensitivity values
palette_ws = SERIES_COLORS[:3]
# Right panel: colors for the five noise levels
palette_noise = SERIES_COLORS

# ======================== Left panel: impact of w_s across noise levels ========================
# Aggregate the available experiment-level observations.
df_avg = df_synth.groupby(['Noise Level', 'init_sensitivity']).agg({
    'MSE': ['mean', 'std']
}).reset_index()
df_avg.columns = ['Noise Level', 'init_sensitivity', 'MSE_mean', 'MSE_std']

# Plot the line curves
for i, ws in enumerate(unique_ws):
    data_ws = df_avg[df_avg['init_sensitivity'] == ws]
    axes[0].plot(
        data_ws['Noise Level'],
        data_ws['MSE_mean'],
        marker=SERIES_MARKERS[i],
        linestyle=SERIES_LINESTYLES[i],
        markersize=7.0,
        linewidth=2.1,
        color=palette_ws[i],
        label=f'{ws:.1f}',
        zorder=3,
        markeredgecolor='white',
        markeredgewidth=0.55,
    )
    axes[0].fill_between(
        data_ws['Noise Level'],
        data_ws['MSE_mean'] - data_ws['MSE_std'].fillna(0),
        data_ws['MSE_mean'] + data_ws['MSE_std'].fillna(0),
        color=palette_ws[i],
        alpha=0.12,
        linewidth=0,
    )

panel_title(axes[0], '(a)', r'Sensitivity $\gamma$ across noise levels')
axes[0].set_xlabel(r'Noise Level $\sigma$')
axes[0].set_ylabel('MSE')
axes[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
axes[0].set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
axes[0].tick_params(axis='both', labelsize=10.0)
leg0 = axes[0].legend(
    title=r'Sensitivity $\gamma$',
    loc='upper left',
    fontsize=9.2,
    title_fontsize=9.5,
    frameon=True,
    fancybox=False,
    edgecolor='#D2D7DD',
    framealpha=0.95,
    borderpad=0.35,
    handlelength=1.6,
)
axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.12)
style_axis(axes[0])

# ======================== Right panel: scatter plot + mean lines ========================
# Show all raw points as a scatter and connect the means

# Give each sensitivity value its own sub-position (grouped layout)
sensitivity_positions = {1.0: 0, 5.0: 1, 10.0: 2}
noise_offset = {0.1: -0.15, 0.3: -0.075, 0.5: 0, 0.7: 0.075, 0.9: 0.15}

# Plot the raw points
for i, noise in enumerate(unique_noises):
    for j, sens in enumerate(unique_ws):
        data_subset = df_synth[(df_synth['Noise Level'] == noise) &
                               (df_synth['init_sensitivity'] == sens)]

        if len(data_subset) > 0:
            # x position = sensitivity base position + noise-level offset
            x_pos = sensitivity_positions[sens] + noise_offset[noise]

            # Plot the raw data points (semi-transparent)
            axes[1].scatter(
                [x_pos] * len(data_subset),
                data_subset['MSE'],
                color=palette_noise[i],
                alpha=0.38,
                s=28,
                edgecolors='white',
                linewidth=0.55,
                zorder=2
            )

            # Plot the mean marker (filled, more prominent)
            mean_val = data_subset['MSE'].mean()
            axes[1].scatter(
                x_pos,
                mean_val,
                color=palette_noise[i],
                marker=SERIES_MARKERS[i],
                s=42,
                edgecolors='white',
                linewidth=0.7,
                zorder=3,
                label=f'{noise:.1f}' if j == 0 else None  # add the legend entry only for the first sensitivity
            )

# Draw a line connecting the means for each noise level
for i, noise in enumerate(unique_noises):
    x_positions = []
    y_means = []

    for sens in unique_ws:
        data_subset = df_synth[(df_synth['Noise Level'] == noise) &
                               (df_synth['init_sensitivity'] == sens)]
        if len(data_subset) > 0:
            x_pos = sensitivity_positions[sens] + noise_offset[noise]
            x_positions.append(x_pos)
            y_means.append(data_subset['MSE'].mean())

    # Draw the connecting line
    if len(x_positions) > 1:
        axes[1].plot(x_positions, y_means,
                    color=palette_noise[i],
                    linewidth=1.35,
                    alpha=0.7,
                    linestyle=SERIES_LINESTYLES[i],
                    zorder=1)

# Set the x-axis
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['1.0', '5.0', '10.0'])
axes[1].set_xlabel(r'Sensitivity $\gamma$')
axes[1].set_ylabel('MSE')
axes[1].tick_params(axis='both', labelsize=10.0)
panel_title(axes[1], '(b)', r'MSE distribution by $\gamma$')

# Legend: noise levels
axes[1].legend(
    title=r'Noise $\sigma$',
    loc='upper left',
    ncol=1,
    fontsize=9.0,
    title_fontsize=9.5,
    handletextpad=0.45,
    columnspacing=0.7,
    borderpad=0.35,
    frameon=True,
    fancybox=False,
    edgecolor='#D2D7DD',
    framealpha=0.95,
)

axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.12)
style_axis(axes[1])

# Save the high-resolution figure
save_figure(fig, 'parameter_sensitivity_analysis.pdf', formats=EXPORT_FORMATS, dpi=600)

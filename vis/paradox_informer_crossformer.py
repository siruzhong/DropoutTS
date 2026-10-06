import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tkde_style import PALETTE, apply_tkde_style, panel_title, save_figure, style_axis

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
    "lines.markersize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
EXPORT_FORMATS = (".pdf",)

# 2. Prepare data (extracted from the LaTeX tables)
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

# Informer: unstable -> stable (largest gains)
informer_raw = [0.966, 0.978, 0.936, 0.841, 0.828]
informer_dt  = [0.514, 0.507, 0.494, 0.471, 0.464]

# Crossformer: already stable -> still lower (incremental gains)
cross_raw = [0.450, 0.439, 0.431, 0.411, 0.409]
cross_dt  = [0.386, 0.378, 0.391, 0.379, 0.360]

informer_reduction = (np.array(informer_raw) - np.array(informer_dt)) / np.array(informer_raw) * 100
cross_reduction = (np.array(cross_raw) - np.array(cross_dt)) / np.array(cross_raw) * 100

# 3. Plot (IEEE two-column width, slightly taller for large fonts)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), constrained_layout=True)

# --- Left panel: Informer ---
ax1 = axes[0]
ax1.plot(
    noise_levels, informer_raw, 'o--',
    color=PALETTE['baseline'], label='Backbone',
    markersize=7.0, linewidth=2.0, markeredgecolor='white', markeredgewidth=0.55,
)
ax1.plot(
    noise_levels, informer_dt, 's-',
    color=PALETTE['ours'], label='Backbone + SACM',
    markersize=6.5, linewidth=2.15, markeredgecolor='white', markeredgewidth=0.55,
)

ax1.fill_between(noise_levels, informer_dt, informer_raw,
                 color=PALETTE['ours_light'], alpha=0.18, linewidth=0, zorder=0)
informer_avg_imp = informer_reduction.mean()
ax1.annotate(
    "",
    xy=(0.84, informer_dt[-1]),
    xytext=(0.84, informer_raw[-1]),
    arrowprops=dict(arrowstyle='<->', color=PALETTE['neutral'], lw=1.05),
)
ax1.text(0.88, 0.5 * (informer_raw[-1] + informer_dt[-1]),
         f"−{informer_reduction[-1]:.0f}%",
         color=PALETTE['neutral'], fontsize=9.0, ha='left', va='center')
# Mean reduction centered between the two curves
ax1.text(0.5, 0.5 * (informer_raw[2] + informer_dt[2]),
         f"Mean −{informer_avg_imp:.0f}%",
         color=PALETTE['ours'], fontsize=9.5,
         ha='center', va='center', fontweight='semibold')

# Low / mid / high noise regime markers (labels just inside top edge)
noise_regime_x = [0.2, 0.5, 0.8]
noise_regime_labels = ['Low', 'Mid', 'High']
for x, lab in zip(noise_regime_x, noise_regime_labels):
    ax1.axvline(x, color=PALETTE['light'], linestyle='--', linewidth=1.0, zorder=0)
    ax1.text(x, 0.97, lab, transform=ax1.get_xaxis_transform(),
             ha='center', va='top', fontsize=8.2, color=PALETTE['neutral'])

panel_title(ax1, '(a)', 'Informer')
ax1.set_xlabel(r'Noise Level ($\sigma$)')
ax1.set_ylabel('MSE')
ax1.set_xticks(noise_levels)
ax1.tick_params(axis='both', labelsize=10.0)
ax1.set_ylim(0.38, 1.05)
style_axis(ax1)

# --- Right panel: Crossformer ---
ax2 = axes[1]
ax2.plot(
    noise_levels, cross_raw, 'o--',
    color=PALETTE['baseline'], label='Backbone',
    markersize=7.0, linewidth=2.0, markeredgecolor='white', markeredgewidth=0.55,
)
ax2.plot(
    noise_levels, cross_dt, 's-',
    color=PALETTE['ours'], label='Backbone + SACM',
    markersize=6.5, linewidth=2.15, markeredgecolor='white', markeredgewidth=0.55,
)

ax2.fill_between(noise_levels, cross_dt, cross_raw,
                 color=PALETTE['ours_light'], alpha=0.18, linewidth=0, zorder=0)

avg_imp = cross_reduction.mean()
ax2.annotate(
    "",
    xy=(0.84, cross_dt[-1]),
    xytext=(0.84, cross_raw[-1]),
    arrowprops=dict(arrowstyle='<->', color=PALETTE['neutral'], lw=1.05),
)
ax2.text(0.88, 0.5 * (cross_raw[-1] + cross_dt[-1]),
         f"−{cross_reduction[-1]:.0f}%",
         color=PALETTE['neutral'], fontsize=9.0, ha='left', va='center')
ax2.text(0.5, 0.5 * (cross_raw[2] + cross_dt[2]),
         f"Mean −{avg_imp:.0f}%",
         color=PALETTE['ours'], fontsize=9.5,
         ha='center', va='center', fontweight='semibold')

for x, lab in zip(noise_regime_x, noise_regime_labels):
    ax2.axvline(x, color=PALETTE['light'], linestyle='--', linewidth=1.0, zorder=0)
    ax2.text(x, 0.97, lab, transform=ax2.get_xaxis_transform(),
             ha='center', va='top', fontsize=8.2, color=PALETTE['neutral'])

panel_title(ax2, '(b)', 'Crossformer')
ax2.set_xlabel(r'Noise Level ($\sigma$)')
ax2.set_ylabel('MSE')
ax2.set_xticks(noise_levels)
ax2.tick_params(axis='both', labelsize=10.0)
ax2.set_ylim(0.35, 0.47)
style_axis(ax2)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='outside upper center',
    ncol=2,
    fontsize=9.5,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
    handletextpad=0.45,
)
save_figure(fig, 'paradox_informer_crossformer.pdf', formats=EXPORT_FORMATS, dpi=600)

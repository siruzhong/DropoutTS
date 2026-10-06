"""
Visualize that low signal-to-noise ratios are ubiquitous in real-world time series datasets.
This supports the motivation for adaptive regularization in DropoutTS.
Also shows SFM-SNR correlation analysis.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from pathlib import Path

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
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
EXPORT_FORMATS = (".pdf",)
SCRIPT_DIR = Path(__file__).resolve().parent

def compute_sfm_log_scale(signal_data):
    """
    Compute Spectral Flatness Measure (SFM) in log scale, also known as Wiener Entropy.
    
    SFM = exp(Geometric Mean(log A) - log(Arithmetic Mean(A)))
    where A_k = |X_k| is the amplitude at frequency k.
    
    Computing in log scale reduces the impact of extreme values and is more
    perceptually meaningful.
    
    Args:
        signal_data: 1D array of signal values
        
    Returns:
        SFM value (float) in range [0, 1]
    """
    # Compute FFT
    fft_vals = np.fft.rfft(signal_data)
    # Compute amplitude (magnitude)
    amplitudes = np.abs(fft_vals)
    
    # Avoid zeros for log calculation
    epsilon = 1e-10
    amplitudes = amplitudes + epsilon
    
    # Compute in log scale
    log_amplitudes = np.log(amplitudes)
    
    # Geometric mean in log domain = arithmetic mean of logs
    geometric_mean_log = np.mean(log_amplitudes)
    
    # Arithmetic mean: compute in linear domain then take log
    arithmetic_mean_linear = np.mean(amplitudes)
    arithmetic_mean_log = np.log(arithmetic_mean_linear)
    
    # SFM = exp(GM_log - AM_log) = GM / AM
    sfm = np.exp(geometric_mean_log - arithmetic_mean_log)
    
    return np.clip(sfm, 0.0, 1.0)

# Load data
df = pd.read_csv(SCRIPT_DIR / 'low_snr_ubiquitous_analysis.csv')

# Filter datasets: only include specified ones, exclude Traffic
target_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Illness']
# Note: 'Illness' in CSV corresponds to 'ILI' dataset
df_filtered = df[df['Dataset'].isin(target_datasets)].copy()
df_filtered = df_filtered.reset_index(drop=True)

# Extract relevant metrics
datasets = df_filtered['Dataset'].values
snr = df_filtered['SNR (dB, top10% freqs)'].values
noise_power_ratio = df_filtered['Noise Power Ratio'].values * 100  # Convert to percentage
recon_rmse_pct = df_filtered['Recon RMSE (% of std)'].values
recon_corr = df_filtered['Recon Correlation'].values
sfm_original = df_filtered['SFM'].values
cv = df_filtered['CV'].values

print(f"Filtered datasets: {list(datasets)}")
print(f"Excluded: Traffic")

# Recompute SFM in log scale from original data
print("Recomputing SFM in log scale from original datasets...")
sfm_log = np.zeros_like(sfm_original)
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../'))
raw_data_dir = os.path.join(base_dir, 'datasets', 'raw_data')

for i, dataset_name in enumerate(datasets):
    dataset_dir = os.path.join(raw_data_dir, dataset_name)
    data_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    
    if os.path.exists(data_path):
        try:
            df_data = pd.read_csv(data_path)
            # Use first numeric column as signal
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                signal = df_data[numeric_cols[0]].values
                # Remove NaN values
                signal = signal[~np.isnan(signal)]
                if len(signal) > 0:
                    sfm_log[i] = compute_sfm_log_scale(signal)
                    print(f"  {dataset_name}: SFM (original) = {sfm_original[i]:.6f}, SFM (log) = {sfm_log[i]:.6f}")
                else:
                    sfm_log[i] = sfm_original[i]
                    print(f"  {dataset_name}: No valid data, using original SFM")
            else:
                sfm_log[i] = sfm_original[i]
                print(f"  {dataset_name}: No numeric columns, using original SFM")
        except Exception as e:
            sfm_log[i] = sfm_original[i]
            print(f"  {dataset_name}: Error computing log-scale SFM ({e}), using original SFM")
    else:
        sfm_log[i] = sfm_original[i]
        print(f"  {dataset_name}: Data file not found, using original SFM")

# Use log-scale SFM
sfm = sfm_log
print(f"\nUsing log-scale SFM values.")
print(f"SFM range: [{np.min(sfm):.6f}, {np.max(sfm):.6f}]")

print("=" * 80)
print("Real-World Datasets: Low SNR is Ubiquitous")
print("=" * 80)
print(f"\nSNR Statistics:")
print(f"  Mean SNR: {np.mean(snr):.2f} dB")
print(f"  Median SNR: {np.median(snr):.2f} dB")
print(f"  Min SNR: {np.min(snr):.2f} dB ({datasets[np.argmin(snr)]})")
print(f"  Max SNR: {np.max(snr):.2f} dB ({datasets[np.argmax(snr)]})")
print(f"  Std SNR: {np.std(snr):.2f} dB")
print(f"\nDatasets with SNR < 15 dB: {np.sum(snr < 15)} / {len(snr)} ({np.sum(snr < 15)/len(snr)*100:.1f}%)")
print(f"Datasets with SNR < 20 dB: {np.sum(snr < 20)} / {len(snr)} ({np.sum(snr < 20)/len(snr)*100:.1f}%)")
print(f"Datasets with SNR < 30 dB: {np.sum(snr < 30)} / {len(snr)} ({np.sum(snr < 30)/len(snr)*100:.1f}%)")
print("=" * 80)

# Calculate SFM-SNR correlations
pearson_r, pearson_p = stats.pearsonr(sfm, snr)
spearman_r, spearman_p = stats.spearmanr(sfm, snr)

# Fit regression line for SFM vs SNR
z = np.polyfit(sfm, snr, 1)
p = np.poly1d(z)
x_line = np.linspace(sfm.min(), sfm.max(), 100)
y_line = p(x_line)

# ============ Create IEEE single-column version (1 row, 3 columns) ============
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.75), constrained_layout=True)

# ============ Left: SNR Distribution by Dataset ============
sorted_idx = np.argsort(snr)[::-1]  # Descending
colors = [PALETTE['baseline'] if s < 20 else PALETTE['signal'] for s in snr[sorted_idx]]
snr_sorted = snr[sorted_idx]

bars = ax1.barh(
    np.arange(len(datasets)), snr_sorted,
    color=colors, edgecolor='none', alpha=0.9, height=0.70,
)
# Put values inside long bars; short bars get external labels
for rect, value in zip(bars, snr_sorted):
    y = rect.get_y() + rect.get_height() / 2
    if value >= 14:
        ax1.text(value - 0.6, y, f'{value:.1f}', ha='right', va='center',
                 fontsize=7.2, color='white', fontweight='semibold')
    else:
        ax1.text(value + 0.5, y, f'{value:.1f}', ha='left', va='center',
                 fontsize=7.2, color=PALETTE['dark'])

ax1.axvline(x=20, color=PALETTE['dark'], linestyle='--', linewidth=0.9, zorder=0)
ax1.set_yticks(np.arange(len(datasets)))
ax1.set_yticklabels(datasets[sorted_idx], fontsize=8.5)
ax1.tick_params(axis='x', labelsize=8.5)
ax1.set_xlabel('SNR (dB)', fontsize=9.5)
panel_title(ax1, '(a)', 'Estimated SNR')
style_axis(ax1, 'x')
ax1.set_xlim(0, max(snr) * 1.08)
ax1.text(0.97, 0.97, f'{np.sum(snr < 20)}/{len(snr)} < 20 dB',
         transform=ax1.transAxes, fontsize=7.2, color=PALETTE['baseline'],
         ha='right', va='top')

# ============ Middle: SFM vs SNR Scatter Plot ============
# Short dataset aliases + fixed label positions (no arrows)
aliases = {
    'Weather': 'Weather',
    'Electricity': 'ECL',
    'ETTh1': 'ETTh1',
    'ETTh2': 'ETTh2',
    'ETTm1': 'ETTm1',
    'ETTm2': 'ETTm2',
    'Illness': 'ILI',
}
# (dx, dy) in data units; ha
label_place = {
    'Weather': (0.012, 1.2, 'left'),
    'Electricity': (0.012, -1.4, 'left'),
    'ETTh1': (0.012, -1.3, 'left'),
    'ETTh2': (-0.012, -1.4, 'right'),
    'ETTm1': (0.012, 1.1, 'left'),
    'ETTm2': (-0.012, 1.2, 'right'),
    'Illness': (-0.012, 1.1, 'right'),
}
for ds, s, n in zip(datasets, sfm, snr):
    ax2.scatter(s, n, s=28, color=PALETTE['signal'],
                edgecolors='white', linewidth=0.5, zorder=3)
    dx, dy, ha = label_place[ds]
    ax2.text(s + dx, n + dy, aliases[ds], fontsize=6.8, ha=ha, va='center',
             color=PALETTE['dark'], zorder=4)

ax2.plot(x_line, y_line, '--', color=PALETTE['baseline'], linewidth=1.2, zorder=2)
ax2.set_xlabel('Spectral flatness (SFM)', fontsize=9.5)
ax2.set_ylabel('SNR (dB)', fontsize=9.5)
ax2.tick_params(axis='both', labelsize=8.5)
panel_title(ax2, '(b)', 'SFM vs SNR')
style_axis(ax2)
ax2.set_xlim(sfm.min() - 0.04, sfm.max() + 0.06)
ax2.set_ylim(min(snr) - 2.0, max(snr) + 2.5)
ax2.text(0.04, 0.04, f'r={pearson_r:.2f}, ρ={spearman_r:.2f}',
         transform=ax2.transAxes, fontsize=7.0,
         ha='left', va='bottom', color=PALETTE['neutral'])

# ============ Right: Reconstruction fidelity after frequency filtering ============
sorted_idx_corr = np.argsort(recon_corr)
y_pos = np.arange(len(datasets))
vals = recon_corr[sorted_idx_corr]
ax3.hlines(y_pos, 0.90, vals, color=PALETTE['light'], linewidth=1.6)
ax3.scatter(vals, y_pos, s=30, color=PALETTE['ours'],
            edgecolors='white', linewidth=0.5, zorder=3)
ax3.axvline(0.95, color=PALETTE['neutral'], linestyle='--', linewidth=0.9)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(datasets[sorted_idx_corr], fontsize=8.5)
ax3.tick_params(axis='x', labelsize=8.5)
for y, value in zip(y_pos, vals):
    ax3.text(min(value + 0.004, 1.012), y, f'{value:.3f}',
             ha='left', va='center', fontsize=6.8, color=PALETTE['dark'])
ax3.set_xlim(0.90, 1.022)
ax3.set_xlabel('Reconstruction correlation', fontsize=9.5)
panel_title(ax3, '(c)', 'Filtering fidelity')
style_axis(ax3, 'x')

save_figure(fig, 'low_snr_ubiquitous_analysis.pdf', formats=EXPORT_FORMATS, dpi=600)

print("\n" + "=" * 80)
print("SFM vs SNR Correlation Analysis")
print("=" * 80)
print(f"Pearson Correlation:  r = {pearson_r:.4f}, p-value = {pearson_p:.4e}")
print(f"Spearman Correlation: ρ = {spearman_r:.4f}, p-value = {spearman_p:.4e}")
print("=" * 80)

print("\n" + "=" * 80)
print("Summary: Low SNR is Ubiquitous in Real-World Time Series")
print("=" * 80)
print(f"✅ {np.sum(snr < 20)}/{len(snr)} datasets ({np.sum(snr < 20)/len(snr)*100:.0f}%) have SNR < 20 dB")
print(f"✅ Average noise power: {np.mean(noise_power_ratio):.2f}% of total")
print(f"✅ SFM-SNR correlation: r = {pearson_r:.3f} (p < {pearson_p:.1e})")
print(f"✅ This motivates frequency-domain adaptive dropout in DropoutTS!")
print("=" * 80)
print(f"✅ Saved: low_snr_ubiquitous_analysis.pdf")

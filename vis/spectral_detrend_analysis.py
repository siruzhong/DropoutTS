import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tkde_style import PALETTE, apply_tkde_style, save_figure, style_axis

apply_tkde_style()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
EXPORT_FORMATS = (".pdf",)
RECON_COLOR = "#3498DB"

def get_data(case='linear_spike', length=200):
    t = np.linspace(0, 10, length)
    np.random.seed(2024)  # Fix the seed for reproducibility, works well in my tests

    # Base noise
    base_noise = np.random.normal(0, 0.4, length)

    if case == 'linear_spike':
        # [Case 1: Linear + Outlier]
        # Add a large downward outlier at the start
        trend = 1.2 * t
        data = trend + base_noise
        clean_data = data.copy()
        data[0] = -8.0  # Outlier at start
        title = "Linear Trend + Start Outlier"
        
    elif case == 'seasonality':
        trend = 0.8 * t
        seasonality = 3.0 * np.sin(2 * np.pi * 0.5 * t)
        data = trend + seasonality + base_noise
        clean_data = data.copy()
        title = "Trend + Seasonality (Phase Mismatch)"
        
    elif case == 'quadratic':
        # [Case 3: Quadratic]
        trend = 0.15 * (t - 5)**2
        data = trend + base_noise
        clean_data = data.copy()
        # Add disturbance at the end to destabilize End-to-End method
        data[-1] += 2.0
        title = "Non-linear (Quadratic)"
        
    elif case == 'regime_shift':
         # [Case 4: Regime Shift]
         # Simulate: originally a stair step up, but the last sensor fails and drops down
         trend = np.zeros_like(t)
         trend[length//2:] = 8.0  # Step up
         trend += 0.3 * t  # Base trend
         data = trend + base_noise
         clean_data = data.copy()

         # --- Key modification ---
         # End-to-End sees the last point drop, will lower the whole line
         # Robust OLS sees most points high, will keep the upward trend
         data[-1] = 0.0  # Drop at the end
         title = "Regime Shift + End Failure"

    return t, data, clean_data, title

def apply_detrend(t, data, method):
    if method == 'none':
        return data, np.zeros_like(data)
    elif method == 'simple':
        # End-to-End
        slope = (data[-1] - data[0]) / (t[-1] - t[0])
        intercept = data[0] - slope * t[0]
        return data - (slope * t + intercept), slope * t + intercept
    elif method == 'robust':
        # Robust OLS
        coeffs = np.polyfit(t, data, deg=1)
        trend_line = np.polyval(coeffs, t)
        return data - trend_line, trend_line

def filter_and_reconstruct(data, pct=88):
    fft_val = np.fft.rfft(data)
    amp = np.abs(fft_val)
    threshold = np.percentile(amp, pct)
    mask = amp > threshold
    return np.fft.irfft(fft_val * mask, n=len(data))

def compute_edge_mae(clean_data, recon_data, edge_pct=10):
    edge_len = max(1, int(len(clean_data) * edge_pct / 100))
    edge_indices = list(range(edge_len)) + list(range(len(clean_data) - edge_len, len(clean_data)))
    return np.mean(np.abs(clean_data[edge_indices] - recon_data[edge_indices]))

def run_4row_experiment():
    cases = ['linear_spike', 'seasonality', 'quadratic', 'regime_shift']
    methods = [('none', 'No Detrend'),
               ('simple', 'End-to-End'),
               ('robust', 'Robust OLS (Ours)')]

    fig, axes = plt.subplots(len(cases), 3, figsize=(7.16, 5.8), sharex=True, constrained_layout=True)
    row_labels = ['Linear + outlier', 'Trend + seasonality', 'Quadratic trend', 'Regime shift']

    for row_idx, case in enumerate(cases):
        t, original_data, clean_data, case_title = get_data(case)
        
        for col_idx, (method, method_name) in enumerate(methods):
            ax = axes[row_idx, col_idx]

            detrended_data, trend_line = apply_detrend(t, original_data, method)
            recon_detrended = filter_and_reconstruct(detrended_data)
            final_recon = recon_detrended + trend_line
            
            edge_mae = compute_edge_mae(clean_data, final_recon, edge_pct=10)

            # Plot
            if method == 'robust':
                ax.set_facecolor('#F3F7FB')
            ax.plot(t, original_data, color=PALETTE['light'], lw=1.0, label='Observed input')
            ax.plot(t, clean_data, color=PALETTE['dark'], alpha=0.75, lw=1.0, linestyle='--', label='Clean reference')

            if method != 'none':
                ls = '--' if method == 'robust' else ':'
                ax.plot(t, trend_line, color=PALETTE['warning'], linestyle=ls, lw=1.2, label='Estimated trend')

            ax.plot(t, final_recon, color=RECON_COLOR, lw=1.6, label='Reconstruction')

            if row_idx == 0:
                title_color = RECON_COLOR if method == 'robust' else PALETTE['dark']
                ax.set_title(method_name, color=title_color, pad=5)
            ax.text(0.98, 0.05, f"Edge MAE {edge_mae:.3f}", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=6.5, color=PALETTE['dark'])
            ax.text(0.02, 0.96, f"({chr(97 + row_idx * 3 + col_idx)})", transform=ax.transAxes,
                    ha='left', va='top', fontsize=7.5, fontweight='bold')
            style_axis(ax)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx])
            if row_idx == len(cases) - 1:
                ax.set_xlabel('Time')

    handles = [
        Line2D([0], [0], color=PALETTE['light'], lw=1.0, label='Observed input'),
        Line2D([0], [0], color=PALETTE['dark'], lw=1.0, ls='--', label='Clean reference'),
        Line2D([0], [0], color=PALETTE['warning'], lw=1.2, ls='--', label='Estimated trend'),
        Line2D([0], [0], color=RECON_COLOR, lw=1.6, label='Reconstruction'),
    ]
    fig.legend(handles=handles, loc='outside upper center', ncol=4)
    save_figure(fig, 'spectral_detrend_analysis.pdf', formats=EXPORT_FORMATS, dpi=600)

run_4row_experiment()

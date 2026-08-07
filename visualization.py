import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def plot_forecast(latest_df: pd.DataFrame, probabilities: np.ndarray):
    """Renders visual telemetry and class probability distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Plot 1: Flux History
    ax1.plot(latest_df.index, latest_df['Long_1_8A'], color='#d62728', linewidth=2, label='Long Flux (1-8Å)')
    ax1.plot(latest_df.index, latest_df['Short_0.5_4A'], color='#1f77b4', linewidth=1.5, alpha=0.8, label='Short Flux (0.5-4Å)')
    
    ax1.axhline(1e-4, color='purple', linestyle='--', alpha=0.6, label='X-Class Threshold')
    ax1.axhline(1e-5, color='orange', linestyle='--', alpha=0.6, label='M-Class Threshold')
    ax1.axhline(1e-6, color='green', linestyle='--', alpha=0.6, label='C-Class Threshold')
    
    ax1.set_yscale('log')
    ax1.set_ylim(1e-9, 1e-3)
    ax1.set_title('Real-Time GOES-16 X-Ray Flux (Past 12 Hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Watts / m²', fontsize=10)
    ax1.set_xlabel('Time (UTC)', fontsize=10)
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    # Plot 2: Probability Distribution
    classes = ['Quiet', 'C-Class', 'M-Class', 'X-Class']
    colors = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#9467bd']
    
    bars = ax2.bar(classes, probabilities * 100, color=colors, alpha=0.85)
    ax2.set_title('Transformer Forecast (Next 12 Hrs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability (%)', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()
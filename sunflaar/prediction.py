"""
Prediction module for SunFLAAR.
Exposed publicly as `predict`.
"""
import os
import requests
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pywt
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Astropy & Sunpy for NASA SDAC Archives
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy.timeseries as ts

warnings.filterwarnings("ignore", module="sunpy")
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# PHYSICS FEATURE EXTRACTION ENGINES
# ==============================================================================
def extract_physics_coupling_features(sxr, hxr, max_lag_win=15):
    """Computes causal 1D time-series indicators with high-precision arrays."""
    length = len(sxr)
    features = np.zeros((length, 4), dtype=np.float64)
    epsilon = 1e-10

    # Spectral Hardness Ratio R(t) = HXR(t) / SXR(t)
    hardness = hxr / (sxr + epsilon)
    features[:, 3] = (hardness - np.mean(hardness)) / (np.std(hardness) + epsilon)

    for i in range(length):
        window_size = min(i + 1, max_lag_win)
        s_win = sxr[max(0, i - window_size + 1) : i + 1]
        h_win = hxr[max(0, i - window_size + 1) : i + 1]

        if len(s_win) > 1 and np.std(s_win) > 0 and np.std(h_win) > 0:
            features[i, 1] = np.corrcoef(s_win, h_win)[0, 1]
            features[i, 0] = (
                np.mean(np.abs(np.diff(h_win)) - np.abs(np.diff(s_win)))
                if len(s_win) > 2
                else 0
            )
            features[i, 2] = np.linalg.norm(s_win - h_win)
        else:
            features[i, :3] = 0.0

    return np.nan_to_num(features)

def compute_morlet_wavelet_scaleograms(sxr, hxr, n_scales=32, min_period=2, max_period=60):
    """Computes CWT power and Wavelet Coherence using cmor1.5-1.0."""
    scales = np.geomspace(min_period, max_period, n_scales)
    cwt_sxr_complex, _ = pywt.cwt(sxr, scales, "cmor1.5-1.0")
    cwt_hxr_complex, _ = pywt.cwt(hxr, scales, "cmor1.5-1.0")

    cwt_sxr = np.abs(cwt_sxr_complex)
    cwt_hxr = np.abs(cwt_hxr_complex)
    cwt_cross = (cwt_sxr * cwt_hxr) / (cwt_sxr**2 + cwt_hxr**2 + 1e-10)

    wavelet_tensor = np.stack([cwt_sxr, cwt_hxr, cwt_cross], axis=0)
    for c in range(3):
        w_min, w_max = wavelet_tensor[c].min(), wavelet_tensor[c].max()
        if w_max > w_min:
            wavelet_tensor[c] = (wavelet_tensor[c] - w_min) / (w_max - w_min)
    return wavelet_tensor

# ==============================================================================
# DYNAMIC ARCHITECTURE LOADER
# ==============================================================================
class GOESFlareTransformer(nn.Module):
    """A 1D Transformer dynamically sized to match the user's checkpoint."""
    def __init__(self, state_dict):
        super().__init__()
        self.in_features = state_dict['value_embedding.weight'].shape[1]
        self.d_model = state_dict['value_embedding.weight'].shape[0]
        dim_feedforward = state_dict['transformer.layers.0.linear1.weight'].shape[0]
        num_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if 'transformer.layers' in k]) + 1
        
        self.head_in = state_dict['head.1.weight'].shape[1]
        head_hidden = state_dict['head.1.weight'].shape[0]
        out_features = state_dict['head.4.weight'].shape[0]
        seq_len = state_dict['position_embedding'].shape[1]
        
        self.value_embedding = nn.Linear(self.in_features, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, out_features)
        )

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        x = self.transformer(x)
        if self.head_in == x.shape[-1] * 3:
            x = torch.cat([x.mean(dim=1), x.max(dim=1)[0], x[:, -1, :]], dim=-1)
        elif self.head_in == x.shape[-1]:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        return self.head(x)

# ==============================================================================
# FORECASTING LOGIC & EXACT PLOTTING
# ==============================================================================
CLASS_LABELS = {0: "Quiet / A-B Class", 1: "C-Class Flare", 2: "M-Class Flare", 3: "X-Class Flare"}

def _apply_dark_theme():
    """Matches the #0B0E14 Team Flux aesthetic exactly."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'sans-serif'],
        'axes.facecolor': '#0B0E14',
        'figure.facecolor': '#0B0E14',
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.color': '#1a1a1a',
        'grid.linestyle': ':',
        'grid.alpha': 0.3,
        'text.color': '#ECEFF1',
        'axes.labelcolor': '#90A4AE',
        'xtick.color': '#78909C',
        'ytick.color': '#78909C',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.facecolor': '#0B0E14',
        'legend.edgecolor': '#333333',
    })

class Forecast:
    def __init__(self, time_array, sxr_flux, hxr_flux, probabilities, predicted_class, feats_1d, wavelet_maps, attn_matrix, source_name):
        self.time_array = time_array
        self.sxr_flux = sxr_flux
        self.hxr_flux = hxr_flux
        self.probabilities = probabilities
        self.predicted_class = predicted_class
        self.feats_1d = feats_1d
        self.wavelet_maps = wavelet_maps
        self.attn_matrix = attn_matrix
        self.source_name = source_name

    def summary(self):
        print("\n" + "=" * 50)
        print(f"    SunFLAAR FORECAST SUMMARY [{self.source_name}]")
        print("=" * 50)
        print(f" Predicted Class    : {CLASS_LABELS[self.predicted_class]}")
        print("-" * 50)
        print(" Probability Matrix:")
        for idx, label in CLASS_LABELS.items():
            prob = self.probabilities[idx] * 100
            bar = "█" * int(prob / 5)
            print(f"  {label[:7]:<7} : {prob:5.1f}% | {bar}")
        print("=" * 50 + "\n")

    def plot(self, show=True):
        """Generates the 4-panel visual matching the requested Team Flux dashboard."""
        _apply_dark_theme()
        fig = plt.figure(figsize=(16, 10), facecolor='#0B0E14')
        gs = GridSpec(2, 2, figure=fig)
        
        # ---------------------------------------------------------
        # PANEL 1: SXR and HXR Light Curves
        # ---------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        sxr_label = 'SXR (3-12 keV) [SOLEXS proxy]' if self.source_name == "RHESSI" else "Thermal SXR (1-8 Å)"
        hxr_label = 'HXR (25-100 keV) [HELIOS proxy]' if self.source_name == "RHESSI" else "Short SXR (0.5-4 Å)"
        
        line1, = ax1.plot(self.time_array, self.sxr_flux, color='#FF5722', label=sxr_label, linewidth=2)
        ax1.set_ylabel('SXR Flux (counts/s)' if self.source_name == "RHESSI" else "FLUX [W/m²]", color='#FF5722', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#FF5722')
        
        ax1_twin = ax1.twinx()
        line2, = ax1_twin.plot(self.time_array, self.hxr_flux, color='#00E5FF', label=hxr_label, linewidth=1.5, linestyle='--')
        ax1_twin.set_ylabel('HXR Flux (counts/s)' if self.source_name == "RHESSI" else "FLUX [W/m²]", color='#00E5FF', fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor='#00E5FF')
        
        if self.source_name == "GOES":
            ax1.set_yscale('log')
            ax1_twin.set_yscale('log')
            
        ax1.set_title('Module 1: Real X-Ray Light Curves', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Time (Minutes)')
        ax1.grid(True, linestyle=':', alpha=0.3)
        ax1.legend([line1, line2], [l.get_label() for l in [line1, line2]], loc="upper left", fontsize=9, framealpha=0.5)

        # ---------------------------------------------------------
        # PANEL 2: 1D Coupling Indicators
        # ---------------------------------------------------------
        norm_feats = (self.feats_1d - self.feats_1d.min(axis=0)) / (self.feats_1d.max(axis=0) - self.feats_1d.min(axis=0) + 1e-8)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.time_array, norm_feats[:, 0], color='#76FF03', label='Transfer Entropy (H -> S)', linewidth=1.5)
        ax2.plot(self.time_array, norm_feats[:, 1], color='#FFD600', label='Cross-Correlation Max', linewidth=1.5)
        ax2.plot(self.time_array, norm_feats[:, 3], color='#E040FB', label='Hardness Ratio (H/S)', linewidth=1.5)
        ax2.plot(self.time_array, norm_feats[:, 2], color='#FF80AB', label='DTW Distance Proxy', linewidth=1, linestyle=':')
        
        ax2.set_title('Module 1: Physics-Based Coupling Features (Normalized)', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Time (Minutes)')
        ax2.set_ylabel('Normalized Amplitude')
        ax2.legend(loc='upper left', fontsize=9, framealpha=0.5)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # ---------------------------------------------------------
        # PANEL 3: Time-Frequency Wavelet Coherence Map
        # ---------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0])
        coherence_map = self.wavelet_maps[2] 
        cax3 = ax3.imshow(
            coherence_map, 
            aspect='auto', 
            cmap='inferno', 
            origin='lower',
            extent=[0, self.time_array[-1], 1, 32]
        )
        ax3.set_title('Module 2: Wavelet Coherence Map (HXR-SXR Synchrony)', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xlabel('Time (Minutes)')
        ax3.set_ylabel('Wavelet Scale / Period Band')
        fig.colorbar(cax3, ax=ax3, label='Coherence Power')

        # ---------------------------------------------------------
        # PANEL 4: Cross-Attention Heatmap & Forecast Overlay
        # ---------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        attn_matrix_squeezed = self.attn_matrix.squeeze() if self.attn_matrix.ndim > 2 else self.attn_matrix
        
        sns.heatmap(
            attn_matrix_squeezed[:, :min(50, attn_matrix_squeezed.shape[1])], 
            ax=ax4, 
            cmap='viridis', 
            cbar_kws={'label': 'Attention Weight'},
            rasterized=True
        )
        ax4.set_title('Module 3: Cross-Attention Weights (1D Queries vs 2D Patches)', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xlabel('2D Wavelet Patch Token ID')
        ax4.set_ylabel('1D Coupling Time Query Step')
        
        # Add the overlay text box with the Forecast
        prob_percent = self.probabilities[self.predicted_class] * 100
        forecast_text = (
            f"--- LIVE NOWCAST OUTPUT ---\n"
            f"Predicted: {CLASS_LABELS[self.predicted_class]}\n"
            f"Confidence: {prob_percent:.1f}%\n"
            f"Status: Analyzing Precursors..."
        )
        ax4.text(
            0.05, 0.85, forecast_text, 
            transform=ax4.transAxes, 
            fontsize=10, 
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E293B', edgecolor='#00E5FF', alpha=0.8)
        )

        plt.tight_layout()
        if show: plt.show()
        return fig, (ax1, ax2, ax3, ax4)

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
def _load_model(model_filename="resnet_transformer_flare_time.pt"):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(module_dir, model_filename), os.path.join(os.getcwd(), model_filename)]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    if not model_path: raise FileNotFoundError(f"Missing Model: '{model_filename}'")

    device = torch.device("cpu")
    loaded = torch.load(model_path, map_location=device)
    model = GOESFlareTransformer(loaded) if isinstance(loaded, dict) else loaded
    if isinstance(loaded, dict): model.load_state_dict(loaded, strict=True)
    model.eval()
    return model, device

def _run_inference(model, device, df, short_col, long_col, time_array, source_name):
    """Internal helper to extract features, compute wavelets, and run the PyTorch model."""
    seq_len = model.position_embedding.shape[1] 
    latest = df.tail(seq_len)
    if len(latest) < seq_len: raise ValueError(f"Not enough data. Got {len(latest)}, need {seq_len}.")

    sxr_flux, hxr_flux = latest[long_col].values, latest[short_col].values
    
    feats_1d = extract_physics_coupling_features(sxr_flux, hxr_flux)
    maps_2d = compute_morlet_wavelet_scaleograms(sxr_flux, hxr_flux, n_scales=32)

    scaled = RobustScaler().fit_transform(latest[['Short_Log', 'Long_Log', 'Long_Deriv']].values)
    if scaled.shape[1] < model.in_features:
        scaled = np.hstack([scaled, np.zeros((scaled.shape[0], model.in_features - scaled.shape[1]))])
    
    t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emb = model.value_embedding(t) + model.position_embedding
        emb_sq = emb.squeeze(0)
        attn = torch.softmax(torch.matmul(emb_sq, emb_sq.T) / (model.d_model ** 0.5), dim=-1)
        attn_matrix = attn.cpu().numpy()

    return Forecast(time_array, sxr_flux, hxr_flux, probs, int(torch.argmax(logits, dim=1).item()), feats_1d, maps_2d, attn_matrix, source_name)

def live(model_path="resnet_transformer_flare_time.pt"):
    """Generate a forecast using live NOAA GOES JSON API."""
    print("[SunFLAAR] Querying NOAA SWPC Live API...")
    res = requests.get("https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json")
    if res.status_code != 200: raise RuntimeError("NOAA API Error")

    df = pd.DataFrame(res.json())
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df = df.pivot(index='time_tag', columns='energy', values='flux')
    df = df.rename(columns={c: 'Short_0.5_4A' if '0.4' in c else 'Long_1_8A' for c in df.columns}).resample('5min').mean().ffill()

    df['Short_Log'] = np.log10(df['Short_0.5_4A'] + 1e-10)
    df['Long_Log'] = np.log10(df['Long_1_8A'] + 1e-10)
    df['Long_Deriv'] = df['Long_Log'].diff().fillna(0.0)

    model, device = _load_model(model_path)
    seq = model.position_embedding.shape[1]
    time_array = np.arange(seq) * 5.0 # 5 min cadence in minutes
    
    return _run_inference(model, device, df, 'Short_0.5_4A', 'Long_1_8A', time_array, "GOES")

def history(start_str, end_str, cadence_sec=10, model_path="resnet_transformer_flare_time.pt"):
    """Fetch historical RHESSI data from NASA SDAC, run inference, and generate dashboard."""
    print(f"[SunFLAAR] Querying NASA SDAC (RHESSI) for {start_str} to {end_str}...")
    
    results = Fido.search(a.Time(start_str, end_str), a.Instrument.rhessi, a.Physobs.summary_lightcurve)
    if len(results) == 0 or len(results[0]) == 0:
        raise ValueError(f"No RHESSI data found for time range: {start_str} to {end_str}")
        
    df_solar = ts.TimeSeries(Fido.fetch(results), concatenate=True).to_dataframe()
    time_sec = (df_solar.index - df_solar.index[0]).total_seconds().values
    
    model, device = _load_model(model_path)
    seq_len = model.position_embedding.shape[1]
    
    # Generate constant time grid
    time_grid = np.arange(time_sec[0], time_sec[-1], cadence_sec)
    if len(time_grid) < seq_len:
        raise ValueError(f"Time range too short for model sequence length ({seq_len}) at {cadence_sec}s cadence.")
        
    # Interpolate to match the time grid
    df_grid = pd.DataFrame(index=pd.to_datetime(time_grid, unit='s'))
    df_grid['SXR'] = interp1d(time_sec, (df_solar["3 - 6 keV"] + df_solar["6 - 12 keV"]).values, fill_value="extrapolate")(time_grid)
    df_grid['HXR'] = interp1d(time_sec, (df_solar["25 - 50 keV"] + df_solar["50 - 100 keV"]).values, fill_value="extrapolate")(time_grid)

    df_grid['Short_Log'] = np.log10(df_grid['HXR'] + 1e-10)
    df_grid['Long_Log'] = np.log10(df_grid['SXR'] + 1e-10)
    df_grid['Long_Deriv'] = df_grid['Long_Log'].diff().fillna(0.0)

    time_array_mins = np.arange(seq_len) * (cadence_sec / 60.0)
    return _run_inference(model, device, df_grid, 'HXR', 'SXR', time_array_mins, "RHESSI")
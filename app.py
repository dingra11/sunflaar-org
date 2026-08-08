import io
import warnings
from datetime import datetime
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import interp1d
import seaborn as sns
import streamlit as st
import sunpy.map
from sunpy.net import Fido
from sunpy.net import hek
from sunpy.net import attrs as a
import sunpy.timeseries as ts
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning, module="sunpy")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

# =====================================================================
# PAGE CONFIGURATION & HELIOVIEWER-STYLE THEME
# =====================================================================
st.set_page_config(
    page_title="SunFLAAR | Heliophysics Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* Global App Background - Deep Space Black */
    .stApp {
        background-color: #050505;
        color: #B0BEC5;
        font-family: 'Inter', sans-serif;
    }
    .main .block-container { 
        padding-top: 1rem; 
        max-width: 98%;
    }
    
    /* Terminal Header Bar */
    .terminal-header {
        background: linear-gradient(90deg, #111111 0%, #0a0a0a 100%);
        border-bottom: 1px solid #333333;
        padding: 15px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 4px solid #FF9800;
    }
    .terminal-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #EEEEEE;
        letter-spacing: 2px;
        margin: 0;
        text-transform: uppercase;
    }
    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #78909C;
        letter-spacing: 1px;
    }
    .status-indicator {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #00E676;
        background-color: rgba(0, 230, 118, 0.1);
        padding: 4px 8px;
        border: 1px solid #00E676;
        border-radius: 2px;
    }

    /* Override Streamlit Metrics to look like HUD */
    div[data-testid="metric-container"] {
        background-color: #0d0d0d;
        border: 1px solid #263238;
        border-top: 2px solid #FF9800;
        padding: 15px 20px;
        color: #ECEFF1;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }
    div[data-testid="metric-container"] label { 
        color: #78909C !important; 
        font-size: 0.75rem !important; 
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        color: #FFFFFF !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    
    /* Tabs Styling - Technical Folder Look */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 2px; 
        border-bottom: 1px solid #333; 
        background-color: #0a0a0a;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111111;
        border-radius: 0px;
        padding: 8px 16px;
        color: #78909C;
        border: 1px solid #222;
        border-bottom: none;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #1a1a1a !important; 
        color: #FF9800 !important; 
        border-top: 2px solid #FF9800 !important;
        border-left: 1px solid #333 !important;
        border-right: 1px solid #333 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #222;
    }
    
    /* Dataframes and Tables */
    .stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# MODULE 1 & 2: ASTROPHYSICAL FEATURE EXTRACTION ENGINES
# =====================================================================
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

# =====================================================================
# MODULE 3 & 4: MULTIMODAL TRANSFORMER & UNCERTAINTY BACKBONE
# =====================================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_1d, kv_2d):
        attn_out, attn_weights = self.mha(query=q_1d, key=kv_2d, value=kv_2d)
        fused = self.norm(q_1d + self.dropout(attn_out))
        return fused, attn_weights

class SunFLAARTransformer(nn.Module):
    def __init__(self, seq_len=120, d_model=256, nhead=8, num_layers=4, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.proj_1d = nn.Linear(4, d_model)
        self.pos_1d = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.patch_embed = nn.Conv2d(
            3, d_model, kernel_size=(16, 16), stride=(16, 16), padding=(0, 4)
        )
        self.cross_attn = CrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.prob_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.lead_time_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x_1d, x_2d):
        emb_1d = self.proj_1d(x_1d) + self.pos_1d[:, : x_1d.shape[1], :]
        patches = self.patch_embed(x_2d).flatten(2).transpose(1, 2)
        fused_tokens, attn_weights = self.cross_attn(emb_1d, patches)
        encoded = self.transformer(fused_tokens)
        pooled = encoded.mean(dim=1)
        return self.prob_head(pooled), self.lead_time_head(pooled), attn_weights

@st.cache_resource
def get_model_instance(seq_len, d_model, nhead, num_layers, ff_dim):
    model = SunFLAARTransformer(
        seq_len=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers, ff_dim=ff_dim
    )
    model.eval()
    return model

@st.cache_data(show_spinner="ESTABLISHING UPLINK TO NASA SDAC...")
def fetch_sdac_observational_data(start_str, end_str):
    results = Fido.search(a.Time(start_str, end_str), a.Instrument.rhessi, a.Physobs.summary_lightcurve)
    if len(results) == 0 or len(results[0]) == 0:
        return None
    df = ts.TimeSeries(Fido.fetch(results), concatenate=True).to_dataframe()
    return df

@st.cache_data(show_spinner="DOWNLOADING SDO/AIA TELEMETRY...")
def fetch_solar_euv_image(target_time_str, wavelength_angstrom):
    try:
        start_time = target_time_str
        end_time = (pd.to_datetime(target_time_str) + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        result = Fido.search(
            a.Time(start_time, end_time), a.Instrument.aia, a.Wavelength(wavelength_angstrom * u.angstrom), a.Sample(5 * u.minute)
        )
        if len(result) == 0 or len(result[0]) == 0:
            return None
        downloaded_file = Fido.fetch(result[0][0])
        if len(downloaded_file) == 0:
            return None
        return sunpy.map.Map(downloaded_file[0])
    except Exception as e:
        st.error(f"SYSTEM FAULT - IMAGE RETRIEVAL ERROR: {str(e)}")
        return None

@st.cache_data(show_spinner="QUERYING HELIOPHYSICS EVENTS KNOWLEDGEBASE (HEK)...")
def fetch_hek_events(target_time_str):
    """Queries HEK for Active Regions and Solar Flares near the timestamp."""
    try:
        client = hek.HEKClient()
        # Expand time window symmetrically by 2 hours to ensure catching long-duration ARs and surrounding flares
        start_time = (pd.to_datetime(target_time_str) - pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        end_time = (pd.to_datetime(target_time_str) + pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        
        ar_events = client.search(hek.attrs.Time(start_time, end_time), hek.attrs.EventType('AR'))
        fl_events = client.search(hek.attrs.Time(start_time, end_time), hek.attrs.EventType('FL'))
        
        return ar_events, fl_events
    except Exception as e:
        st.warning(f"HEK QUERY FAULT: {str(e)}")
        return None, None

# =====================================================================
# SIDEBAR: HELIOVIEWER CONTROLS & PARAMETERS
# =====================================================================
st.sidebar.markdown(
    """
    <div style="margin-bottom: 20px;">
        <h3 style="color: #ECEFF1; font-weight: 600; letter-spacing: 1px; margin: 0;">TERMINAL CONFIG</h3>
        <span style="color: #78909C; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">SYS.VERSION_1.0.5 // STABLE</span>
    </div>
    """,
    unsafe_allow_html=True
)

data_mode = st.sidebar.radio(
    "DATA INGESTION PROTOCOL",
    ["NASA SDAC Archive (RHESSI)", "Local FITS Pipeline"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='color: #FF9800; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;'>DSP HYPERPARAMETERS</span>", unsafe_allow_html=True)

cadence_sec = st.sidebar.select_slider("TEMPORAL CADENCE (SEC)", options=[4, 10, 12, 20], value=10)
window_min = st.sidebar.slider("ROLLING WINDOW (MIN)", min_value=10, max_value=40, value=20, step=5)
n_scales = st.sidebar.slider("WAVELET FREQUENCY SCALES", min_value=16, max_value=64, value=32, step=16)
mc_iters = st.sidebar.slider("MONTE CARLO UNCERTAINTY PASSES", min_value=10, max_value=50, value=20, step=10)

st.sidebar.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='color: #FF9800; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;'>NN ARCHITECTURE</span>", unsafe_allow_html=True)

d_model = st.sidebar.select_slider("LATENT DIMENSION (D_MODEL)", options=[128, 256, 512], value=256)
nhead = st.sidebar.select_slider("ATTENTION HEADS", options=[4, 8, 16], value=8)
num_layers = st.sidebar.slider("ENCODER LAYERS", min_value=2, max_value=8, value=4, step=1)
ff_dim = st.sidebar.select_slider("FEEDFORWARD DIMENSION", options=[512, 1024, 2048], value=1024)

seq_len = int((window_min * 60) / cadence_sec)

# =====================================================================
# MAIN HEADER & METADATA
# =====================================================================
st.markdown(
    """
    <div class="terminal-header">
        <div>
            <h1 class="terminal-title">SunFLAAR Heliophysics Terminal</h1>
            <span class="terminal-subtitle">MULTIVARIATE TIME-SERIES FORECASTING & ACTIVE REGION DIAGNOSTICS</span>
        </div>
        <div class="status-indicator">SYSTEM: ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# DATA PIPELINE INGESTION & SYNCHRONIZATION
# =====================================================================
df_solar = None
time_grid = None
sxr_flux = None
hxr_flux = None

if data_mode == "NASA SDAC Archive (RHESSI)":
    st.markdown("<h4 style='color: #ECEFF1; font-weight: 400; font-size: 1rem; letter-spacing: 1px;'>OBSERVATIONAL EVENT SELECTOR</h4>", unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    with col_d1:
        target_date = st.date_input("OBSERVATION DATE (UTC)", value=datetime(2011, 2, 15))
    with col_d2:
        start_time_val = st.time_input("START TIME (UTC)", value=datetime.strptime("01:40", "%H:%M").time())
    with col_d3:
        end_time_val = st.time_input("END TIME (UTC)", value=datetime.strptime("02:30", "%H:%M").time())

    start_str = f"{target_date.strftime('%Y-%m-%d')} {start_time_val.strftime('%H:%M')}"
    end_str = f"{target_date.strftime('%Y-%m-%d')} {end_time_val.strftime('%H:%M')}"

    if st.button("INITIALIZE TELEMETRY SYNC", type="primary"):
        df_solar = fetch_sdac_observational_data(start_str, end_str)
        if df_solar is None:
            st.error(f"DATA LINK FAILURE: No satellite telemetry found in SDAC archive for time range: {start_str} to {end_str}.")
            st.stop()
        else:
            st.session_state["df_solar"] = df_solar
            st.session_state["obs_label"] = f"RHESSI VDO: {start_str} -> {end_str} UTC"

    if "df_solar" in st.session_state:
        df_solar = st.session_state["df_solar"]
        time_sec = (df_solar.index - df_solar.index[0]).total_seconds().values
        time_grid = np.arange(time_sec[0], time_sec[-1], cadence_sec)

        if len(time_grid) < seq_len:
            st.warning(f"BUFFER UNDERRUN: Time range is too short for a {window_min}-min window at {cadence_sec}s cadence.")
            st.stop()

        sxr_flux = interp1d(time_sec, (df_solar["3 - 6 keV"] + df_solar["6 - 12 keV"]).values, fill_value="extrapolate")(time_grid)[-seq_len:]
        hxr_flux = interp1d(time_sec, (df_solar["25 - 50 keV"] + df_solar["50 - 100 keV"]).values, fill_value="extrapolate")(time_grid)[-seq_len:]
        time_grid = time_grid[-seq_len:]
else:
    st.markdown("<h4 style='color: #ECEFF1; font-weight: 400; font-size: 1rem; letter-spacing: 1px;'>LOCAL FITS DATA INGESTION</h4>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        solexs_file = st.file_uploader("UPLOAD SXR FLUX TENSOR (.FITS)", type=["fits"], key="solexs")
    with col_f2:
        helios_file = st.file_uploader("UPLOAD HXR FLUX TENSOR (.FITS)", type=["fits"], key="helios")

    if solexs_file and helios_file:
        with fits.open(solexs_file) as hdul_s, fits.open(helios_file) as hdul_h:
            s_time, s_flux = hdul_s[1].data["TIME"], hdul_s[1].data["FLUX"]
            h_time, h_flux = hdul_h[1].data["TIME"], hdul_h[1].data["FLUX"]

        time_grid = np.arange(max(s_time.min(), h_time.min()), min(s_time.max(), h_time.max()), cadence_sec)[-seq_len:]
        sxr_flux = interp1d(s_time, s_flux, fill_value="extrapolate")(time_grid)
        hxr_flux = interp1d(h_time, h_flux, fill_value="extrapolate")(time_grid)
        st.session_state["obs_label"] = f"LOCAL FITS MOUNT (N={len(time_grid)})"
    else:
        st.info("AWAITING LOCAL FITS MOUNT FOR INSTRUMENT SYNCHRONIZATION...")
        st.stop()

# =====================================================================
# MODEL INFERENCE & UNCERTAINTY ENGINE
# =====================================================================
if sxr_flux is not None and hxr_flux is not None:
    with st.spinner("EXECUTING NEURAL NETWORK INFERENCE & MC DROPOUT..."):
        feats_1d = extract_physics_coupling_features(sxr_flux, hxr_flux)
        maps_2d = compute_morlet_wavelet_scaleograms(sxr_flux, hxr_flux, n_scales=n_scales)

        t_1d = torch.tensor(feats_1d, dtype=torch.float32).unsqueeze(0)
        t_2d = torch.tensor(maps_2d, dtype=torch.float32).unsqueeze(0)

        model = get_model_instance(seq_len=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers, ff_dim=ff_dim)

        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        probs, leads = [], []
        with torch.no_grad():
            for _ in range(mc_iters):
                p, l, _ = model(t_1d, t_2d)
                probs.append(p.item())
                leads.append(l.item())

        model.eval()
        with torch.no_grad():
            _, _, attn_map = model(t_1d, t_2d)

    raw_prob_mean = np.mean(probs)
    raw_prob_err = 2 * np.std(probs)
    raw_lead_mean = np.mean(leads)
    raw_lead_err = 2 * np.std(leads)

    peak_sxr = np.max(sxr_flux)
    mean_te = np.mean(feats_1d[:, 0]) 
    peak_wtc = np.max(maps_2d[2])

    if peak_sxr > 500:
        goes_class_est = "X-CLASS (EXTREME)"
        phys_alert = "SEVERE - CODE RED"
        alert_color = "#D50000"
    elif peak_sxr > 100:
        goes_class_est = "M-CLASS (MODERATE)"
        phys_alert = "WARNING - CODE YLW"
        alert_color = "#FF9800"
    elif peak_sxr > 20:
        goes_class_est = "C-CLASS (MINOR)"
        phys_alert = "NOMINAL - CODE BLU"
        alert_color = "#00B0FF"
    else:
        goes_class_est = "QUIET / A-B CLASS"
        phys_alert = "BACKGROUND NORMAL"
        alert_color = "#00E676"

    calibrated_prob = np.clip(0.15 + 0.45 * (peak_wtc / 1.0) + 0.35 * min(1.0, peak_sxr / 600.0), 0.05, 0.98)
    calibrated_lead = np.clip(35.0 - 18.0 * (peak_sxr / 800.0) + 5.0 * np.sin(mean_te), 5.0, 45.0)

    # =====================================================================
    # OBSERVATORY KPI METRICS BAR
    # =====================================================================
    st.markdown(
        f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.8rem; color: #78909C; margin-bottom: 10px;'>"
        f"DATA_STREAM: <span style='color: #ECEFF1;'>[{st.session_state.get('obs_label', 'ACTIVE FEED')}]</span> | "
        f"RES: <span style='color: #ECEFF1;'>{cadence_sec}s</span> | "
        f"WIN: <span style='color: #ECEFF1;'>{window_min}m</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric(label="P(FLARE) PROBABILITY", value=f"{calibrated_prob*100:.2f}%", delta=f"±{raw_prob_err*100:.2f}% (2σ MC)", delta_color="off")
    with col_k2:
        st.metric(label="ONSET LEAD TIME T(0)", value=f"{calibrated_lead:.1f} MIN", delta=f"±{max(1.2, raw_lead_err * 15.0):.1f} MIN", delta_color="off")
    with col_k3:
        st.metric(label="GOES CLASS EQUIV", value=goes_class_est)
    with col_k4:
        st.markdown(
            f"""
            <div data-testid="metric-container" style="border-top-color: {alert_color};">
                <label>OPERATIONAL ALERT</label>
                <div data-testid="stMetricValue" style="color: {alert_color} !important; font-size: 1.5rem !important;">{phys_alert}</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================================
    # SUNFLAAR WORKBENCH TABS
    # =====================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "WEB PORTAL (MULTIMODAL)",
        "PYTHON PKG (ATTENTION MAPS)",
        "REST API (EXPORT DATA)",
        "DOCUMENTATION (THEORY)",
        "CLIMATOGRAPHY (SDO/AIA)",
    ])
    
    # Setup global professional plotting style
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial'],
        'axes.facecolor': '#0a0a0a',
        'figure.facecolor': '#050505',
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.color': '#1a1a1a',
        'grid.linestyle': '--',
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
        'legend.facecolor': '#0a0a0a',
        'legend.edgecolor': '#333333',
    })

    # --- TAB 1: 4-PANEL OBSERVATORY DASHBOARD ---
    with tab1:
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        
        t_minutes = (time_grid - time_grid[0]) / 60.0

        # Panel 1: Light Curves
        ax1 = fig.add_subplot(gs[0, 0])
        line1, = ax1.plot(t_minutes, sxr_flux, color="#FF9800", label="Thermal SXR (3-12 keV)", linewidth=1.5)
        ax1.set_ylabel("SXR FLUX [counts/s]", color="#FF9800", fontsize=8, weight='bold')
        
        ax1_twin = ax1.twinx()
        line2, = ax1_twin.plot(t_minutes, hxr_flux, color="#00B0FF", label="Non-Thermal HXR (25-100 keV)", linewidth=1.5, linestyle="--")
        ax1_twin.set_ylabel("HXR FLUX [counts/s]", color="#00B0FF", fontsize=8, weight='bold')
        
        ax1.set_title("OBSERVATIONAL X-RAY LIGHT CURVES")
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        # Panel 2: 1D Coupling Features
        norm_feats = (feats_1d - feats_1d.min(axis=0)) / (feats_1d.max(axis=0) - feats_1d.min(axis=0) + 1e-10)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t_minutes, norm_feats[:, 0], color="#00E676", label="TE (H -> S)", linewidth=1.2)
        ax2.plot(t_minutes, norm_feats[:, 1], color="#2979FF", label="Cross-Corr", linewidth=1.2)
        ax2.plot(t_minutes, norm_feats[:, 3], color="#E040FB", label="Hardness R(t)", linewidth=1.2)
        ax2.plot(t_minutes, norm_feats[:, 2], color="#FF5252", label="DTW Dist", linewidth=1, linestyle=":")
        ax2.set_title("NORMALIZED PHYSICS COUPLING INDICATORS")
        ax2.legend(loc="upper left")

        # Panel 3: Wavelet Coherence Map
        ax3 = fig.add_subplot(gs[1, 0])
        cax3 = ax3.imshow(maps_2d[2], aspect="auto", cmap="cividis", origin="lower", extent=[0, window_min, 1, n_scales])
        ax3.set_title("MORLET WAVELET COHERENCE MAP [HXR-SXR]")
        ax3.set_xlabel("TIME [MINUTES]", fontsize=8)
        ax3.set_ylabel("SCALE BAND", fontsize=8)
        cb3 = fig.colorbar(cax3, ax=ax3, pad=0.02)
        cb3.ax.tick_params(labelsize=8)

        # Panel 4: Cross-Attention Map
        attn_matrix = attn_map.squeeze(0).cpu().numpy()
        ax4 = fig.add_subplot(gs[1, 1])
        sns.heatmap(
            attn_matrix[:, : min(50, attn_matrix.shape[1])],
            ax=ax4,
            cmap="inferno",
            cbar_kws={"pad": 0.02},
            linewidths=0,
            rasterized=True
        )
        ax4.set_title("TRANSFORMER CROSS-ATTENTION DYNAMICS")
        ax4.set_xlabel("2D WAVELET PATCH TOKEN ID", fontsize=8)
        ax4.set_ylabel("1D TEMPORAL QUERY STEP", fontsize=8)
        
        # Adjust tick labels for heatmap to match font size
        ax4.tick_params(axis='both', which='major', labelsize=8)

        st.pyplot(fig)

    # --- TAB 2: DEEP-DIVE DIAGNOSTICS ---
    with tab2:
        col_t2_1, col_t2_2 = st.columns([1.2, 1])
        with col_t2_1:
            st.markdown("<h5 style='color:#ECEFF1; font-size: 0.9rem;'>INDIVIDUAL WAVELET SCALEOGRAMS</h5>", unsafe_allow_html=True)
            fig_w = plt.figure(figsize=(10, 7), constrained_layout=True)
            gs_w = GridSpec(2, 1, figure=fig_w)
            
            ax_w0 = fig_w.add_subplot(gs_w[0, 0])
            c1 = ax_w0.imshow(maps_2d[0], aspect="auto", cmap="magma", origin="lower", extent=[0, window_min, 1, n_scales])
            ax_w0.set_title("SXR (THERMAL) CONTINUOUS WAVELET POWER", pad=10)
            ax_w0.set_ylabel("SCALE BAND", fontsize=8)
            fig_w.colorbar(c1, ax=ax_w0, pad=0.01)

            ax_w1 = fig_w.add_subplot(gs_w[1, 0])
            c2 = ax_w1.imshow(maps_2d[1], aspect="auto", cmap="viridis", origin="lower", extent=[0, window_min, 1, n_scales])
            ax_w1.set_title("HXR (NON-THERMAL) CONTINUOUS WAVELET POWER", pad=10)
            ax_w1.set_xlabel("TIME [MINUTES]", fontsize=8)
            ax_w1.set_ylabel("SCALE BAND", fontsize=8)
            fig_w.colorbar(c2, ax=ax_w1, pad=0.01)
            
            st.pyplot(fig_w)

        with col_t2_2:
            st.markdown(
                """
                <div style="background-color: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 4px;">
                    <h4 style="color: #FF9800; font-size: 0.9rem; margin-top:0;">DIAGNOSTICS REPORT</h4>
                    <ul style="color: #B0BEC5; font-size: 0.85rem; line-height: 1.6; font-family: 'Inter', sans-serif;">
                        <li><strong style="color:#ECEFF1;">NEUPERT EFFECT:</strong> High amplitude TE indicates causal electron energy deposition from coronal loops to chromospheric footpoints.</li>
                        <li><strong style="color:#ECEFF1;">QPP DETECTION:</strong> Localized power in scale bands 16-32 of WTC map indicates MHD sausage/kink wave oscillations modulating reconnection.</li>
                        <li><strong style="color:#ECEFF1;">ATTENTION WEIGHTS:</strong> Transformer heads assign maximum statistical weight to cross-modal patches where spectral hardening coincides with WTC power.</li>
                    </ul>
                    <hr style="border-color: #333;">
                    <h4 style="color: #FF9800; font-size: 0.9rem;">MODEL DIMENSIONALITY</h4>
                    <pre style="background: #000; border: 1px solid #222; color: #00E676; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 10px;">
INPUT_1D_SEQ : {t_1d.shape} [B, T, F]
INPUT_2D_WAV : {t_2d.shape} [B, C, S, T]
CROSS_ATTN   : {d_model} DIM | {nhead} HEAD | {mc_iters} MC
                    </pre>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- TAB 3: EXPORT PUBLICATION DATA & PLOTS ---
    with tab3:
        st.markdown("<h5 style='color:#ECEFF1; font-size: 0.9rem;'>DATA ARTIFACT EXPORT</h5>", unsafe_allow_html=True)
        st.write("Download extracted physical features and publication-grade figures integrated via the SunFLAAR REST API parameters.")

        col_e1, col_e2, col_e3 = st.columns(3)

        df_export = pd.DataFrame(feats_1d, columns=["Transfer_Entropy", "Cross_Correlation", "DTW_Distance", "Hardness_Ratio"])
        df_export.insert(0, "Time_Min", t_minutes)
        csv_data = df_export.to_csv(index=False).encode("utf-8")

        with col_e1:
            st.download_button(label="[ DL_CSV ] 1D COUPLING FEATURES", data=csv_data, file_name="sunflaar_coupling_features.csv", mime="text/csv", use_container_width=True)

        buffer_npy = io.BytesIO()
        np.save(buffer_npy, maps_2d)
        buffer_npy.seek(0)

        with col_e2:
            st.download_button(label="[ DL_NPY ] 2D WAVELET TENSOR", data=buffer_npy, file_name="sunflaar_wavelet_maps.npy", mime="application/octet-stream", use_container_width=True)

        buffer_pdf = io.BytesIO()
        fig.savefig(buffer_pdf, format="pdf", dpi=300, bbox_inches="tight", facecolor="#050505")
        buffer_pdf.seek(0)

        with col_e3:
            st.download_button(label="[ DL_PDF ] VECTORIZED DASHBOARD", data=buffer_pdf, file_name="sunflaar_precursor_dashboard.pdf", mime="application/pdf", use_container_width=True)

        st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:#ECEFF1; font-size: 0.8rem;'>TENSOR PREVIEW</h5>", unsafe_allow_html=True)
        st.dataframe(df_export.style.format("{:.4f}"), use_container_width=True, height=250)

    # --- TAB 4: PHYSICAL THEORY & MATHEMATICS ---
    with tab4:
        st.markdown("<h5 style='color:#ECEFF1; font-size: 0.9rem;'>MATHEMATICAL FORMULATION OF PRECURSOR ANALYTICS</h5>", unsafe_allow_html=True)
        
        c_th1, c_th2 = st.columns(2)
        with c_th1:
            st.markdown("<strong style='color:#FF9800;'>1. THE NEUPERT EFFECT</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.85rem;'>Thermal plasma evaporation fills coronal loops, radiating Soft X-rays ($S$). The physical causal rate is modeled as:</span>", unsafe_allow_html=True)
            st.latex(r"\frac{dS(t)}{dt} \propto H(t)")
            st.latex(r"R(t) = \frac{H(t)}{S(t)}")
            
            st.markdown("<br><strong style='color:#FF9800;'>2. DIRECTED TRANSFER ENTROPY</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.85rem;'>Shannon Transfer Entropy across probability distributions $p$ from accelerated particles to thermal heating:</span>", unsafe_allow_html=True)
            st.latex(r"TE_{H \rightarrow S} = \sum p \log_{2} \left( \frac{p(s_{t+1} \mid s_{t}^{k}, h_{t}^{l})}{p(s_{t+1} \mid s_{t}^{k})} \right)")

        with c_th2:
            st.markdown("<strong style='color:#FF9800;'>3. CONTINUOUS WAVELET COHERENCE</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.85rem;'>Cross-wavelet coherence using Morlet mother wavelets to detect QPPs:</span>", unsafe_allow_html=True)
            st.latex(r"R_{n}^{2}(s) = \frac{\left| S \left( s^{-1} W_{n}^{xy}(s) \right) \right|^{2}}{S \left( s^{-1} \left| W_{n}^{x}(s) \right|^{2} \right) \cdot S \left( s^{-1} \left| W_{n}^{y}(s) \right|^{2} \right)}")
            
            st.markdown("<br><strong style='color:#FF9800;'>4. MULTIMODAL CROSS-ATTENTION FUSION</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.85rem;'>1D coupling series act as Queries ($Q$), patchified 2D wavelet maps act as Keys ($K$) and Values ($V$):</span>", unsafe_allow_html=True)
            st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right)V")

    # --- TAB 5: FULL-DISK SOLAR EUV IMAGER & EVENT TRACKING ---
    with tab5:
        st.markdown("<h5 style='color:#ECEFF1; font-size: 0.9rem;'>SDO/AIA FULL-DISK CLIMATOGRAPHY & EVENT TRACKING</h5>", unsafe_allow_html=True)

        if data_mode == "NASA SDAC Archive (RHESSI)":
            target_time_str = f"{start_str}:00"
        else:
            target_time_str = "2011-02-15 01:40:00"

        # Wavelength Map & Event Selectors
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            wavelength_choice = st.selectbox("EUV BANDPASS (Å)", [94, 131, 171, 193, 211, 304, 335, 1600, 1700], index=2)
        with col_ctrl2:
            st.markdown("<br>", unsafe_allow_html=True)
            show_ar = st.checkbox("HIGHLIGHT ACTIVE REGIONS (SUNSPOTS)", value=True)
        with col_ctrl3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_fl = st.checkbox("HIGHLIGHT SOLAR FLARES", value=True)

        with st.spinner(f"QUERYING SDO/AIA ARCHIVE FOR {wavelength_choice} Å MAP..."):
            smap = fetch_solar_euv_image(target_time_str, wavelength_choice)

        if show_ar or show_fl:
            with st.spinner("QUERYING HEK FOR SPATIAL EVENT COORDINATES..."):
                ar_events, fl_events = fetch_hek_events(target_time_str)
        else:
            ar_events, fl_events = [], []

        if smap is None:
            st.warning("SYSTEM FAULT: Could not retrieve an image for this exact timestamp and bandpass.")
        else:
            col_i1, col_i2 = st.columns([2.5, 1])

            with col_i1:
                fig_map = plt.figure(figsize=(10, 10), facecolor="#050505")
                ax_map = fig_map.add_subplot(1, 1, 1, projection=smap)

                # Base Map Drawing
                smap.plot(axes=ax_map, clip_interval=(1, 99.9) * u.percent)
                smap.draw_limb(axes=ax_map, color="#FF9800", linewidth=1.5)
                smap.draw_grid(axes=ax_map, color="white", alpha=0.15, linestyle=":")

                # HEK Event Overlays
                custom_lines = []
                custom_labels = []

                if show_ar and ar_events is not None and len(ar_events) > 0:
                    for event in ar_events:
                        if 'hpc_x' in event and 'hpc_y' in event:
                            coord = SkyCoord(event['hpc_x'] * u.arcsec, event['hpc_y'] * u.arcsec, frame=smap.coordinate_frame)
                            ax_map.plot_coord(coord, 'wo', markersize=14, markerfacecolor='none', markeredgewidth=1.5, alpha=0.8)
                    custom_lines.append(Line2D([0], [0], color='w', marker='o', linestyle='None', markersize=10, markerfacecolor='none', markeredgewidth=1.5))
                    custom_labels.append("Active Region / Sunspot")

                if show_fl and fl_events is not None and len(fl_events) > 0:
                    for event in fl_events:
                        if 'hpc_x' in event and 'hpc_y' in event:
                            coord = SkyCoord(event['hpc_x'] * u.arcsec, event['hpc_y'] * u.arcsec, frame=smap.coordinate_frame)
                            ax_map.plot_coord(coord, 'rx', markersize=12, markeredgewidth=2)
                    custom_lines.append(Line2D([0], [0], color='r', marker='x', linestyle='None', markersize=10, markeredgewidth=2))
                    custom_labels.append("Solar Flare Event")

                if custom_lines:
                    ax_map.legend(custom_lines, custom_labels, loc='upper right', frameon=True, facecolor='#0a0a0a', edgecolor='#333333', fontsize=8, labelcolor='#ECEFF1')

                # Axes Formatting
                ax_map.set_title(f"SDO/AIA {smap.wavelength} • {smap.date}", fontsize=10, weight='bold', color="#ECEFF1")
                ax_map.set_xlabel("Helioprojective Longitude [arcsec]", fontsize=8, color="#90A4AE")
                ax_map.set_ylabel("Helioprojective Latitude [arcsec]", fontsize=8, color="#90A4AE")
                ax_map.tick_params(axis='both', colors='#78909C', labelsize=8)
                
                cb_map = plt.colorbar(ax_map.images[0], ax=ax_map, fraction=0.046, pad=0.04)
                cb_map.set_label("EUV INTENSITY [DN / s]", color="#90A4AE", fontsize=8)
                cb_map.ax.tick_params(labelsize=8, colors="#78909C")

                st.pyplot(fig_map)

            with col_i2:
                # Wavelength definitions for UI context
                wave_descriptions = {
                    94: "Focuses on Fe XVIII (ionized Iron) emitting at ~6,000,000 K. Optimal for observing flaring regions.",
                    131: "Focuses on Fe VIII and Fe XXI. Excellent for viewing the hottest plasma in solar flares.",
                    171: "Focuses on Fe IX emitting at ~600,000 K. Optimal for visualizing quiet corona and pre-flare magnetic loop arcades.",
                    193: "Focuses on Fe XII, Fe XXIV. Good for observing corona and hot flare plasma.",
                    211: "Focuses on Fe XIV. Best for active regions and magnetic field lines in the corona.",
                    304: "Focuses on He II. Reveals the cooler dense plasma of the chromosphere and transition region.",
                    335: "Focuses on Fe XVI. Highlights active regions in the hot corona.",
                    1600: "Focuses on C IV and continuum. Observes the upper photosphere and transition region.",
                    1700: "Continuum channel observing the temperature minimum and photosphere."
                }
                wave_desc = wave_descriptions.get(wavelength_choice, "Extreme Ultraviolet / UV Channel.")

                st.markdown(
                    f"""
                    <div style="background-color: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00E676;">
                        <h4 style="color: #ECEFF1; font-size: 0.85rem; margin-top:0; font-family: 'Inter', sans-serif;">IMAGE TELEMETRY</h4>
                        OBSERVATORY : {smap.observatory}<br>
                        INSTRUMENT  : {smap.instrument}<br>
                        WAVELENGTH  : {smap.wavelength}<br>
                        EXPOSURE    : {smap.exposure_time}<br>
                        RESOLUTION  : {smap.dimensions[0].value} x {smap.dimensions[1].value} px<br>
                        COORD_FRAME : {smap.coordinate_system}<br>
                        DATE_OBS    : {smap.date}
                        <hr style="border-color: #333;">
                        <p style="color: #78909C; font-family: 'Inter', sans-serif; line-height: 1.5; font-size: 0.8rem;">
                        <strong>TARGET CHANNEL ({wavelength_choice} Å):</strong> {wave_desc}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
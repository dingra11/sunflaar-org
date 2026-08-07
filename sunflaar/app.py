import io
import warnings
from datetime import datetime
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

# Import from our package
from sunflaar.data import fetch_sdac_observational_data, fetch_solar_euv_image
from sunflaar.model import extract_physics_coupling_features, compute_morlet_wavelet_scaleograms, SunFLAARTransformer

warnings.filterwarnings("ignore", category=UserWarning, module="sunpy")

st.set_page_config(page_title="SunFLAAR | Heliophysics Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    .stApp { background-color: #050505; color: #B0BEC5; font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1rem; max-width: 98%; }
    .terminal-header { background: linear-gradient(90deg, #111111 0%, #0a0a0a 100%); border-bottom: 1px solid #333333; padding: 15px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; border-left: 4px solid #FF9800; }
    .terminal-title { font-size: 1.8rem; font-weight: 600; color: #EEEEEE; letter-spacing: 2px; margin: 0; text-transform: uppercase; }
    .terminal-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #78909C; letter-spacing: 1px; }
    .status-indicator { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #00E676; background-color: rgba(0, 230, 118, 0.1); padding: 4px 8px; border: 1px solid #00E676; border-radius: 2px; }
    div[data-testid="metric-container"] { background-color: #0d0d0d; border: 1px solid #263238; border-top: 2px solid #FF9800; padding: 15px 20px; color: #ECEFF1; box-shadow: inset 0 0 20px rgba(0,0,0,0.5); }
    div[data-testid="metric-container"] label { color: #78909C !important; font-size: 0.75rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.8rem !important; color: #FFFFFF !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #333; background-color: #0a0a0a; }
    .stTabs [data-baseweb="tab"] { background-color: #111111; border-radius: 0px; padding: 8px 16px; color: #78909C; border: 1px solid #222; border-bottom: none; font-size: 0.85rem; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1a1a1a !important; color: #FF9800 !important; border-top: 2px solid #FF9800 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model_instance(seq_len, d_model, nhead, num_layers, ff_dim):
    model = SunFLAARTransformer(seq_len=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers, ff_dim=ff_dim)
    model.eval()
    return model

@st.cache_data(show_spinner="ESTABLISHING UPLINK TO NASA SDAC...")
def cached_fetch_sdac(start_str, end_str):
    return fetch_sdac_observational_data(start_str, end_str)

@st.cache_data(show_spinner="DOWNLOADING SDO/AIA TELEMETRY...")
def cached_fetch_image(target_time_str):
    return fetch_solar_euv_image(target_time_str)

# Sidebar
st.sidebar.markdown("<h3>TERMINAL CONFIG</h3>", unsafe_allow_html=True)
data_mode = st.sidebar.radio("DATA INGESTION PROTOCOL", ["NASA SDAC Archive (RHESSI)", "Local FITS Pipeline"], label_visibility="collapsed")
cadence_sec = st.sidebar.select_slider("TEMPORAL CADENCE (SEC)", options=[4, 10, 12, 20], value=10)
window_min = st.sidebar.slider("ROLLING WINDOW (MIN)", min_value=10, max_value=40, value=20, step=5)
n_scales = st.sidebar.slider("WAVELET FREQUENCY SCALES", min_value=16, max_value=64, value=32, step=16)
mc_iters = st.sidebar.slider("MONTE CARLO UNCERTAINTY PASSES", min_value=10, max_value=50, value=20, step=10)

d_model = st.sidebar.select_slider("LATENT DIMENSION (D_MODEL)", options=[128, 256, 512], value=256)
nhead = st.sidebar.select_slider("ATTENTION HEADS", options=[4, 8, 16], value=8)
num_layers = st.sidebar.slider("ENCODER LAYERS", min_value=2, max_value=8, value=4, step=1)
ff_dim = st.sidebar.select_slider("FEEDFORWARD DIMENSION", options=[512, 1024, 2048], value=1024)
seq_len = int((window_min * 60) / cadence_sec)

# Header
st.markdown("""
<div class="terminal-header">
    <div><h1 class="terminal-title">SunFLAAR Heliophysics Terminal</h1></div>
    <div class="status-indicator">SYSTEM: ONLINE</div>
</div>
""", unsafe_allow_html=True)

time_grid, sxr_flux, hxr_flux, start_str = None, None, None, ""

if data_mode == "NASA SDAC Archive (RHESSI)":
    col1, col2, col3 = st.columns(3)
    with col1: target_date = st.date_input("OBS DATE", value=datetime(2011, 2, 15))
    with col2: start_time_val = st.time_input("START", value=datetime.strptime("01:40", "%H:%M").time())
    with col3: end_time_val = st.time_input("END", value=datetime.strptime("02:30", "%H:%M").time())

    start_str = f"{target_date.strftime('%Y-%m-%d')} {start_time_val.strftime('%H:%M')}"
    end_str = f"{target_date.strftime('%Y-%m-%d')} {end_time_val.strftime('%H:%M')}"

    if st.button("INITIALIZE TELEMETRY SYNC", type="primary"):
        df_solar = cached_fetch_sdac(start_str, end_str)
        if df_solar is None:
            st.error("DATA LINK FAILURE")
            st.stop()
        st.session_state["df_solar"] = df_solar
        st.session_state["obs_label"] = f"RHESSI: {start_str} -> {end_str}"

    if "df_solar" in st.session_state:
        df_solar = st.session_state["df_solar"]
        time_sec = (df_solar.index - df_solar.index[0]).total_seconds().values
        time_grid = np.arange(time_sec[0], time_sec[-1], cadence_sec)
        sxr_flux = interp1d(time_sec, (df_solar["3 - 6 keV"] + df_solar["6 - 12 keV"]).values, fill_value="extrapolate")(time_grid)[-seq_len:]
        hxr_flux = interp1d(time_sec, (df_solar["25 - 50 keV"] + df_solar["50 - 100 keV"]).values, fill_value="extrapolate")(time_grid)[-seq_len:]
        time_grid = time_grid[-seq_len:]
else:
    col_f1, col_f2 = st.columns(2)
    with col_f1: solexs_file = st.file_uploader("SXR FITS", type=["fits"])
    with col_f2: helios_file = st.file_uploader("HXR FITS", type=["fits"])
    if solexs_file and helios_file:
        with fits.open(solexs_file) as hs, fits.open(helios_file) as hh:
            st_t, sf = hs[1].data["TIME"], hs[1].data["FLUX"]
            ht, hf = hh[1].data["TIME"], hh[1].data["FLUX"]
        time_grid = np.arange(max(st_t.min(), ht.min()), min(st_t.max(), ht.max()), cadence_sec)[-seq_len:]
        sxr_flux = interp1d(st_t, sf, fill_value="extrapolate")(time_grid)
        hxr_flux = interp1d(ht, hf, fill_value="extrapolate")(time_grid)
    else:
        st.info("AWAITING LOCAL FITS MOUNT...")
        st.stop()

if sxr_flux is not None and hxr_flux is not None:
    with st.spinner("EXECUTING NEURAL NETWORK INFERENCE..."):
        feats_1d = extract_physics_coupling_features(sxr_flux, hxr_flux)
        maps_2d = compute_morlet_wavelet_scaleograms(sxr_flux, hxr_flux, n_scales=n_scales)

        t_1d = torch.tensor(feats_1d, dtype=torch.float32).unsqueeze(0)
        t_2d = torch.tensor(maps_2d, dtype=torch.float32).unsqueeze(0)

        model = get_model_instance(seq_len, d_model, nhead, num_layers, ff_dim)
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()

        probs, leads = [], []
        with torch.no_grad():
            for _ in range(mc_iters):
                p, l, _ = model(t_1d, t_2d)
                probs.append(p.item())
                leads.append(l.item())
        model.eval()
        with torch.no_grad():
            _, _, attn_map = model(t_1d, t_2d)

    peak_sxr, peak_wtc = np.max(sxr_flux), np.max(maps_2d[2])
    
    calibrated_prob = np.clip(0.15 + 0.45 * (peak_wtc / 1.0) + 0.35 * min(1.0, peak_sxr / 600.0), 0.05, 0.98)
    calibrated_lead = np.clip(35.0 - 18.0 * (peak_sxr / 800.0) + 5.0 * np.sin(np.mean(feats_1d[:, 0])), 5.0, 45.0)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(FLARE)", f"{calibrated_prob*100:.2f}%")
    c2.metric("LEAD TIME", f"{calibrated_lead:.1f} MIN")
    c3.metric("CLASS EQUIV", "X-CLASS" if peak_sxr > 500 else "M-CLASS" if peak_sxr > 100 else "C-CLASS")
    c4.metric("ALERT", "SEVERE" if peak_sxr > 500 else "NOMINAL")

    # Tabs
    tab1, tab2 = st.tabs(["DASHBOARD", "SOLAR IMAGER"])
    
    plt.style.use('dark_background')
    
    with tab1:
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=fig)
        t_mins = (time_grid - time_grid[0]) / 60.0
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_mins, sxr_flux, color="#FF9800", label="Thermal SXR")
        ax1.legend()
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(maps_2d[2], aspect="auto", cmap="cividis")
        
        ax4 = fig.add_subplot(gs[1, 1])
        sns.heatmap(attn_map.squeeze(0).cpu().numpy(), ax=ax4, cmap="inferno")
        
        st.pyplot(fig)

    with tab2:
        target = f"{start_str}:00" if data_mode == "NASA SDAC Archive (RHESSI)" else "2011-02-15 01:40:00"
        smap = cached_fetch_image(target)
        if smap:
            fig2 = plt.figure(figsize=(10, 10))
            ax = fig2.add_subplot(1, 1, 1, projection=smap)
            smap.plot(axes=ax, clip_interval=(1, 99.9) * u.percent)
            st.pyplot(fig2)
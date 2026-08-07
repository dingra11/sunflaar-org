import numpy as np
import pywt
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

def extract_physics_coupling_features(sxr, hxr, max_lag_win=15):
    """Computes causal 1D time-series indicators."""
    length = len(sxr)
    features = np.zeros((length, 4), dtype=np.float64)
    epsilon = 1e-10

    hardness = hxr / (sxr + epsilon)
    features[:, 3] = (hardness - np.mean(hardness)) / (np.std(hardness) + epsilon)

    for i in range(length):
        window_size = min(i + 1, max_lag_win)
        s_win = sxr[max(0, i - window_size + 1) : i + 1]
        h_win = hxr[max(0, i - window_size + 1) : i + 1]

        if len(s_win) > 1 and np.std(s_win) > 0 and np.std(h_win) > 0:
            features[i, 1] = np.corrcoef(s_win, h_win)[0, 1]
            features[i, 0] = (np.mean(np.abs(np.diff(h_win)) - np.abs(np.diff(s_win))) if len(s_win) > 2 else 0)
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

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
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
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=(16, 16), stride=(16, 16), padding=(0, 4))
        self.cross_attn = CrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.prob_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
        self.lead_time_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x_1d, x_2d):
        emb_1d = self.proj_1d(x_1d) + self.pos_1d[:, : x_1d.shape[1], :]
        patches = self.patch_embed(x_2d).flatten(2).transpose(1, 2)
        fused_tokens, attn_weights = self.cross_attn(emb_1d, patches)
        encoded = self.transformer(fused_tokens)
        pooled = encoded.mean(dim=1)
        return self.prob_head(pooled), self.lead_time_head(pooled), attn_weights
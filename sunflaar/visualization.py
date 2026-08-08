"""
Visualization module for SunFLAAR.
Exposed publicly as `sunvis`.
"""
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy.map

warnings.filterwarnings("ignore", module="sunpy")

def _apply_dark_theme():
    """Applies the professional dark theme to Matplotlib."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'sans-serif'],
        'axes.facecolor': '#050505',
        'figure.facecolor': '#050505',
        'text.color': '#ECEFF1',
    })

class Viewer:
    """Object-oriented interface for solar visualization."""
    
    def __init__(self):
        self._smap = None

    def _fetch_map(self, target_time_str, wavelength):
        print(f"[SunVis] Querying SDO/AIA Archive for {wavelength} Å at {target_time_str}...")
        try:
            start_time = target_time_str
            end_time = (pd.to_datetime(target_time_str) + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
            
            result = Fido.search(
                a.Time(start_time, end_time), 
                a.Instrument.aia, 
                a.Wavelength(wavelength * u.angstrom), 
                a.Sample(5 * u.minute)
            )
            
            if len(result) == 0 or len(result[0]) == 0:
                print(f"[SunVis] Error: No image found for {target_time_str}.")
                return False
                
            downloaded_file = Fido.fetch(result[0][0])
            if len(downloaded_file) == 0:
                return False
                
            self._smap = sunpy.map.Map(downloaded_file[0])
            return True
            
        except Exception as e:
            print(f"[SunVis] SYSTEM FAULT - IMAGE RETRIEVAL ERROR: {str(e)}")
            return False

    def live(self, wavelength=171, overlays=True):
        """Fetch the most recent SDO/AIA image."""
        recent_time = (pd.Timestamp.utcnow() - pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        if self._fetch_map(recent_time, wavelength):
            self.overlays = overlays

    def show(self, date=None, wavelength=171, overlays=True):
        """Render the solar image matching the exact styling from the Streamlit web portal."""
        if date is not None:
            if not self._fetch_map(date, wavelength):
                return
            self.overlays = overlays

        if self._smap is None:
            print("[SunVis] No image loaded. Pass a valid date or run .live() first.")
            return

        _apply_dark_theme()
        
        # Exact styling from Streamlit Tab 5
        fig_map = plt.figure(figsize=(10, 10), facecolor="#050505")
        ax_map = fig_map.add_subplot(1, 1, 1, projection=self._smap)

        # Plot the map
        self._smap.plot(axes=ax_map, clip_interval=(1, 99.9) * u.percent)
        
        if self.overlays:
            self._smap.draw_limb(axes=ax_map, color="#FF9800", linewidth=1.5)
            self._smap.draw_grid(axes=ax_map, color="white", alpha=0.15, linestyle=":")

        ax_map.set_title(f"SDO/AIA {self._smap.wavelength} • {self._smap.date}", fontsize=10, weight='bold', color="#ECEFF1")
        ax_map.set_xlabel("Helioprojective Longitude [arcsec]", fontsize=8, color="#90A4AE")
        ax_map.set_ylabel("Helioprojective Latitude [arcsec]", fontsize=8, color="#90A4AE")
        ax_map.tick_params(axis='both', colors='#78909C', labelsize=8)
        
        cb_map = plt.colorbar(ax_map.images[0], ax=ax_map, fraction=0.046, pad=0.04)
        cb_map.set_label("EUV INTENSITY [DN / s]", color="#90A4AE", fontsize=8)
        cb_map.ax.tick_params(labelsize=8, colors="#78909C")

        plt.show()
        return fig_map, ax_map

# --- Functional API Wrappers ---

def live(wavelength=171, overlays=True):
    """Launch a live viewer immediately."""
    viewer = Viewer()
    viewer.live(wavelength=wavelength, overlays=overlays)
    return viewer.show()

def show(date="2011-02-15 01:40:00", wavelength=171, overlays=True):
    """Quickly show a specific historical date."""
    viewer = Viewer()
    return viewer.show(date=date, wavelength=wavelength, overlays=overlays)
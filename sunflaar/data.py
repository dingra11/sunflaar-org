import warnings
import pandas as pd
import requests
import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy.timeseries as ts
import sunpy.map

warnings.filterwarnings("ignore", category=UserWarning, module="sunpy")

def fetch_sdac_observational_data(start_str, end_str):
    """Fetches RHESSI summary lightcurves from NASA SDAC."""
    results = Fido.search(a.Time(start_str, end_str), a.Instrument.rhessi, a.Physobs.summary_lightcurve)
    if len(results) == 0 or len(results[0]) == 0:
        return None
    df = ts.TimeSeries(Fido.fetch(results), concatenate=True).to_dataframe()
    return df

def fetch_solar_euv_image(target_time_str):
    """Downloads SDO/AIA 171 Angstrom EUV images."""
    try:
        start_time = target_time_str
        end_time = (pd.to_datetime(target_time_str) + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        result = Fido.search(
            a.Time(start_time, end_time), a.Instrument.aia, a.Wavelength(171 * u.angstrom), a.Sample(5 * u.minute)
        )
        if len(result) == 0 or len(result[0]) == 0:
            return None
        downloaded_file = Fido.fetch(result[0][0])
        if len(downloaded_file) == 0:
            return None
        return sunpy.map.Map(downloaded_file[0])
    except Exception as e:
        print(f"SYSTEM FAULT - IMAGE RETRIEVAL ERROR: {str(e)}")
        return None

def fetch_goes_data_json(url: str = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json") -> pd.DataFrame:
    """Fetches real-time GOES X-ray flux for the terminal CLI."""
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data from NOAA API. HTTP Status: {response.status_code}")
        
    data = response.json()
    df_raw = pd.DataFrame(data)
    df_raw['time_tag'] = pd.to_datetime(df_raw['time_tag'])
    df_pivot = df_raw.pivot(index='time_tag', columns='energy', values='flux')
    
    short_col = [c for c in df_pivot.columns if '0.4' in c][0]
    long_col = [c for c in df_pivot.columns if '0.8' in c][0]
    
    return df_pivot.rename(columns={short_col: 'Short_0.5_4A', long_col: 'Long_1_8A'})
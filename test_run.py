from sunflaar import predict
from sunflaar import sunvis
import matplotlib.pyplot as plt

# 1. LIVE PREDICTION (NOAA GOES)
forecast_live = predict.live()
forecast_live.summary()
forecast_live.plot() 

# 2. HISTORICAL PREDICTION (NASA SDAC RHESSI)
forecast_hist = predict.history(start_str="2011-02-15 01:40:00", end_str="2011-02-15 02:30:00")
forecast_hist.plot()

# 3. SDO/AIA IMAGE VISUALIZATION (Match Streamlit Date)
sunvis.show(date="2011-02-15 01:40:00", wavelength=171)
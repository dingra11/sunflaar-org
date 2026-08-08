# sunflaar/__init__.py

"""
SunFLAAR: Solar visualization, data processing, and flare forecasting.
"""

# Core modules
from . import visualization as sunvis
from . import prediction as predict

# Standard data/model modules
from . import data
from . import model
from . import plotting

# Do NOT import `app.py` here to avoid Streamlit Context errors.
__all__ = [
    "sunvis",
    "predict",
    "data",
    "model",
    "plotting"
]
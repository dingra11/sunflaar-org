<p align="center">
  <img src="logo.svg" alt="SunFLAAR Logo" width="70" style="vertical-align:middle;">

  <span style="
      font-family: Outfit, Arial, Helvetica, sans-serif;
      font-size:58px;
      font-weight:800;
      vertical-align:middle;
      margin-left:12px;
      letter-spacing:-2px;
      color:#ffffff;">
      Sun<span style="color:#facc15;">FLAAR</span>
  </span>
</p>

<h1 align="center">
SunFLAAR
</h1>

<p align="center">
Open-source Python package for AI-powered Solar Flare Analysis
</p>

<p align="center">

[![PyPI](https://img.shields.io/pypi/v/sunflaar.svg)](https://pypi.org/project/sunflaar/)
[![Python](https://img.shields.io/pypi/pyversions/sunflaar.svg)]()
[![License](https://img.shields.io/github/license/dingra11/sunflaar)]()
[![Downloads](https://static.pepy.tech/badge/sunflaar)]()
[![GitHub Stars](https://img.shields.io/github/stars/dingra11/sunflaar)]()

</p>

---

## Overview

SunFLAAR is an open-source Python package for **solar flare forecasting, active region classification, multivariate time-series analysis, and space weather research**.

It combines deep learning, statistical learning, and solar physics utilities into a single research framework suitable for:

- Solar flare prediction
- Time-series forecasting
- Active region analysis
- Imbalanced dataset evaluation
- Scientific visualization
- Space weather research

The project aims to provide an accessible interface for researchers while maintaining reproducibility and scientific transparency.

---

## Features

✔ Solar flare prediction using Deep Learning

✔ Transformer + ResNet models

✔ Multivariate Time-Series preprocessing

✔ Active Region classification

✔ Statistical evaluation metrics

✔ Publication-ready plots

✔ REST API support

✔ Command-line Interface

✔ Research friendly

---

## Installation

Install from PyPI

```bash
pip install sunflaar
```

or

```bash
python -m pip install sunflaar
```

---

## Quick Start

```python
from sunflaar import Predictor

model = Predictor()

prediction = model.predict("timeseries.csv")

print(prediction)
```

---

## Modules

| Module | Description |
|---------|-------------|
| data | Data preprocessing |
| model | AI prediction models |
| plotting | Scientific plotting |
| app | REST API |
| cli | Command Line Interface |

---

## Example Workflow

```text
GOES Data
      │
      ▼
Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Deep Learning Model
      │
      ▼
Prediction
      │
      ▼
Visualization
```

---

## Documentation

Documentation

https://sunflaar.vercel.app

API Reference

https://sunflaar.vercel.app/docs

PyPI

https://pypi.org/project/sunflaar/

---

## Scientific Applications

SunFLAAR can be used for

- Solar flare prediction
- Space weather forecasting
- Solar cycle studies
- Machine Learning research
- AI for heliophysics
- Active Region analysis

---

## Citation

If you use SunFLAAR in your research, please cite

```bibtex
@software{sunflaar2026,
  author = {Jashanpreet Singh Dingra},
  title = {SunFLAAR: Open-source Python package for Solar Flare Analysis},
  year = {2026},
  version = {1.0.4},
  url = {https://github.com/dingra11/sunflaar}
}
```

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## License

MIT License

---

## Acknowledgements

SunFLAAR builds upon the scientific Python ecosystem including

- NumPy
- SciPy
- PyTorch
- Matplotlib
- Pandas

Special thanks to the solar physics and machine learning communities.

---

## Author

**Jashanpreet Singh Dingra**

Founder — SunFLAAR

GitHub: https://github.com/dingra11

Website: https://sunflaar.vercel.app

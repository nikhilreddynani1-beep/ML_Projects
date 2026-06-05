# Metro Interstate Traffic Volume Predictor

This repository contains a notebook and dataset for predicting hourly traffic volume on a metro interstate. The notebook performs EDA, preprocessing, feature engineering, model training, evaluation, and saves a trained model.

**Author:** Nikhil
**Date:** 2026-06-05

## Contents
- `Metro_interstate_traffic_volume_predictor.ipynb` — main analysis and modeling notebook
- `Metro_Interstate_Traffic_Volume.csv` — dataset used in the notebook
- `rf_traffic_volume.joblib` — trained Random Forest model (saved by the notebook)

## Setup
1. Create and activate a Python 3.8+ virtual environment.

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS / Linux
source .venv/bin/activate
```

2. Install dependencies (recommended inside the virtualenv):

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the essentials:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## How to run
- Open and run the notebook `Metro_interstate_traffic_volume_predictor.ipynb` in Jupyter or VS Code.
- The notebook is organized into sections: Dataset, EDA, Preprocessing & Feature Engineering, Modeling, Evaluation, Conclusion.
- Running the full notebook will produce evaluation metrics and save a trained Random Forest model as `rf_traffic_volume.joblib` in the same folder.

## Results (summary)
- The notebook builds temporal features (hour, day-of-week, month) including cyclical transforms and one-hot encodes weather/holiday categories.
- A `RandomForestRegressor` is trained and evaluated (MSE, MAE, R2 reported in the notebook).
- Feature importances are displayed to show the most predictive features.

## Next steps
- Add cross-validation and hyperparameter tuning (GridSearchCV or RandomizedSearchCV).
- Create a temporal holdout to validate model stability across time.
- Add external features (event calendar, roadworks, holidays dataset) to improve predictions.
- Package the model as a simple API (FastAPI or Flask) for inference.

## Notes
- The notebook is intended to be readable and reproducible; commit both the notebook and `rf_traffic_volume.joblib` when pushing to the repo.

---

If you want, I can also generate a `requirements.txt`, add a short `README` at the repo root, or export the notebook as HTML for sharing.
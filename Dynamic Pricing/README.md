# Dynamic Pricing

## Project Overview

This project builds a dynamic pricing strategy for a ride-sharing business. It uses historical ride and customer data to analyze pricing trends, engineer demand-supply features, and train regression models that predict ride cost and support surge pricing decisions.

## Business Problem

How can a ride-sharing company optimize prices in real time to balance supply and demand while improving revenue and customer satisfaction?

## Key Objectives

- Perform exploratory data analysis with visualizations
- Engineer features such as demand-supply ratios and gap metrics
- Train regression models for price prediction
- Compare baseline and ensemble approaches
- Implement a live inference workflow for surge/discount logic

## Dataset

- `dynamic_pricing.csv`

This dataset contains historical ride data, including customer loyalty status, vehicle type, location category, ride duration, number of riders, number of drivers, and historical ride cost.

## Notebook

- `dynamic_pricing.ipynb`

The notebook includes:
- Data loading and cleaning
- Exploratory data analysis
- Feature engineering and one-hot encoding
- Model training and evaluation
- Linear Regression and Random Forest regression comparisons

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Open `dynamic_pricing.ipynb` in Jupyter Notebook or JupyterLab.
2. Run each cell sequentially.
3. Confirm that `dynamic_pricing.csv` is present in the same folder as the notebook.

## Notes

- The notebook standardizes features before modeling.
- Evaluation metrics include Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.
- You can extend the notebook with additional models such as XGBoost or LightGBM for improved performance.

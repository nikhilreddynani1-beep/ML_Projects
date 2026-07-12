# Student Placement Prediction + Segmentation

## Project Overview

This repository contains a student placement prediction project using academic, training, and profile data. The notebook demonstrates data preprocessing, label encoding, TF-IDF for categorical text features, and classifier comparison.

## Files

- `Student Placement Prediction.ipynb` — main notebook with the complete workflow
- `Sample.csv` — dataset used for the analysis

## What is included

- Data loading and exploration
- Missing value checks
- Boolean and categorical label encoding
- TF-IDF for selected categorical text fields
- Standard scaling of numeric features
- Logistic Regression and Random Forest classification
- Model evaluation with accuracy and classification reports

## How to run

1. Open `Student Placement Prediction.ipynb` in Jupyter or VS Code.
2. Ensure `Sample.csv` is in the same directory.
3. Run the notebook cells from top to bottom.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Next improvements

- Add model tuning and cross-validation
- Expand feature engineering with more text or resume data
- Add model explainability and feature importance analysis

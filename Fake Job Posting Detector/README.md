# Fake Job Posting Detector

## Project Overview

This project detects fraudulent job postings using a classic machine learning workflow with scikit-learn.

## What it uses

- `TfidfVectorizer` for text feature extraction
- `StandardScaler` for numeric feature scaling
- `OneHotEncoder` for categorical encoding
- `LogisticRegression` and `RandomForestClassifier` for classification

## Files

- `Fake Job Posting Detector.ipynb` — updated notebook with the complete workflow
- `fake_job_postings.csv` — dataset used for model training and evaluation

## How to run

1. Open `Fake Job Posting Detector.ipynb` in Jupyter or VS Code.
2. Ensure `fake_job_postings.csv` is in the same folder.
3. Run all cells from top to bottom.

## Notes

- This implementation avoids Keras/TensorFlow/XGBoost and uses only scikit-learn.
- The model uses class balancing for the imbalanced target.
- The notebook includes both Logistic Regression and Random Forest comparisons.

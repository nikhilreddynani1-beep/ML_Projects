# Smart Resume Ranking System

## Project Overview

This notebook project builds a simple resume screening model using a resume dataset and recruiter decisions. It demonstrates data cleaning, preprocessing, classification, and model evaluation with Logistic Regression and Random Forest.

## What is included

- Data loading and inspection
- Missing value handling
- Feature engineering and label encoding
- Train/test split and feature scaling
- Model training and comparison
- Accuracy metrics, classification reports, and confusion matrix visualization

## Dataset

- `AI_Resume_Screening.csv`

The dataset contains resume-related features such as Experience, Projects Count, Salary Expectation, AI Score, Job Role, Skills, Education, Certifications, and Recruiter Decision.

## Notebook

- `Smart Resume Ranking System.ipynb`

This notebook includes the full analysis pipeline and model comparison.

## How to run

1. Open the notebook in Jupyter or VS Code.
2. Ensure `AI_Resume_Screening.csv` is in the same folder.
3. Run the notebook cells from top to bottom.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Next improvements

- Add NLP processing for `Skills` and `Education`
- Expand feature engineering with resume text
- Tune model hyperparameters
- Add cross-validation and additional classifiers

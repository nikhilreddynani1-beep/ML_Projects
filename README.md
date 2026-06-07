# Machine Learning Projects

A curated collection of applied machine learning projects and notebooks. Each folder contains a self-contained analysis, dataset, and supporting files for a different business or academic use case.

## Repository Structure

- `Customer Personality Analysis/`
  - `Customer Personality Analysis.ipynb`
  - `marketing_campaign.csv`
  - `README.md`
- `Dynamic Pricing/`
  - `dynamic_pricing.ipynb`
  - `dynamic_pricing.csv`
  - `README.md`
- `Fake Job Posting Detector/`
  - `Fake Job Posting Detector.ipynb`
  - `fake_job_postings.csv`
  - `README.md`
- `Metro Interstate Traffic Volume/`
  - `Metro_interstate_traffic_volume_predictor.ipynb`
  - `Metro_Interstate_Traffic_Volume.csv`
  - `README.md`
  - `rf_traffic_volume.joblib`
- `Smart Resume Ranking System/`
  - `Smart Resume Ranking System.ipynb`
  - `AI_Resume_Screening.csv`
  - `README.md`
- `Startup Success Predictor/`
  - `startup_success_predictor.ipynb`
  - `start-up-success-classification-streamlit-app.ipynb`
  - `investments_VC.csv`
  - `README.md`
- `Student Placement Prediction + Segmentation/`
  - `Student Placement Prediction.ipynb`
  - `Sample.csv`
  - `README.md`
- `.gitignore`

## Projects

1. **Customer Personality Analysis**
   - Segments customers and models personality traits using marketing campaign data.
   - Includes exploratory data analysis, feature engineering, and visualizations.

2. **Dynamic Pricing**
   - Builds pricing strategies from supply, demand, and product data.
   - Demonstrates regression and optimization techniques.

3. **Fake Job Posting Detector**
   - Classifies job postings as legitimate or fraudulent.
   - Includes text preprocessing, feature extraction, and model validation.

4. **Metro Interstate Traffic Volume**
   - Predicts traffic volume using weather, holiday, and road data.
   - Includes model training, evaluation, and a saved `joblib` model.

5. **Smart Resume Ranking System**
   - Ranks resumes using recruiter decisions and candidate profile features.
   - Includes data cleaning, classification, and performance comparison.

6. **Startup Success Predictor**
   - Estimates startup success using investment data and company attributes.
   - Includes notebook analysis and a Streamlit demo app.

7. **Student Placement Prediction + Segmentation**
   - Predicts student placement outcomes and segments candidates.
   - Uses academic metrics, profile data, TF-IDF text features, and classifier comparison.

## Getting Started

1. Clone this repository.
2. Open the workspace in VS Code or Jupyter.
3. Open any project folder and launch the notebook for that project.
4. Confirm the dataset file is available in the same folder as the notebook.

## Recommended Environment

Install the most common dependencies used across these projects:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Optional extras for notebooks that use additional libraries:

```bash
pip install xgboost lightgbm streamlit
```

If you use a virtual environment, create and activate it first:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notes

- Each project folder may include its own `README.md` with project-specific setup and usage details.
- Keep notebook outputs clean before committing or sharing the repository.
- Use `.gitignore` to exclude temporary files, environment folders, and large artifacts.

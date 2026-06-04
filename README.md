# Machine Learning Projects

A collection of applied machine learning notebooks covering customer analysis, resume ranking, startup success prediction, and student placement modeling.

## Repository Structure

- `Customer Personality Analysis/`
  - `Customer Personality Analysis.ipynb`
  - `marketing_campaign.csv`
  - `eda_distributions.png`
  - `eda_pca.png`
- `Smart Resume Ranking System/`
  - `Smart Resume Ranking System.ipynb`
  - `AI_Resume_Screening.csv`
  - `README.md`
- `Startup Success Predictor/`
  - `startup_success_predictor.ipynb`
  - `start-up-success-classification-streamlit-app.ipynb`
  - `investments_VC.csv`
- `Student Placement Prediction + Segmentation/`
  - `Student Placement Prediction.ipynb`
  - `Sample.csv`
  - `README.md`
- `.gitignore`

## Projects

1. **Customer Personality Analysis**
   - Customer segmentation and personality modeling using marketing campaign data.
   - Includes feature exploration and visualization.
2. **Smart Resume Ranking System**
   - Resume screening model using recruiter decisions and profile features.
   - Includes preprocessing, classification, and evaluation.
3. **Startup Success Predictor**
   - Predicts startup success using venture capital investments and company attributes.
   - Includes both notebook analysis and Streamlit app.
4. **Student Placement Prediction + Segmentation**
   - Predicts student placement outcomes using academic and profile features.
   - Includes preprocessing, TF-IDF for text features, and classifier comparison.

## Getting Started

1. Clone the repository.
2. Open the workspace in VS Code or Jupyter.
3. Open any project folder and launch the notebook.
4. Verify the dataset file is present in the same project folder as the notebook.

## Recommended Environment

Install the common dependencies used across projects:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

If you use a virtual environment, create one first:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notes

- Each project folder may include its own `README.md` with project-specific details.
- Keep notebook outputs clean before pushing to the repository.
- Use the `.gitignore` file to exclude temporary files and environment artifacts.

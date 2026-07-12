# Employee Productivity & Burnout Risk Analyzer
### Using IBM HR Analytics Dataset

Predict employee burnout risk levels (Low / Medium / High) using domain-driven feature engineering, K-Means clustering, and multiple ML + ANN models — built on IBM's own publicly released HR dataset.

---

## Why This Project

Employee burnout costs companies billions annually through attrition, reduced productivity, and healthcare costs. This project builds a system to **identify at-risk employees before they leave**, giving HR teams a data-driven tool for early intervention.

IBM released this dataset themselves — making it directly relevant to IBM's own internal HR AI tools.

---

## Dataset

**Source:** [IBM HR Analytics Employee Attrition — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- 1,470 employees, 35 features, zero missing values
- Created by IBM data scientists
- Features include: Age, Department, JobSatisfaction, WorkLifeBalance, OverTime, MonthlyIncome, YearsAtCompany, and more

---

## The Unique Part — Feature Engineering

This project creates a new target variable `BurnoutRisk` that **does not exist in the original dataset** — built from domain knowledge:

```python
df['BurnoutScore'] = (
    (df['OverTime'] == 'Yes').astype(int) * 3 +        # overtime = strongest signal
    (df['WorkLifeBalance'] <= 2).astype(int) * 2 +     # poor work life balance
    (df['JobSatisfaction'] <= 2).astype(int) * 2 +     # low job satisfaction
    (df['EnvironmentSatisfaction'] <= 2).astype(int) + # poor environment
    (df['JobInvolvement'] <= 2).astype(int)             # low involvement
)
```

| Score | Risk Level | Label |
|---|---|---|
| 0 – 3 | Low Risk | 0 |
| 4 – 6 | Medium Risk | 1 |
| 7 – 9 | High Risk | 2 |

**Weight reasoning:**
- OverTime (×3) — biggest burnout driver in HR research
- WorkLifeBalance + JobSatisfaction (×2) — direct satisfaction signals
- EnvironmentSatisfaction + JobInvolvement (×1) — supporting signals

---

## Project Pipeline

```
Load Data → Feature Engineering → EDA → Preprocessing → K-Means Clustering → ML Models → ANN → Evaluation
```

---

## EDA Insights

- **Cluster 1** (burnout score 5.00) has **30.5% attrition** — over 4x higher than Cluster 2 (7.1%)
- Employees working **OverTime** have significantly higher average burnout scores
- **Sales department** shows highest average burnout score across departments
- Low JobSatisfaction + Low WorkLifeBalance employees are concentrated in high-risk cluster

---

## K-Means Clustering (Unsupervised Learning)

Segmented employees into 4 groups based on burnout-related features:

| Cluster | Size | Avg Burnout Score | Attrition Rate | Profile |
|---|---|---|---|---|
| 0 | 383 | 3.20 | 12.8% | Moderate risk — some dissatisfaction |
| 1 | 416 | 5.00 | **30.5%** | High risk — burnt out, overtime workers |
| 2 | 368 | 0.84 | 7.1% | Low risk — happy, satisfied employees |
| 3 | 303 | 2.14 | 11.6% | Low-moderate risk — generally stable |

**Key insight:** Cluster 1 employees have 30.5% attrition — HR should prioritize retention efforts on this segment.

---

## Preprocessing

- Dropped constant columns: `EmployeeCount`, `Over18`, `StandardHours` (same value for all rows)
- Dropped `EmployeeNumber` — serial ID with no predictive value
- Applied `LabelEncoder` to all categorical columns
- Scaled features with `StandardScaler`
- Train/test split: 80/20 with `stratify=y` to preserve class distribution

**Class distribution (imbalanced):**
- Low Risk: 652 (66%)
- Medium Risk: 287 (29%)
- High Risk: 45 (5%)

Handled with `class_weight = {0: 2.0, 1: 3.0, 2: 5.0}` in ANN and `class_weight='balanced'` in SVC.

---

## Model Results

| Model | Accuracy | Low Risk F1 | Medium Risk F1 | High Risk F1 |
|---|---|---|---|---|
| Logistic Regression | 87.4% | 0.93 | 0.78 | 0.56 |
| SVC (balanced) | 86.6% | 0.93 | 0.78 | 0.44 |
| ANN | 85.0% | 0.92 | 0.75 | 0.38 |
| **Random Forest** | **93.0%** | **0.97** | **0.88** | 0.37 |

**Key observations:**
- **Random Forest achieves the highest accuracy (93%)** — best overall performance
- **Logistic Regression achieves the best High Risk F1 (0.56)** — best at catching actual burnout cases
- ANN and SVC underperform on the minority High Risk class despite class weights — due to limited training samples (only 45 High Risk employees)
- This is a classic **precision vs recall tradeoff** — Random Forest is precise when it predicts High Risk but misses many cases (recall 0.23)

### ANN Architecture

```
Input (31 features)
    → Dense(128, ReLU) → Dropout(0.3)
    → Dense(64, ReLU)  → Dropout(0.3)
    → Dense(3, Softmax)
```

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy (multiclass)
- Class weights: `{0: 2.0, 1: 3.0, 2: 5.0}`
- EarlyStopping: `patience=5`, stopped at epoch 18, restored from epoch 13

---

## Visualizations

| # | Chart | Insight |
|---|---|---|
| 1 | Burnout Risk Distribution | 66% Low, 29% Medium, 5% High |
| 2 | Burnout Risk vs Attrition | High risk employees leave far more |
| 3 | Department-wise Burnout | Sales highest, Research lowest |
| 4 | Income vs Satisfaction Heatmap | Low income + low satisfaction = high burnout |
| 5 | Overtime vs Burnout Score | Overtime employees have 2x higher burnout |
| 6 | Elbow Curve | Optimal K=4 clusters |
| 7 | Cluster Scatter + Burnout Bar | Cluster 1 clearly separated as high burnout |
| 8 | ANN Training Curve | Healthy convergence, early stopping at epoch 18 |
| 9 | Confusion Matrix | Model performance breakdown per class |

---

## Limitations & Honest Observations

- High Risk class has only 45 training samples — all models struggle with recall on this class
- Burnout score is a **proxy metric** — not directly measured in the dataset but derived from domain logic
- A larger dataset with direct burnout measurements would significantly improve High Risk prediction

---

## Tech Stack

```
Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/employee-burnout-risk-analyzer

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# 3. Download dataset from Kaggle and place in project folder
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# 4. Run the notebook
jupyter notebook burnout_analyzer.ipynb
```

---

## Resume Line

> Built an Employee Burnout Risk Analyzer on IBM's HR dataset — engineered a BurnoutRisk target variable from domain knowledge, applied K-Means clustering to identify a high-risk employee segment (30.5% attrition rate), and compared Logistic Regression, SVC, Random Forest, and ANN; Random Forest achieved 93% accuracy while Logistic Regression achieved the best High Risk F1 score (0.56) on the minority class.

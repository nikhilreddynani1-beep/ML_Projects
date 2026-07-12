# Customer Churn Prediction — Telco Dataset

Predict which customers are likely to leave a telecom company using Machine Learning, Deep Learning, and K-Means Clustering.

---

## Problem Statement

Customer churn (customers leaving a service) is a major business problem in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a system to identify at-risk customers before they leave.

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers, 21 features
- Target column: `Churn` (Yes / No)
- Class imbalance: 73.5% No Churn / 26.5% Churned

---

## Project Pipeline

```
Load Data → EDA → Preprocessing → K-Means Clustering → Classification Models → Evaluation
```

---

## EDA Insights

- **Month-to-month contract customers churn at 42.7%** — nearly 3x higher than annual contract customers
- Churned customers have **lower tenure** (median ~10 months) vs retained customers (median ~38 months)
- Churned customers pay **higher monthly charges** on average
- `TotalCharges` has a strong negative correlation with churn (-0.20) — longer customers spend more and stay

---

## K-Means Clustering (Unsupervised Learning)

Segmented customers into 3 groups based on `tenure`, `MonthlyCharges`, and `TotalCharges`:

| Cluster | Size | Avg Tenure | Avg Monthly Charges | Churn Rate | Profile |
|---|---|---|---|---|---|
| 0 | 1,315 | 47.3 months | $31.5 | **4.9%** | Loyal low-cost customers |
| 1 | 2,738 | 51.5 months | $90.2 | **20.2%** | Long-term high-value customers |
| 2 | 2,990 | 8.3 months | $56.1 | **41.9%** | New customers — highest risk |

**Key insight:** Cluster 2 (new customers with short tenure) has a 41.9% churn rate — nearly 3x the overall average. Retention efforts should focus on the first 8 months of the customer lifecycle.

---

## Preprocessing

- Fixed `TotalCharges` — stored as string with 11 hidden empty values (new customers with tenure=0)
- Dropped `customerID` — no predictive value
- Applied `LabelEncoder` to all categorical columns
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) with `StandardScaler`
- Train/test split: 75% train / 25% test with `stratify=y` to preserve class ratio

---

## Models

| Model | Accuracy | F1 Score (Churn class) |
|---|---|---|
| Logistic Regression | 79.1% | 0.52 |
| Support Vector Classifier | 79.1% | 0.52 |
| ANN (2 hidden layers + Dropout) | 75.7% | 0.61 |

**Note on ANN:** Although ANN shows lower accuracy, it achieves a higher F1 score on the minority churn class (0.61 vs 0.52). This matters more than accuracy for imbalanced datasets — correctly catching churners is the business goal, not overall accuracy.

### ANN Architecture

```
Input Layer  →  Dense(128, ReLU)  →  Dropout(0.3)
             →  Dense(64, ReLU)   →  Dropout(0.3)
             →  Dense(1, Sigmoid)
```

- Optimizer: Adam
- Loss: Binary Crossentropy
- Class weight: `{0: 1.0, 1: 2.8}` to handle imbalance
- EarlyStopping: `patience=3`, `restore_best_weights=True`

---

## Visualizations

| # | Chart | Purpose |
|---|---|---|
| 1 | Churn distribution (bar + pie) | Show class imbalance |
| 2 | Tenure & MonthlyCharges by Churn (boxplot) | Key driver analysis |
| 3 | Churn rate by Contract type | Business insight |
| 4 | Correlation heatmap | Feature relationships |
| 5 | Elbow curve | Optimal K for clustering |
| 6 | Cluster scatter + churn rate per cluster | Segment analysis |
| 7 | ANN training curve | Overfitting check |
| 8 | Confusion matrices (all 3 models) | Error analysis |
| 9 | Model comparison bar chart | Final summary |

---

## Tech Stack

```
Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/customer-churn-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# 3. Download dataset from Kaggle and place in project folder
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. Run the notebook
jupyter notebook churn.ipynb
```

---

## Resume Line

> Built a customer churn predictor on Telco data — applied K-Means clustering to identify a high-risk customer segment (41.9% churn rate), compared Logistic Regression, SVC, and ANN; ANN achieved 0.61 F1 score on minority churn class using class weights and EarlyStopping.

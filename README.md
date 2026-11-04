# Telco-Churn-Prediction-Extraction
# Customer Churn: Exploration & Preprocessing

A Python script (exported from a Google Colab notebook) that explores a telecom customer churn dataset, cleans and encodes it, and prepares train/test splits for a churn-prediction model.

> **Scope:** This script covers data loading, EDA, preprocessing, and the train/test split. It does **not** train or evaluate a model yet.

## Dataset

The script expects a CSV in the format of the **Telco Customer Churn** dataset (e.g. `WA_Fn-UseC_-Telco-Customer-Churn.csv` on Kaggle). Required columns:

| Type | Columns |
|------|---------|
| Identifier (dropped) | `customerID` |
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Binary (Yes/No) | `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling` |
| Categorical | `gender`, `Contract`, `InternetService`, `PaymentMethod` |
| Target | `Churn` (Yes/No) |

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

The script uses `google.colab.files.upload()`, so it is designed for **Google Colab**:

1. Open the notebook in Colab (or upload the script as a notebook).
2. Run the cells in order.
3. When prompted, upload your churn CSV.

**Running locally:** remove these lines:

```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
```

and replace them with:

```python
df = pd.read_csv("your_churn_file.csv")
```

## What the Script Does

1. **Load data** from the uploaded CSV and print `info()` and the missing-value counts.
2. **Clean data**
   - Drops `customerID`.
   - Converts `TotalCharges` to numeric (invalid values become `NaN`) and drops those rows.
   - Maps `Churn` from `Yes/No` to `1/0`.
3. **Exploratory data analysis**
   - KDE plots of `tenure`, `MonthlyCharges`, and `TotalCharges` split by churn status.
   - Count plots of churn by `Contract`, `InternetService`, and `PaymentMethod`.
   - Box plot of `tenure` by churn.
   - Correlation heatmap of the three numeric features.
4. **Feature engineering**
   - Collapses `"No phone service"` / `"No internet service"` into `"No"` for the relevant service columns.
   - Converts Yes/No columns to 1/0, and `gender` to Female = 1, Male = 0.
   - One-hot encodes `Contract`, `InternetService`, and `PaymentMethod` into `df2`.
5. **Split:** 80/20 train/test split with `random_state=42`.
6. **Scaling:** Min-max scales `tenure`, `MonthlyCharges`, and `TotalCharges` to the 0 to 1 range, fitting the scaler on the training set and applying it to the test set.

## Outputs

| Variable | Description |
|----------|-------------|
| `df2` | Preprocessed DataFrame (all numeric) |
| `X_train`, `X_test` | Feature matrices (80% / 20%), numeric columns scaled |
| `Y_train`, `Y_test` | Churn labels (1 = churned, 0 = stayed) |


## Suggested Next Steps

- Train baseline models (Logistic Regression, Random Forest, XGBoost).
- Evaluate with precision, recall, F1, and ROC-AUC rather than accuracy alone.
- Tune hyperparameters with cross-validation.
- Analyze feature importance to identify the main churn drivers.

## File Structure

```
.
├── churndataset.py   # Exploration and preprocessing script
└── README.md
```

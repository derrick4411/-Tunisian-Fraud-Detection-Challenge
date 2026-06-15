# Tax Fraud Detection Project

**Detecting fraudulent tax declarations using Machine Learning**

A complete data science project for identifying potential VAT/Tax fraud based on Tunisian tax authority data.

---

## Project Overview

This project builds a robust fraud detection model using tax declaration data. It includes:

- Exploratory Data Analysis
- Feature Engineering focused on tax compliance patterns
- Handling class imbalance (SMOTE)
- Machine Learning model (Random Forest)
- Model explainability
- Tableau visualization recommendations
- Production-ready prediction pipeline

## Project Structure

```bash
tax-fraud-detection/
├── notebook/
│   └── Tax_Fraud_Detection.ipynb          # Main Jupyter Notebook
├── data/
│   └── raw_data.csv
├── models/
│   └── tax_fraud_detection_model.pkl
├── src/
│   ├── feature_engineering.py
│   └── utils.py
├── README.md
├── requirements.txt
└── Tableau_Dashboard_Guide.pdf
```



## Key Features & Insights

### Engineered Features
- `TVA_COMPLIANCE_RATIO`
- `DEDUCTION_RATIO`
- `ACTIVITY_DECLARATION_DIFF`
- `SUSPICIOUS_HIGH_DEDUCTION`
- High-risk category flags

### Fraud Indicators
- Abnormally high deduction-to-turnover ratio
- Large refund requests with low declared activity
- Mismatches in different TVA declarations
- Companies in high-risk categories with suspicious patterns

---

## Model Performance

- **Algorithm**: xgboost + SMOTE
- **Key Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Best Use Case**: High Recall to catch most fraud cases

---


## How to Use

### Run Prediction on New Data
```python
import joblib
import pandas as pd

model = joblib.load('models/tax_fraud_detection_model.pkl')
new_data = pd.DataFrame(...)  # Must have same columns
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:, 1]
```


---

## Resources & Next Steps

- Improve with LightGBM + Optuna tunin
- Add SHAP explainability
- Deploy as API (FastAPI)
- Real-time monitoring system


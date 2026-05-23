# 🏠 House Price Prediction using Gradient Boosting

> **Task 2 — Machine Learning Internship**
> Predicting residential house sale prices using ensemble regression models on the Ames Housing dataset.

---

## 📋 Project Overview

This project implements a complete supervised regression pipeline to predict house sale prices based on structural and locational features. Three models were trained and evaluated — **Linear Regression**, **Random Forest**, and **Gradient Boosting Regressor** — with Gradient Boosting delivering the best performance.

| Metric | Linear Regression | Random Forest | **Gradient Boosting** |
|--------|:-----------------:|:-------------:|:---------------------:|
| MAE    | ~$30,000          | ~$18,000      | **~$15,500**          |
| RMSE   | ~$45,000          | ~$27,000      | **~$22,800**          |
| R² Score | ~0.72           | ~0.88         | **~0.92** ★           |

---

## 📁 Repository Structure

```
HousePrediction/
│
├── task2houseprice.ipynb                    # Full ML pipeline notebook
├── house_price_gradient_boosting_model.pkl  # Trained Gradient Boosting model
├── house_price_preprocessor.pkl             # Fitted ColumnTransformer (scaler + encoder)
└── README.md                                # This file
```

---

## 🚀 Quick Start

### Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jmp0MGqC-lqNKWsm1Q9GVyq9HugF64dq?usp=sharing)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Vign3shposhala/HousePrediction.git
cd HousePrediction

# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Open the notebook
jupyter notebook task2houseprice.ipynb
```

---

## 📊 Dataset

The **Ames Housing Dataset** contains 1,460 residential property records from Ames, Iowa (2006–2010).

**Key features used:**

| Feature | Description |
|---------|-------------|
| `GrLivArea` | Above-grade living area (sq ft) |
| `OverallQual` | Overall material & finish quality (1–10) |
| `GarageArea` | Garage size (sq ft) |
| `YearBuilt` | Year of construction |
| `TotRmsAbvGrd` | Total rooms above grade |
| `FullBath` / `HalfBath` | Bathroom counts |
| `Neighborhood` | Physical location within city |
| `TotalSF` | Engineered: total square footage |
| `SalePrice` | **Target variable** — sale price (USD) |

---

## 🔧 Methodology

1. **EDA** — distribution analysis, correlation heatmaps, scatter plots
2. **Preprocessing** — `ColumnTransformer` with `StandardScaler` (numeric) + `OneHotEncoder` (categorical), median imputation
3. **Feature Engineering** — `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
4. **Target Transform** — log1p applied to `SalePrice` for normality
5. **Models** — Linear Regression → Random Forest → Gradient Boosting
6. **Evaluation** — MAE, RMSE, R² on 20% held-out test set
7. **Serialisation** — model and preprocessor saved as `.pkl` files

---

## 🔮 Inference Example

```python
import pickle
import numpy as np
import pandas as pd

# Load artefacts
with open('house_price_gradient_boosting_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('house_price_preprocessor.pkl', 'rb') as f:
    prep = pickle.load(f)

# Sample prediction
sample = pd.DataFrame([{
    'GrLivArea'    : 2000,
    'BedroomAbvGr' : 4,
    'FullBath'     : 2,
    'HalfBath'     : 1,
    'TotRmsAbvGrd' : 8,
    'GarageArea'   : 500,
    'YearBuilt'    : 2005,
    'OverallQual'  : 7,
    'Neighborhood' : 'NAmes',
    'TotalSF'      : 3200,
}])

X_proc = prep.transform(sample)
price  = np.expm1(model.predict(X_proc))[0]
print(f'Predicted Sale Price: ${price:,.0f}')
# → Predicted Sale Price: ~$185,000
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab)

---

## 📈 Future Work

- Hyperparameter tuning with `GridSearchCV` or `Optuna`
- Evaluation of `XGBoost` / `LightGBM` as alternative boosters
- Deployment as a `FastAPI` or `Streamlit` web application
- Incorporation of additional engineered features (house age, price-per-sqft)

---

## 👤 Author

**Vignesh Poshala**
B.Tech – CSE (Artificial Intelligence & Machine Learning)
Nalla Malla Reddy Engineering College

---

## 📄 License

This project is for academic and educational purposes.

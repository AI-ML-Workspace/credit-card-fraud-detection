# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using advanced classification algorithms, hyperparameter optimization, and production model preparation.

---

## 📌 Project Overview

Credit card fraud is a significant challenge for financial institutions. Fraudulent transactions represent a tiny fraction of total transactions, making this a highly imbalanced classification problem. 

This repository provides a step-by-step, reproducible machine learning workflow—from initial data understanding to hyperparameter tuning and production model serialization.

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset, which contains anonymized credit card transactions made by European cardholders. The dataset is highly imbalanced, with fraudulent transactions representing a very small percentage of all records.

Target Variable:
- **Class = 0** → Legitimate Transaction
- **Class = 1** → Fraudulent Transaction

## 📁 Repository Structure

```text
credit-card-fraud-detection/
├── data/
│   ├── raw/                       # Original transaction dataset
│   └── processed/                 # Scaled & preprocessed dataset (processed_creditcard.csv)
├── models/                        # Serialized model artifacts (.pkl)
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── random_forest_tuned.pkl
│   └── final_model.pkl            # Production-ready model
├── notebooks/                     # Sequential Jupyter Notebooks
│   ├── 1_data_understanding.ipynb
│   ├── 2_data_analysis.ipynb
│   ├── 3_data_preprocessing.ipynb
│   ├── 4_model_training.ipynb
│   ├── 5_model_evaluation.ipynb
│   ├── 6_hyperparameter_tuning.ipynb
│   └── 7_final_model_preparation.ipynb
├── src/                           # Application source code (under development)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Work Completed Pipeline

The project follows a structured 7-stage workflow across dedicated Jupyter Notebooks:

### 1. Data Understanding (`1_data_understanding.ipynb`)
- Inspected dataset shape, data types, missing values, and column summaries.
- Analyzed class distribution to identify severe class imbalance (Legitimate vs. Fraudulent transactions).

### 2. Exploratory Data Analysis (`2_data_analysis.ipynb`)
- Conducted univariate and bivariate statistical analysis.
- Visualized feature distributions, transaction amount spreads, and correlation heatmaps to extract key patterns.

### 3. Data Preprocessing (`3_data_preprocessing.ipynb`)
- Handled missing values and duplicate records.
- Scaled `Time` and `Amount` feature variables using standard scaling techniques.
- Exported clean, transformed dataset to `../data/processed/processed_creditcard.csv`.

### 4. Model Training (`4_model_training.ipynb`)
- Split features (`X`) and target (`Y = Class`) with a standard 80/20 train-test split (`random_state=42`).
- Trained benchmark models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier (`class_weight="balanced"`)
  - Gradient Boosting Classifier
- Serialized baseline models in `../models/`.

### 5. Model Evaluation (`5_model_evaluation.ipynb`)
- Evaluated models using comprehensive metrics: **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.
- Selected Random Forest Classifier based on the highest overall F1 Score and ROC-AUC performance on the test dataset.

### 6. Hyperparameter Tuning (`6_hyperparameter_tuning.ipynb`)
- Tuned the winning Random Forest model using `GridSearchCV` across parameters:
  - `n_estimators`: `[100, 200]`
  - `max_depth`: `[10, 20]`
  - `min_samples_split`: `[2, 5]`
- Used 3-fold cross-validation (`cv=3`) with `f1` scoring optimization.
- Serialized the tuned model to `../models/random_forest_tuned.pkl`.

### 7. Final Model Preparation (`7_final_model_preparation.ipynb`)
- Loaded the tuned model (`random_forest_tuned.pkl`).
- Executed sample prediction checks to verify pipeline inference consistency.
- Serialized and exported the finalized production artifact to `../models/final_model.pkl`.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.14+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `GridSearchCV`, `train_test_split`, metrics)
- **Model Serialization:** `joblib`
- **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AI-ML-Workspace/credit-card-fraud-detection
   cd credit-card-fraud-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute Notebooks sequentially:**
   Open Jupyter Notebook or VS Code / Antigravity IDE and run notebooks 1 through 7 sequentially in `notebooks/`.

---

## 📋 Project Status & Next Steps

- [x] Data Understanding & EDA
- [x] Preprocessing & Feature Engineering
- [x] Model Training & Evaluation
- [x] Hyperparameter Tuning
- [x] Final Production Model Serialization (`final_model.pkl`)
- [ ] **Next:** Notebook 8 / App Deployment (Interactive prediction demo)
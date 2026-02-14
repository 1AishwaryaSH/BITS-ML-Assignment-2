# Wine Quality Classification - ML Assignment 2

## a. Problem Statement

Predict whether a wine is of High Quality (score >= 6) or Low Quality (score <= 5) based on its physicochemical properties. This is a binary classification problem using 12 features derived from the UCI Wine Quality dataset.

---

## b. Dataset Description

| Property | Value |
|----------|-------|
| Name | Wine Quality - Red Wine |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/ml/datasets/wine+quality |
| Instances | 1599 (minimum required: 500) |
| Features | 12 (minimum required: 12) |
| Target | Binary: 0 = Low Quality (score <=5), 1 = High Quality (score >=6) |
| Class balance | Low Quality: 857 (54%), High Quality: 742 (46%) |
| Train/Test Split | 80% training (1279), 20% test (320) |

### Feature Descriptions (12 Features)

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 1 | fixed_acidity | Mostly tartaric acid | g/dm3 |
| 2 | volatile_acidity | Acetic acid (high = vinegar taste) | g/dm3 |
| 3 | citric_acid | Adds freshness and flavour | g/dm3 |
| 4 | residual_sugar | Sugar remaining after fermentation | g/dm3 |
| 5 | chlorides | Sodium chloride (salt content) | g/dm3 |
| 6 | free_sulfur_dioxide | Free SO2 (prevents microbial growth) | mg/dm3 |
| 7 | total_sulfur_dioxide | Total SO2 (free + bound) | mg/dm3 |
| 8 | density | Density of wine | g/cm3 |
| 9 | pH | Acidity on 0-14 scale | - |
| 10 | sulphates | Potassium sulphate (antimicrobial additive) | g/dm3 |
| 11 | alcohol | Alcohol content | % vol |
| 12 | colour | Wine colour (1 = Red, 0 = White) | binary |

---

## c. Models Used

All 6 models were trained on the same dataset with an 80/20 stratified split.
StandardScaler was applied for Logistic Regression, kNN, and Naive Bayes.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8344 | 0.9206 | 0.8322 | 0.8041 | 0.8179 | 0.6664 |
| Decision Tree | 0.7688 | 0.7907 | 0.7434 | 0.7635 | 0.7533 | 0.5359 |
| kNN | 0.7531 | 0.8242 | 0.7315 | 0.7365 | 0.7340 | 0.5037 |
| Naive Bayes | 0.8156 | 0.9166 | 0.8296 | 0.7568 | 0.7915 | 0.6290 |
| Random Forest (Ensemble) | 0.8063 | 0.8918 | 0.7986 | 0.7770 | 0.7877 | 0.6098 |
| XGBoost (Ensemble) | 0.8156 | 0.8858 | 0.8069 | 0.7905 | 0.7986 | 0.6288 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Best overall performer - highest Accuracy (0.8344), AUC (0.9206), F1 (0.8179), and MCC (0.6664). The binary wine quality target has a near-linear relationship with alcohol and volatile acidity in the scaled feature space. Fully interpretable via coefficients. Ideal for deployment where explainability matters. |
| Decision Tree | Weakest AUC (0.7907) among all models. Greedy splits capture non-linear rules but max_depth=6 is needed to prevent overfitting. Standalone trees have higher variance compared to ensemble methods, reflected in the lowest AUC score. |
| kNN | Lowest accuracy (0.7531) but reasonable AUC (0.8242). k=9 with StandardScaling is adequate for 1599 instances. Suffers from the curse of dimensionality across 12 features and is slow at inference time on larger datasets. |
| Naive Bayes | Strong AUC (0.9166) - second highest overall. The Gaussian assumption holds reasonably for physicochemical features. Lower recall (0.7568) leads to more false negatives but the model trains near-instantly and generalises well. |
| Random Forest (Ensemble) | Solid ensemble performance (AUC 0.8918, MCC 0.6098). Bagging over 200 trees reduces variance significantly compared to a single Decision Tree. Feature importance shows alcohol, sulphates, and volatile_acidity as top predictors. |
| XGBoost (Ensemble) | Strong ensemble performance (Accuracy 0.8156, AUC 0.8858, MCC 0.6288). Sequential boosting corrects residual errors and regularisation prevents overfitting. Highest precision (0.8069) and recall (0.7905) among ensemble models. |

---

## Project Structure

```
BITS-ML-Assignment-2/
|-- app.py
|-- requirements.txt
|-- README.md
|-- model/
    |-- train_models.py
    |-- meta.pkl
    |-- scaler.pkl
    |-- Logistic_Regression.pkl
    |-- Decision_Tree.pkl
    |-- kNN.pkl
    |-- Naive_Bayes.pkl
    |-- Random_Forest.pkl
    |-- XGBoost.pkl
```

---

## Streamlit App Features

- Dataset Upload (CSV) - Upload test CSV for on-the-fly predictions
- Model Selection Dropdown - Choose any of the 6 trained classifiers
- Evaluation Metrics - Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion Matrix - Heatmap with class labels (Low/High Quality)
- ROC Curve - Per-model AUC-annotated ROC curve
- Classification Report - Full per-class breakdown
- Model Comparison Table - All 6 models side-by-side with highlighting

---

## Setup and Run Locally

```
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

---

## Deployment

Deployed on Streamlit Community Cloud.
Live App: https://bits-ml-assignment-2-3dejyhmv2rsffrysrxkfgx.streamlit.app

---

## Requirements

```
streamlit>=1.32.0
scikit-learn>=1.4.0
xgboost>=2.0.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

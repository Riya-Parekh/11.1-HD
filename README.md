# SIT720 11.1HD — Heart Attack Prediction

This repository contains the reproducibility code for the submitted SIT720 11.1HD research project.

## Structure

- `SIT720_11_1HD_Execution.py` — completed end-to-end executable script.
- `src/data_loader.py` — dataset loading and exact-duplicate removal.
- `src/pipeline.py` — preprocessing, six classifiers, stacking and sigmoid calibration.
- `src/evaluate.py` — holdout and cross-validation metric functions.
- `data/heart.csv` — original 1,025-row dataset; download from the documented source if not included.
- `data/heart_deduplicated.csv` — generated 302-row unique dataset.
- `requirements.txt` — pinned dependencies.

## Reproduction

1. Install Python 3.11+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run from the repository root:

```bash
python SIT720_11_1HD_Execution.py
```

The script creates `data/heart.csv` if it is missing and then generates `data/heart_deduplicated.csv`.

## Dataset

The raw dataset used in the working notebook is the 1,025-row `heart.csv` dataset hosted at:

https://raw.githubusercontent.com/GehadGad/Heart-disease-dataset/main/heart.csv

The report records 1,025 observations, 14 columns, 13 predictors plus the binary `target`, and 723 exact duplicate rows, leaving 302 unique records.

## Method implemented

### Part 1

- Stratified 80:20 train/test split (`random_state=42`)
- Median imputation + StandardScaler for continuous features
- Most-frequent imputation + OneHotEncoder for coded categorical features
- Logistic Regression, Decision Tree, Random Forest, XGBoost, Gaussian Naive Bayes and KNN
- Logistic Regression meta-classifier stacking with 5-fold internal cross-validation

### Part 2

- Exact duplicate removal before partitioning
- Same fold-contained preprocessing architecture
- Stratified 80:20 clean holdout
- 5-fold stratified internal validation
- Sigmoid probability calibration (`CalibratedClassifierCV(method='sigmoid', cv=3)`)

The exact numerical results reported in the assessment report should be reproduced using the same environment, dataset version and notebook execution used for submission.

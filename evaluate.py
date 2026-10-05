import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "Brier": brier_score_loss(y_test, prob),
    }, model


def repeated_stratified_cv(model, X, y, cv):
    """Return fold metrics for a supplied StratifiedKFold/RepeatedStratifiedKFold object."""
    from sklearn.base import clone
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        m = clone(model)
        Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
        m.fit(Xtr, ytr)
        pred = m.predict(Xv)
        prob = m.predict_proba(Xv)[:, 1]
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(yv, pred),
            "roc_auc": roc_auc_score(yv, prob),
        })
    return pd.DataFrame(rows)

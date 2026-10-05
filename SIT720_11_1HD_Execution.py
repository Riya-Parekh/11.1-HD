"""SIT720 11.1HD - End-to-end reproduction and duplicate-aware solution.

Run from the repository root:
    python SIT720_11_1HD_Execution.py

The script downloads data/heart.csv from the documented source when it is not
present, creates data/heart_deduplicated.csv, executes Part 1 and Part 2,
and prints the resulting metrics.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_data, create_deduplicated_data
from pipeline import (
    TARGET, RANDOM_STATE, build_base_models, build_pipeline,
    build_stacking_pipeline, build_calibrated_stacking_pipeline,
)
from evaluate import evaluate_classifier, repeated_stratified_cv


def main():
    print("=== SIT720 11.1HD ===")
    print("Random state:", RANDOM_STATE)

    df = load_data(ROOT / "data" / "heart.csv")
    clean = create_deduplicated_data(df, ROOT / "data" / "heart_deduplicated.csv")

    print(f"Original shape: {df.shape}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Deduplicated shape: {clean.shape}")
    print("Target distribution after deduplication:")
    print(clean[TARGET].value_counts().sort_index())

    # ---------------- Part 1 ----------------
    print("\n=== PART 1: RAW-DATA REPRODUCTION ===")
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    for name, model in build_base_models().items():
        row, _ = evaluate_classifier(
            name, build_pipeline(model), X_train, X_test, y_train, y_test
        )
        rows.append(row)

    stack_row, stack_model = evaluate_classifier(
        "Stacking Ensemble", build_stacking_pipeline(),
        X_train, X_test, y_train, y_test
    )
    rows.append(stack_row)
    part1 = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    print(part1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- Part 2 ----------------
    print("\n=== PART 2: DUPLICATE-AWARE + CALIBRATED SOLUTION ===")
    Xc = clean.drop(columns=[TARGET])
    yc = clean[TARGET].astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.20, stratify=yc, random_state=RANDOM_STATE
    )

    # Fit the proposed calibrated stacking model on the clean training subset.
    proposed = build_calibrated_stacking_pipeline()
    proposed_row, proposed = evaluate_classifier(
        "Proposed Calibrated Stacking", proposed,
        Xc_train, Xc_test, yc_train, yc_test
    )
    print(pd.DataFrame([proposed_row]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Internal 5-fold stratified validation on the clean training subset.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_df = repeated_stratified_cv(build_calibrated_stacking_pipeline(), Xc_train, yc_train, cv)
    print("\nInternal 5-fold CV:")
    print(cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Mean CV Accuracy: {cv_df.accuracy.mean():.4f} +/- {cv_df.accuracy.std(ddof=1):.4f}")
    print(f"Mean CV ROC-AUC: {cv_df.roc_auc.mean():.4f} +/- {cv_df.roc_auc.std(ddof=1):.4f}")

    print("\nGenerated files:")
    print(ROOT / "data" / "heart.csv")
    print(ROOT / "data" / "heart_deduplicated.csv")


if __name__ == "__main__":
    main()

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_base_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(algorithm="kd_tree"),
    }


def build_pipeline(model):
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


def build_stacking_pipeline():
    estimators = [
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
        ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier(algorithm="kd_tree")),
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return build_pipeline(stacking)


def build_calibrated_stacking_pipeline():
    """Part 2: sigmoid calibration around the stacking classifier (cv=3)."""
    base_stack = build_stacking_pipeline()
    calibrated = CalibratedClassifierCV(base_stack, method="sigmoid", cv=3)
    return calibrated

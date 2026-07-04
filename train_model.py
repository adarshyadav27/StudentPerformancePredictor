"""
train_model.py
----------------
Trains several classification algorithms on the student performance dataset,
evaluates them, picks the best-performing model, and saves it (plus the
scaler/encoders and evaluation metrics) to the models/ directory.

Run:
    python train_model.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import full_preprocess, GRADE_ORDER

DATASET_PATH = "dataset/student_performance.csv"
MODELS_DIR = "models"

# Optional: XGBoost if it's installed in the environment
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_model_zoo():
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            use_label_encoder=False, eval_metric="mlogloss", random_state=42,
        )
    return models


def evaluate_model(model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            metrics["roc_auc"] = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    return metrics, y_pred


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("Preprocessing (missing values, feature engineering, encoding, scaling)...")
    X, y, scaler, encoders = full_preprocess(df, fit=True)

    classes = [c for c in GRADE_ORDER if c in y.unique()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    models = get_model_zoo()
    results = {}
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start
        metrics, _ = evaluate_model(model, X_test, y_test, classes)
        metrics["train_time_sec"] = round(duration, 3)
        results[name] = metrics
        trained_models[name] = model
        print(f"  {name:<24} accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (f1_score={results[best_name]['f1_score']:.4f})")

    # Feature importance (only for tree-based models)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(X.columns.tolist(), best_model.feature_importances_.tolist()))

    # Persist artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_columns.pkl"))

    metadata = {
        "best_model": best_name,
        "classes": classes,
        "results": results,
        "feature_importance": feature_importance,
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model + scaler + encoders + metrics to '{MODELS_DIR}/'")


if __name__ == "__main__":
    main()

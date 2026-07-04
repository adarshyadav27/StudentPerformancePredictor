
"""
preprocessing.py
-----------------
Shared preprocessing pipeline used by both train_model.py and predict.py so that
training-time and inference-time transformations always stay in sync.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that are identifiers / free text and should never be fed to the model
ID_COLUMNS = ["Student_ID", "Name", "Roll_Number"]

# Categorical columns that need label encoding
CATEGORICAL_COLUMNS = ["Gender", "Department", "Family_Income", "Internet_Access", "Extra_Curricular"]

# Numeric columns used as model features
NUMERIC_COLUMNS = [
    "Age",
    "Semester",
    "Study_Hours",
    "Attendance",
    "Assignments",
    "Previous_GPA",
    "Internal_Marks",
    "Sleep_Hours",
    "Stress_Level",
    "Participation",
]

TARGET_COLUMN = "Final_Grade"
GRADE_ORDER = ["Poor", "Average", "Good", "Excellent"]


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric columns, mode for categorical."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features that help the models."""
    df = df.copy()
    df["Study_Attendance_Ratio"] = df["Study_Hours"] / (df["Attendance"] / 10 + 1e-6)
    df["Academic_Load_Score"] = (df["Assignments"] + df["Internal_Marks"]) / 2
    df["Wellbeing_Score"] = df["Sleep_Hours"] * 10 - df["Stress_Level"] * 5
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None, fit: bool = True):
    """Label-encode categorical columns. Returns (df, encoders)."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def get_feature_columns():
    return NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [
        "Study_Attendance_Ratio",
        "Academic_Load_Score",
        "Wellbeing_Score",
    ]


def full_preprocess(df: pd.DataFrame, scaler: StandardScaler | None = None,
                     encoders: dict | None = None, fit: bool = True):
    """
    Runs the full preprocessing pipeline and returns:
        X (scaled feature dataframe), y (target series or None), scaler, encoders
    """
    df = handle_missing_values(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)

    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()

    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN]

    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    return X_scaled, y, scaler, encoders


def compute_risk_level(row: dict) -> tuple[str, int, list]:
    """
    Rule-based risk scoring (0-100, higher = riskier) independent of the ML grade model.
    Returns (risk_label, risk_score, reasons)
    """
    score = 0
    reasons = []

    attendance = float(row.get("Attendance", 100))
    gpa = float(row.get("Previous_GPA", 10))
    assignments = float(row.get("Assignments", 100))
    internal_marks = float(row.get("Internal_Marks", 100))
    study_hours = float(row.get("Study_Hours", 5))
    sleep_hours = float(row.get("Sleep_Hours", 7))
    stress_level = float(row.get("Stress_Level", 5))

    if attendance < 60:
        score += 25
        reasons.append("Low attendance (below 60%)")
    elif attendance < 75:
        score += 12
        reasons.append("Attendance below recommended 75%")

    if gpa < 5:
        score += 25
        reasons.append("Low previous GPA")
    elif gpa < 6.5:
        score += 12
        reasons.append("Below-average previous GPA")

    if assignments < 50:
        score += 15
        reasons.append("Poor assignment completion")
    elif assignments < 70:
        score += 8
        reasons.append("Assignment completion could improve")

    if internal_marks < 50:
        score += 15
        reasons.append("Low internal exam marks")

    if study_hours < 2:
        score += 10
        reasons.append("Very low daily study hours")

    if sleep_hours < 5:
        score += 5
        reasons.append("Insufficient sleep")

    if stress_level >= 8:
        score += 5
        reasons.append("High stress level reported")

    score = min(score, 100)

    if score >= 55:
        label = "High Risk"
    elif score >= 25:
        label = "Medium Risk"
    else:
        label = "Low Risk"

    if not reasons:
        reasons.append("All key indicators are within a healthy range")

    return label, score, reasons

"""
predict.py
-----------
Loads the trained model + scaler + encoders and runs real-time predictions
for a single student (given as a dict) or a batch (DataFrame).
"""

import os
import joblib
import pandas as pd

from preprocessing import full_preprocess, compute_risk_level

MODELS_DIR = "models"


class StudentPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.encoders = joblib.load(os.path.join(models_dir, "encoders.pkl"))
        self.feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))

    def predict_one(self, student: dict) -> dict:
        df = pd.DataFrame([student])
        X, _, _, _ = full_preprocess(df, scaler=self.scaler, encoders=self.encoders, fit=False)
        X = X[self.feature_columns]

        pred_class = self.model.predict(X)[0]
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probabilities = dict(zip(self.model.classes_, proba.round(4).tolist()))

        risk_label, risk_score, reasons = compute_risk_level(student)

        return {
            "prediction": pred_class,
            "probabilities": probabilities,
            "risk_level": risk_label,
            "risk_score": risk_score,
            "risk_reasons": reasons,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _, _, _ = full_preprocess(df, scaler=self.scaler, encoders=self.encoders, fit=False)
        X = X[self.feature_columns]
        preds = self.model.predict(X)
        out = df.copy()
        out["Predicted_Grade"] = preds

        risk_labels, risk_scores = [], []
        for _, row in df.iterrows():
            label, score, _ = compute_risk_level(row.to_dict())
            risk_labels.append(label)
            risk_scores.append(score)
        out["Risk_Level"] = risk_labels
        out["Risk_Score"] = risk_scores
        return out


if __name__ == "__main__":
    predictor = StudentPredictor()
    sample = {
        "Age": 20,
        "Gender": "Male",
        "Department": "Computer Science",
        "Semester": 4,
        "Study_Hours": 2.5,
        "Attendance": 58,
        "Assignments": 45,
        "Previous_GPA": 5.2,
        "Internal_Marks": 48,
        "Family_Income": "Low",
        "Internet_Access": "Yes",
        "Sleep_Hours": 5,
        "Stress_Level": 8,
        "Extra_Curricular": "No",
        "Participation": 3,
    }
    result = predictor.predict_one(sample)
    print(result)

# 🎓 AI-Powered Student Performance Predictor

An end-to-end machine learning web application that predicts a student's academic
performance (Excellent / Good / Average / Poor), flags at-risk students, and
generates personalized recommendations — built with **Python, scikit-learn, and
Streamlit**. Designed as a final-year / portfolio-ready project.

---

## ✨ Features

- **Role-based login** — Admin, Faculty, and Student views (demo auth)
- **Interactive dashboard** — grade distribution, risk breakdown, attendance vs.
  grade, correlation heatmap
- **Student data management** — add / update / delete / search / paginate,
  CSV import & export
- **ML performance prediction** — trains and compares 6 algorithms (Decision
  Tree, Random Forest, Gradient Boosting, Logistic Regression, SVM, KNN — plus
  XGBoost if installed), auto-selects the best model by F1 score
- **Risk prediction engine** — rule-based Low / Medium / High risk scoring with
  explainable reasons
- **Recommendation engine** — personalized, rule-based suggestions
- **Attendance analysis** — distribution, department comparison, simulated
  monthly trend, eligibility indicator
- **Model comparison dashboard** — accuracy / precision / recall / F1 / ROC AUC,
  confusion matrix, feature importance
- **Leaderboard** — composite-score student ranking
- **PDF report generator** — downloadable per-student performance report
- **Modern, responsive Streamlit UI** with custom CSS, badges, cards, and charts

---

## 🗂️ Project Structure

```
StudentPerformancePredictor/
├── app.py                  # Main Streamlit application
├── train_model.py          # Trains & evaluates ML models, saves the best one
├── predict.py               # Loads saved model, runs real-time predictions
├── recommendation.py        # Rule-based recommendation engine
├── preprocessing.py         # Shared cleaning / encoding / scaling / risk logic
├── requirements.txt
├── README.md
├── dataset/
│   ├── generate_dataset.py  # Synthetic dataset generator
│   └── student_performance.csv
├── models/                  # best_model.pkl, scaler.pkl, encoders.pkl, metrics.json
├── utils/
│   ├── auth.py               # Demo role-based login
│   └── pdf_report.py         # ReportLab PDF report builder
└── css/
    └── style.css             # Custom UI styling
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
A dataset is already included, but you can regenerate a fresh synthetic sample:
```bash
python dataset/generate_dataset.py
```

### 3. Train the models
```bash
python train_model.py
```
This trains all algorithms, evaluates them on a held-out test set, and saves the
best-performing model (by weighted F1 score) to `models/best_model.pkl`, along
with the fitted scaler, encoders, and a `metrics.json` summary. You can also
retrain from inside the app via **Model Comparison → Train / Retrain Models**.

### 4. Launch the app
```bash
streamlit run app.py
```

### Demo credentials
| Role    | Username | Password    |
|---------|----------|-------------|
| Admin   | admin    | admin123    |
| Faculty | faculty  | faculty123  |
| Student | student  | student123  |

> ⚠️ This login is for demo purposes only (plain-text credentials in
> `utils/auth.py`). Replace with real authentication before any production use.

---

## 🧠 Machine Learning Pipeline

1. Load CSV dataset
2. Handle missing values (median/mode imputation)
3. Feature engineering (study/attendance ratio, academic load score, wellbeing score)
4. Encode categorical variables (label encoding)
5. Feature scaling (StandardScaler)
6. Train/test split (80/20, stratified)
7. Train multiple classifiers
8. Evaluate: accuracy, precision, recall, F1, ROC AUC, confusion matrix
9. Select and persist the best model
10. Real-time inference via `predict.py` / the Streamlit UI

---

## 📊 Dataset

The included `dataset/student_performance.csv` is **synthetically generated**
(see `dataset/generate_dataset.py`) with realistic distributions and a
composite-score-derived label, so the target correlates sensibly with the
input features. Replace it with your own institution's data (same column
schema) for real use — just re-run `train_model.py` afterwards.

Columns: `Student_ID, Name, Roll_Number, Gender, Age, Department, Semester,
Study_Hours, Attendance, Assignments, Previous_GPA, Internal_Marks,
Family_Income, Internet_Access, Sleep_Hours, Stress_Level, Extra_Curricular,
Participation, Final_Grade`

---

## ☁️ Deployment Guide

### Streamlit Community Cloud (free, easiest)
1. Push this project to a public/private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select the repo/branch, and set the main file to `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t student-predictor .
docker run -p 8501:8501 student-predictor
```

### Any VM / server
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🛣️ Roadmap / Ideas for Extension

- Swap the demo dataset for real institutional data via the CSV import feature
- Replace demo auth with a real identity provider or hashed-password database
- Add SHAP-based per-prediction explainability (`shap` library)
- Add an AI chatbot (e.g. via the Anthropic API) for student guidance
- Add email delivery of PDF reports (e.g. via SMTP or SendGrid)
- Persist student data to SQLite/Postgres instead of in-memory session state

---

## ⚠️ Disclaimer

Predictions are generated by a demo model trained on synthetic data and are
**indicative only**. They should always be used alongside guidance from
qualified faculty/academic advisors, not as a sole basis for decisions about
a student.

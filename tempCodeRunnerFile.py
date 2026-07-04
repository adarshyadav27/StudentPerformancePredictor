"""
app.py
-------
AI-Powered Student Performance Predictor
Main Streamlit application.

Run:
    streamlit run app.py
"""

import os
import subprocess
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from preprocessing import compute_risk_level
from predict import StudentPredictor
from recommendation import generate_recommendations
from utils.auth import authenticate
from utils.pdf_report import build_student_report_pdf

DATASET_PATH = "dataset/student_performance.csv"
MODELS_DIR = "models"

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------------------
def init_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv(DATASET_PATH)
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"

init_state()


def model_artifacts_exist():
    return os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


@st.cache_resource(show_spinner=False)
def get_predictor():
    return StudentPredictor()


def grade_badge(grade: str) -> str:
    cls = {
        "Excellent": "badge-excellent",
        "Good": "badge-good",
        "Average": "badge-average",
        "Poor": "badge-poor",
    }.get(grade, "badge-average")
    return f'<span class="badge {cls}">{grade}</span>'


def risk_badge(risk: str) -> str:
    cls = {"Low Risk": "badge-low", "Medium Risk": "badge-medium", "High Risk": "badge-high"}.get(risk, "badge-medium")
    return f'<span class="badge {cls}">{risk}</span>'


# --------------------------------------------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------------------------------------------
def login_page():
    st.markdown("<h1 style='text-align:center;'>🎓 AI-Powered Student Performance Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6B7280;'>Sign in to continue</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.success(f"Welcome, {user['name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        with st.expander("Demo credentials"):
            st.markdown(
                "- **Admin** — `admin` / `admin123`\n"
                "- **Faculty** — `faculty` / `faculty123`\n"
                "- **Student** — `student` / `student123`"
            )


# --------------------------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------------------------
def page_dashboard():
    df = st.session_state.data
    st.subheader("📊 Overview Dashboard")

    predictor = get_predictor() if model_artifacts_exist() else None
    risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
    for _, row in df.iterrows():
        label, _, _ = compute_risk_level(row.to_dict())
        risk_counts[label] += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df))
    c2.metric("Avg. Attendance", f"{df['Attendance'].mean():.1f}%")
    c3.metric("Avg. Previous GPA", f"{df['Previous_GPA'].mean():.2f}")
    c4.metric("High Risk Students", risk_counts["High Risk"], delta=None)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Grade Distribution")
        if "Final_Grade" in df.columns:
            fig = px.pie(df, names="Final_Grade", hole=0.45,
                         color="Final_Grade",
                         color_discrete_map={"Excellent": "#16A34A", "Good": "#3B82F6",
                                             "Average": "#D97706", "Poor": "#DC2626"})
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Risk Level Breakdown")
        risk_df = pd.DataFrame({"Risk": list(risk_counts.keys()), "Count": list(risk_counts.values())})
        fig = px.bar(risk_df, x="Risk", y="Count", color="Risk",
                     color_discrete_map={"Low Risk": "#16A34A", "Medium Risk": "#D97706", "High Risk": "#DC2626"})
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Attendance vs. Final Grade")
        if "Final_Grade" in df.columns:
            fig = px.box(df, x="Final_Grade", y="Attendance", color="Final_Grade",
                         category_orders={"Final_Grade": ["Poor", "Average", "Good", "Excellent"]})
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown("#### Correlation Matrix")
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------------------
# STUDENT DATA MANAGEMENT
# --------------------------------------------------------------------------------------
def page_student_management():
    st.subheader("🗂️ Student Data Management")
    df = st.session_state.data

    tab1, tab2, tab3, tab4 = st.tabs(["View / Search", "Add Student", "Update / Delete", "Import / Export"])

    with tab1:
        search = st.text_input("🔍 Search by Name, Roll Number, or Student ID")
        dept_filter = st.multiselect("Filter by Department", sorted(df["Department"].unique()))
        filtered = df.copy()
        if search:
            mask = (
                filtered["Name"].str.contains(search, case=False, na=False)
                | filtered["Roll_Number"].str.contains(search, case=False, na=False)
                | filtered["Student_ID"].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if dept_filter:
            filtered = filtered[filtered["Department"].isin(dept_filter)]

        page_size = 15
        total_pages = max(1, (len(filtered) - 1) // page_size + 1)
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page_num - 1) * page_size
        st.dataframe(filtered.iloc[start:start + page_size], use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(page_size, len(filtered) - start)} of {len(filtered)} matching students (page {page_num}/{total_pages})")

    with tab2:
        with st.form("add_student_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Name")
                roll = st.text_input("Roll Number")
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", 15, 35, 20)
            with c2:
                department = st.selectbox("Department", sorted(df["Department"].unique()))
                semester = st.number_input("Semester", 1, 8, 1)
                attendance = st.slider("Attendance %", 0, 100, 75)
                assignments = st.slider("Assignments %", 0, 100, 75)
            with c3:
                previous_gpa = st.slider("Previous GPA", 0.0, 10.0, 7.0, 0.1)
                internal_marks = st.slider("Internal Marks", 0, 100, 60)
                study_hours = st.slider("Study Hours/day", 0.0, 12.0, 4.0, 0.5)
                sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)

            c4, c5, c6 = st.columns(3)
            with c4:
                family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
            with c5:
                internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            with c6:
                extra_curricular = st.selectbox("Extra Curricular", ["Yes", "No"])
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            participation = st.slider("Class Participation (0-10)", 0.0, 10.0, 5.0, 0.5)

            submitted = st.form_submit_button("➕ Add Student", use_container_width=True)

        if submitted:
            if not name or not roll:
                st.warning("Name and Roll Number are required.")
            else:
                new_id = f"STU{len(df) + 1:04d}"
                new_row = {
                    "Student_ID": new_id, "Name": name, "Roll_Number": roll, "Gender": gender,
                    "Age": age, "Department": department, "Semester": semester,
                    "Study_Hours": study_hours, "Attendance": attendance, "Assignments": assignments,
                    "Previous_GPA": previous_gpa, "Internal_Marks": internal_marks,
                    "Family_Income": family_income, "Internet_Access": internet_access,
                    "Sleep_Hours": sleep_hours, "Stress_Level": stress_level,
                    "Extra_Curricular": extra_curricular, "Participation": participation,
                    "Final_Grade": None,
                }
                st.session_state.data = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"Student '{name}' added with ID {new_id}.")
                st.rerun()

    with tab3:
        student_ids = df["Student_ID"].tolist()
        selected_id = st.selectbox("Select Student ID", student_ids)
        student_row = df[df["Student_ID"] == selected_id].iloc[0]

        with st.form("update_student_form"):
            c1, c2 = st.columns(2)
            with c1:
                attendance_u = st.slider("Attendance %", 0, 100, int(student_row["Attendance"]))
                assignments_u = st.slider("Assignments %", 0, 100, int(student_row["Assignments"]))
                study_hours_u = st.slider("Study Hours/day", 0.0, 12.0, float(student_row["Study_Hours"]), 0.5)
            with c2:
                internal_marks_u = st.slider("Internal Marks", 0, 100, int(student_row["Internal_Marks"]))
                previous_gpa_u = st.slider("Previous GPA", 0.0, 10.0, float(student_row["Previous_GPA"]), 0.1)
                sleep_hours_u = st.slider("Sleep Hours", 0.0, 12.0, float(student_row["Sleep_Hours"]), 0.5)

            col_update, col_delete = st.columns(2)
            update_clicked = col_update.form_submit_button("💾 Update Student", use_container_width=True)
            delete_clicked = col_delete.form_submit_button("🗑️ Delete Student", use_container_width=True)

        if update_clicked:
            idx = df[df["Student_ID"] == selected_id].index[0]
            st.session_state.data.loc[idx, "Attendance"] = attendance_u
            st.session_state.data.loc[idx, "Assignments"] = assignments_u
            st.session_state.data.loc[idx, "Study_Hours"] = study_hours_u
            st.session_state.data.loc[idx, "Internal_Marks"] = internal_marks_u
            st.session_state.data.loc[idx, "Previous_GPA"] = previous_gpa_u
            st.session_state.data.loc[idx, "Sleep_Hours"] = sleep_hours_u
            st.success(f"Student {selected_id} updated.")
            st.rerun()

        if delete_clicked:
            st.session_state.data = df[df["Student_ID"] != selected_id].reset_index(drop=True)
            st.success(f"Student {selected_id} deleted.")
            st.rerun()

    with tab4:
        st.markdown("**Import students from CSV** (columns must match the dataset schema)")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            if st.button("Merge into current data"):
                st.session_state.data = pd.concat([df, new_df], ignore_index=True).drop_duplicates(subset="Student_ID")
                st.success(f"Imported {len(new_df)} rows.")
                st.rerun()

        st.markdown("**Export current data**")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV", data=csv_bytes, file_name="student_performance_export.csv", mime="text/csv")


# --------------------------------------------------------------------------------------
# PERFORMANCE PREDICTION
# --------------------------------------------------------------------------------------
def page_prediction(fixed_student_id: str | None = None):
    st.subheader("🤖 Academic Performance Prediction")

    if not model_artifacts_exist():
        st.warning("No trained model found yet. Go to **Model Comparison** and click 'Train / Retrain Models' first.")
        return

    df = st.session_state.data
    predictor = get_predictor()

    mode = "Select existing student" if not fixed_student_id else "Select existing student"
    if not fixed_student_id:
        mode = st.radio("Input mode", ["Select existing student", "Manual entry"], horizontal=True)

    if mode == "Select existing student":
        options = [fixed_student_id] if fixed_student_id else df["Student_ID"].tolist()
        selected_id = st.selectbox("Student", options, disabled=bool(fixed_student_id))
        student = df[df["Student_ID"] == selected_id].iloc[0].to_dict()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            attendance = st.slider("Attendance %", 0, 100, 75)
            study_hours = st.slider("Study Hours/day", 0.0, 12.0, 4.0, 0.5)
            assignments = st.slider("Assignments %", 0, 100, 75)
        with c2:
            previous_gpa = st.slider("Previous GPA", 0.0, 10.0, 7.0, 0.1)
            internal_marks = st.slider("Internal Marks", 0, 100, 60)
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        with c3:
            stress_level = st.slider("Stress Level", 1, 10, 5)
            participation = st.slider("Participation", 0.0, 10.0, 5.0, 0.5)
            department = st.selectbox("Department", sorted(df["Department"].unique()))
        student = {
            "Name": "Manual Entry", "Roll_Number": "-", "Age": 20, "Gender": "Male",
            "Department": department, "Semester": 1, "Attendance": attendance,
            "Study_Hours": study_hours, "Assignments": assignments, "Previous_GPA": previous_gpa,
            "Internal_Marks": internal_marks, "Family_Income": "Medium", "Internet_Access": "Yes",
            "Sleep_Hours": sleep_hours, "Stress_Level": stress_level, "Extra_Curricular": "Yes",
            "Participation": participation,
        }

    if st.button("🔮 Predict Performance", use_container_width=True):
        result = predictor.predict_one(student)
        st.session_state["last_prediction"] = (student, result)

    if "last_prediction" in st.session_state:
        student, result = st.session_state["last_prediction"]
        st.markdown("---")
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.markdown(f"### Predicted Grade: {grade_badge(result['prediction'])}", unsafe_allow_html=True)
            st.markdown(f"### Risk Level: {risk_badge(result['risk_level'])}", unsafe_allow_html=True)
            st.progress(result["risk_score"] / 100, text=f"Risk Score: {result['risk_score']}/100")
        with col2:
            if result["probabilities"]:
                prob_df = pd.DataFrame({"Grade": list(result["probabilities"].keys()),
                                         "Probability": list(result["probabilities"].values())})
                fig = px.bar(prob_df.sort_values("Probability"), x="Probability", y="Grade", orientation="h",
                             color="Probability", color_continuous_scale="Purples")
                fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Why this risk level?"):
            for r in result["risk_reasons"]:
                st.write(f"- {r}")

        recs = generate_recommendations(student, result["prediction"], result["risk_level"])
        st.markdown("#### 💡 Personalized Recommendations")
        for icon, title, detail in recs:
            st.markdown(
                f'<div class="custom-card"><b>{icon} {title}</b><br><span style="color:#6B7280;">{detail}</span></div>',
                unsafe_allow_html=True,
            )

        pdf_bytes = build_student_report_pdf(student, result, recs)
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                            file_name=f"{student.get('Name', 'student').replace(' ', '_')}_report.pdf",
                            mime="application/pdf")


# --------------------------------------------------------------------------------------
# ATTENDANCE ANALYSIS
# --------------------------------------------------------------------------------------
def page_attendance():
    st.subheader("📅 Attendance Analysis")
    df = st.session_state.data

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Attendance Distribution")
        fig = px.histogram(df, x="Attendance", nbins=20, color_discrete_sequence=["#4338CA"])
        fig.add_vline(x=df["Attendance"].mean(), line_dash="dash", line_color="#DC2626",
                      annotation_text="Class Average")
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Avg. Attendance by Department")
        dept_att = df.groupby("Department")["Attendance"].mean().reset_index()
        fig = px.bar(dept_att, x="Department", y="Attendance", color="Attendance", color_continuous_scale="Blues")
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Individual Student — Simulated Monthly Attendance Trend")
    selected_id = st.selectbox("Select Student", df["Student_ID"].tolist(), key="att_student")
    row = df[df["Student_ID"] == selected_id].iloc[0]
    base = row["Attendance"]
    import numpy as np
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    rng = np.random.default_rng(hash(selected_id) % (2**32))
    trend = np.clip(base + rng.normal(0, 6, len(months)), 30, 100)
    trend_df = pd.DataFrame({"Month": months, "Attendance": trend})
    fig = px.line(trend_df, x="Month", y="Attendance", markers=True)
    fig.add_hline(y=75, line_dash="dot", line_color="#D97706", annotation_text="Eligibility threshold (75%)")
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: monthly trend is simulated around the recorded overall attendance percentage for illustration, since the dataset stores a single aggregate value per student.")

    if base < 75:
        st.error(f"⚠️ Attendance risk indicator: {selected_id} is below the 75% eligibility threshold ({base:.1f}%).")
    else:
        st.success(f"✅ {selected_id} meets the attendance eligibility threshold ({base:.1f}%).")


# --------------------------------------------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------------------------------------------
def page_model_comparison():
    st.subheader("⚙️ Model Training & Comparison")

    if st.button("🔁 Train / Retrain Models", use_container_width=True):
        with st.spinner("Training multiple ML algorithms... this may take a moment."):
            result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Training complete!")
            get_predictor.clear()
        else:
            st.error("Training failed. See details below.")
            st.code(result.stderr)

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        st.info("No trained models yet. Click the button above to train models on the current dataset.")
        return

    import json
    with open(metrics_path) as f:
        metadata = json.load(f)

    st.success(f"Current best model: **{metadata['best_model']}**")

    rows = []
    for name, m in metadata["results"].items():
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1 Score": m["f1_score"], "ROC AUC": m.get("roc_auc")})
    comp_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)

    st.markdown("#### Model Comparison")
    st.dataframe(comp_df.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]}),
                 use_container_width=True, hide_index=True)

    melted = comp_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                           var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group")
    fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion Matrix (Best Model)")
        cm = metadata["results"][metadata["best_model"]]["confusion_matrix"]
        classes = metadata["classes"]
        fig = px.imshow(cm, x=classes, y=classes, text_auto=True, color_continuous_scale="Purples",
                         labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Feature Importance (Best Model)")
        fi = metadata.get("feature_importance")
        if fi:
            fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}).sort_values("Importance")
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance",
                         color_continuous_scale="Purples")
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is only available for tree-based models.")


# --------------------------------------------------------------------------------------
# LEADERBOARD / RANKING
# --------------------------------------------------------------------------------------
def page_leaderboard():
    st.subheader("🏆 Student Ranking / Leaderboard")
    df = st.session_state.data.copy()
    df["Composite_Score"] = (
        0.3 * df["Attendance"] + 0.25 * df["Assignments"] + 0.25 * (df["Previous_GPA"] * 10) + 0.2 * df["Internal_Marks"]
    ) / 1.0
    ranked = df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    ranked.index += 1
    st.dataframe(
        ranked[["Name", "Roll_Number", "Department", "Attendance", "Assignments", "Previous_GPA", "Internal_Marks", "Composite_Score"]].head(50),
        use_container_width=True,
    )


# --------------------------------------------------------------------------------------
# MAIN APP / NAVIGATION
# --------------------------------------------------------------------------------------
def main_app():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"### 🎓 Student Performance AI")
        st.markdown(f"**{user['name']}**  \n_{user['role']}_")
        st.markdown("---")

        if user["role"] == "Student":
            pages = ["My Dashboard", "My Prediction", "Attendance Analysis"]
        elif user["role"] == "Faculty":
            pages = ["Dashboard", "Student Management", "Prediction", "Attendance Analysis",
                      "Leaderboard", "Model Comparison"]
        else:  # Admin
            pages = ["Dashboard", "Student Management", "Prediction", "Attendance Analysis",
                      "Leaderboard", "Model Comparison"]

        page = st.radio("Navigate", pages, label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

    if page in ("Dashboard", "My Dashboard"):
        page_dashboard()
    elif page == "Student Management":
        page_student_management()
    elif page in ("Prediction", "My Prediction"):
        if user["role"] == "Student":
            page_prediction(fixed_student_id=user.get("student_id"))
        else:
            page_prediction()
    elif page == "Attendance Analysis":
        page_attendance()
    elif page == "Leaderboard":
        page_leaderboard()
    elif page == "Model Comparison":
        page_model_comparison()


# --------------------------------------------------------------------------------------
if not st.session_state.authenticated:
    login_page()
else:
    main_app()

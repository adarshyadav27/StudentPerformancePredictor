"""
generate_dataset.py
--------------------
Generates a synthetic but realistic student performance dataset and saves it as
dataset/student_performance.csv

Run:
    python dataset/generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 800  # number of students

departments = ["Computer Science", "Electronics", "Mechanical", "Civil", "Electrical", "IT"]
genders = ["Male", "Female"]
internet_access_opts = ["Yes", "No"]
extra_curricular_opts = ["Yes", "No"]

rows = []
scores = []
for i in range(1, N + 1):
    gender = np.random.choice(genders, p=[0.55, 0.45])
    age = np.random.randint(18, 23)
    department = np.random.choice(departments)
    semester = np.random.randint(1, 9)

    study_hours = np.clip(np.random.normal(4, 1.8), 0, 10)
    attendance = np.clip(np.random.normal(78, 14), 30, 100)
    assignments = np.clip(np.random.normal(75, 15), 0, 100)
    previous_gpa = np.clip(np.random.normal(7.0, 1.3), 3.0, 10.0)
    internal_marks = np.clip(np.random.normal(65, 15), 0, 100)
    family_income = np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2])
    net_access = np.random.choice(internet_access_opts, p=[0.85, 0.15])
    sleep_hours = np.clip(np.random.normal(6.5, 1.2), 3, 10)
    stress_level = np.random.randint(1, 11)
    extra_curr = np.random.choice(extra_curricular_opts, p=[0.4, 0.6])
    participation = np.clip(np.random.normal(6, 2), 0, 10)

    score = (
        0.28 * (study_hours / 10) * 100
        + 0.22 * attendance
        + 0.15 * assignments
        + 0.15 * (previous_gpa / 10) * 100
        + 0.12 * internal_marks
        + 0.04 * (sleep_hours / 10) * 100
        + 0.04 * (participation / 10) * 100
        - 0.06 * (stress_level / 10) * 100
    )
    score += np.random.normal(0, 6)
    scores.append(score)

    rows.append(
        {
            "Student_ID": f"STU{i:04d}",
            "Name": f"Student {i}",
            "Roll_Number": f"R{2020000 + i}",
            "Gender": gender,
            "Age": age,
            "Department": department,
            "Semester": semester,
            "Study_Hours": round(study_hours, 1),
            "Attendance": round(attendance, 1),
            "Assignments": round(assignments, 1),
            "Previous_GPA": round(previous_gpa, 2),
            "Internal_Marks": round(internal_marks, 1),
            "Family_Income": family_income,
            "Internet_Access": net_access,
            "Sleep_Hours": round(sleep_hours, 1),
            "Stress_Level": stress_level,
            "Extra_Curricular": extra_curr,
            "Participation": round(participation, 1),
        }
    )

df = pd.DataFrame(rows)
scores = np.array(scores)

q_poor, q_avg, q_good = np.quantile(scores, [0.12, 0.55, 0.87])

def label(s):
    if s >= q_good:
        return "Excellent"
    elif s >= q_avg:
        return "Good"
    elif s >= q_poor:
        return "Average"
    else:
        return "Poor"

df["Final_Grade"] = [label(s) for s in scores]

df.to_csv("dataset/student_performance.csv", index=False)
print(f"Dataset generated: dataset/student_performance.csv ({len(df)} rows)")
print(df["Final_Grade"].value_counts())

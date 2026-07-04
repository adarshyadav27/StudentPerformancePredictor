
"""
recommendation.py
-------------------
Rule-based recommendation engine. Generates personalized suggestions for a
student based on their raw feature values, predicted grade, and risk level.
"""


def generate_recommendations(student: dict, prediction: str, risk_level: str) -> list:
    """Return a prioritized list of (icon, title, detail) recommendation tuples."""
    recs = []

    attendance = float(student.get("Attendance", 100))
    study_hours = float(student.get("Study_Hours", 5))
    assignments = float(student.get("Assignments", 100))
    sleep_hours = float(student.get("Sleep_Hours", 7))
    stress_level = float(student.get("Stress_Level", 5))
    participation = float(student.get("Participation", 5))
    internal_marks = float(student.get("Internal_Marks", 100))
    extra_curricular = str(student.get("Extra_Curricular", "Yes"))

    if attendance < 75:
        recs.append((
            "📅", "Improve attendance",
            f"Current attendance is {attendance:.0f}%. Aim for at least 75% to stay eligible "
            "for exams and to keep pace with coursework."
        ))

    if study_hours < 3:
        recs.append((
            "📚", "Increase daily study hours",
            f"You're studying about {study_hours:.1f} hrs/day. Gradually build up to 3-4 "
            "focused hours using techniques like the Pomodoro method."
        ))

    if assignments < 70:
        recs.append((
            "📝", "Complete pending assignments",
            "Assignment completion is below target. Prioritize outstanding submissions "
            "this week and set reminders for upcoming deadlines."
        ))

    if internal_marks < 60:
        recs.append((
            "🧪", "Take mock tests / revise fundamentals",
            "Internal marks suggest gaps in core concepts. Weekly mock tests can help "
            "identify weak topics early."
        ))

    if risk_level in ("High Risk", "Medium Risk"):
        recs.append((
            "🎓", "Meet your faculty mentor",
            "Schedule a check-in with your faculty mentor to discuss a personalized "
            "improvement plan and access any available academic support."
        ))

    if sleep_hours < 6:
        recs.append((
            "😴", "Improve sleep schedule",
            f"You're averaging {sleep_hours:.1f} hrs of sleep. Aim for 7-8 hours to "
            "improve focus, memory retention, and overall wellbeing."
        ))

    if stress_level >= 7:
        recs.append((
            "🧘", "Manage stress levels",
            "High reported stress can hurt performance. Consider short daily breaks, "
            "physical activity, or speaking with a campus counselor."
        ))

    if participation < 4:
        recs.append((
            "🙋", "Increase class participation",
            "Actively participating in class discussions reinforces learning and "
            "can improve internal assessment scores."
        ))

    if extra_curricular == "No" and risk_level == "Low Risk":
        recs.append((
            "🏅", "Consider extracurricular activities",
            "You're doing well academically — a balanced extracurricular activity "
            "can help build soft skills without hurting performance."
        ))

    # Reduce distractions is a generic but useful tip when several other flags are present
    if len(recs) >= 3:
        recs.append((
            "📵", "Reduce distractions",
            "Try scheduled 'focus blocks' with phone/social media notifications off "
            "during study sessions to make the most of your available time."
        ))

    if not recs:
        recs.append((
            "✅", "Keep up the great work!",
            "All key indicators look healthy. Maintain your current routine and "
            "consider mentoring peers or taking on advanced coursework."
        ))

    return recs

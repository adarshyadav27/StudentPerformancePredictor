"""
utils/pdf_report.py
---------------------
Generates a downloadable PDF performance report for a student using ReportLab.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def build_student_report_pdf(student: dict, result: dict, recommendations: list) -> bytes:
    """
    Builds a PDF report in memory and returns the raw bytes
    (suitable for st.download_button).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#4338CA")
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#4338CA"),
        spaceBefore=14, spaceAfter=6,
    )
    normal = styles["Normal"]

    story = []
    story.append(Paragraph("Student Performance Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')}", normal))
    story.append(Spacer(1, 0.6 * cm))

    # ---- Student info table ----
    info_rows = [
        ["Name", student.get("Name", "-"), "Roll Number", student.get("Roll_Number", "-")],
        ["Department", student.get("Department", "-"), "Semester", str(student.get("Semester", "-"))],
        ["Attendance", f"{student.get('Attendance', '-')}%", "Previous GPA", str(student.get("Previous_GPA", "-"))],
        ["Study Hours/day", str(student.get("Study_Hours", "-")), "Assignments", f"{student.get('Assignments', '-')}%"],
    ]
    info_table = Table(info_rows, colWidths=[3.3 * cm, 4.7 * cm, 3.3 * cm, 4.7 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EEF2FF")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)

    # ---- Prediction summary ----
    story.append(Paragraph("Predicted Performance", heading_style))
    pred_rows = [["Predicted Grade", result.get("prediction", "-")],
                 ["Risk Level", result.get("risk_level", "-")],
                 ["Risk Score", f"{result.get('risk_score', '-')} / 100"]]
    pred_table = Table(pred_rows, colWidths=[5 * cm, 11 * cm])
    pred_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(pred_table)

    probs = result.get("probabilities") or {}
    if probs:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Prediction Probabilities", normal))
        prob_rows = [["Grade", "Probability"]] + [
            [g, f"{p * 100:.1f}%"] for g, p in sorted(probs.items(), key=lambda x: -x[1])
        ]
        prob_table = Table(prob_rows, colWidths=[8 * cm, 8 * cm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(prob_table)

    # ---- Risk reasons ----
    reasons = result.get("risk_reasons") or []
    if reasons:
        story.append(Paragraph("Risk Factors Identified", heading_style))
        for r in reasons:
            story.append(Paragraph(f"• {r}", normal))

    # ---- Recommendations ----
    if recommendations:
        story.append(Paragraph("Personalized Recommendations", heading_style))
        for icon, title, detail in recommendations:
            story.append(Paragraph(f"<b>{icon} {title}</b> — {detail}", normal))
            story.append(Spacer(1, 0.15 * cm))

    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "This report was generated automatically by the AI-Powered Student "
        "Performance Predictor system. Predictions are indicative and should "
        "be used alongside guidance from faculty mentors.",
        ParagraphStyle("footer", parent=normal, fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

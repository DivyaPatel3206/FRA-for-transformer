import pandas as pd
import numpy as np
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from datetime import datetime
import random
import os

# ---------- Dummy AI model (replace later with your TensorFlow model) ----------
def predict_fault_type(df):
    faults = [
        "Normal",
        "Axial Displacement",
        "Radial Deformation",
        "Core Grounding",
        "Turn-to-Turn Fault"
    ]
    fault = random.choice(faults)
    prob = round(random.uniform(0.6, 0.99), 2)
    return fault, prob


# ---------- Main analysis ----------
def analyze_fra_file(df: pd.DataFrame):
    if "Frequency (Hz)" not in df.columns or "Magnitude (dB)" not in df.columns:
        raise ValueError("Input CSV must have columns: Frequency (Hz), Magnitude (dB), Phase (°)")

    fault_type, prob = predict_fault_type(df)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ---------- Plotly interactive graphs ----------
    fig_mag = go.Figure()
    fig_mag.add_trace(go.Scatter(
        x=df["Frequency (Hz)"], y=df["Magnitude (dB)"],
        mode='lines', name='Magnitude (dB)', line=dict(color='royalblue', width=2)
    ))
    fig_mag.update_layout(
        title="Frequency Response (Magnitude)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark"
    )

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(
        x=df["Frequency (Hz)"], y=df["Phase (°)"],
        mode='lines', name='Phase (°)', line=dict(color='orange', width=2)
    ))
    fig_phase.update_layout(
        title="Phase Response",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Phase (°)",
        template="plotly_dark"
    )

    # Save graph images (for PDF)
    mag_path = f"magnitude_{timestamp}.png"
    phase_path = f"phase_{timestamp}.png"
    fig_mag.write_image(mag_path)
    fig_phase.write_image(phase_path)

    # ---------- Maintenance recommendations ----------
    recommendations = {
        "Normal": "No fault detected. Continue routine FRA testing every 12 months.",
        "Axial Displacement": "Inspect the clamping structure and axial end winding supports.",
        "Radial Deformation": "Possible radial bulging of windings. Perform internal inspection.",
        "Core Grounding": "Verify insulation between core laminations and ground path.",
        "Turn-to-Turn Fault": "Severe issue. Immediate isolation and offline diagnostic recommended."
    }

    # ---------- Generate HTML interactive report ----------
    html_report = f"fra_report_{timestamp}.html"
    with open(html_report, "w", encoding="utf-8") as f:
        f.write(f"<h1>🧠 FRA Diagnostic Report</h1>")
        f.write(f"<p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        f.write(f"<p><b>Predicted Fault:</b> {fault_type}</p>")
        f.write(f"<p><b>Confidence:</b> {prob * 100}%</p>")
        f.write(f"<p><b>Recommendation:</b> {recommendations[fault_type]}</p>")
        f.write("<hr>")
        f.write("<h3>Frequency Response (Magnitude)</h3>")
        f.write(fig_mag.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h3>Phase Response</h3>")
        f.write(fig_phase.to_html(full_html=False, include_plotlyjs='cdn'))

    # ---------- Generate PDF Report ----------
    pdf_path = f"fra_report_{timestamp}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>AI-Driven FRA Diagnostic Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Predicted Fault:</b> {fault_type}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {prob * 100}%", styles["Normal"]))
    elements.append(Paragraph(f"<b>Recommendation:</b> {recommendations[fault_type]}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<b>Frequency Response (Magnitude)</b>", styles["Heading2"]))
    elements.append(Image(mag_path, width=6.5 * inch, height=3.2 * inch))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("<b>Phase Response</b>", styles["Heading2"]))
    elements.append(Image(phase_path, width=6.5 * inch, height=3.2 * inch))
    doc.build(elements)

    # Cleanup temp plots
    os.remove(mag_path)
    os.remove(phase_path)

    # ---------- Return summary ----------
    return {
        "fault_type": fault_type,
        "probability": prob,
        "html_report": html_report,
        "pdf_report": pdf_path
    }

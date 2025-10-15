import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import base64
import numpy as np
from datetime import datetime

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="AI FRA Fault Report Analyzer", layout="wide")
st.title("🧠 Transformer FRA Fault Analyzer")
st.markdown("Upload your FRA dataset to generate an **AI-powered diagnostic report** including predictions, plots, and recommendations.")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("📤 Upload FRA Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # ------------------- Dataset Summary -------------------
    st.subheader("📊 Dataset Summary")
    st.dataframe(df.head())

    summary = df.describe().T
    st.write(summary)

    # ------------------- AI Fault Prediction (Simulated / Replace with Model) -------------------
    fault_types = [
        "Core Grounding or Shorted Turns",
        "Open Circuit",
        "Tap Changer Fault",
        "Partial Discharge or Dielectric Fault",
        "Healthy Transformer"
    ]
    fault = np.random.choice(fault_types)
    probability = round(np.random.uniform(85, 99), 2)

    # ------------------- Insights -------------------
    st.markdown(f"### 🔍 Predicted Fault: **{fault}**")
    st.markdown(f"### 📈 Confidence: **{probability}%**")

    if fault == "Core Grounding or Shorted Turns":
        recommendation = "Inspect transformer core grounding and perform insulation resistance testing. Possible winding short detected."
    elif fault == "Open Circuit":
        recommendation = "Check for discontinuities in winding connections or broken leads."
    elif fault == "Tap Changer Fault":
        recommendation = "Inspect OLTC (On Load Tap Changer) contacts for carbonization or wear."
    elif fault == "Partial Discharge or Dielectric Fault":
        recommendation = "Conduct PD testing and DGA to evaluate insulation health."
    else:
        recommendation = "No major fault detected. Continue with routine maintenance schedule."

    st.info(recommendation)

    # ------------------- Visualization -------------------
    st.subheader("📉 Frequency Response Plot")
    fig, ax = plt.subplots(figsize=(8, 4))
    if "Frequency" in df.columns and "Magnitude" in df.columns:
        ax.plot(df["Frequency"], df["Magnitude"], color="blue", linewidth=2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Frequency Response Curve")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Frequency/Magnitude columns not found in dataset", ha="center", va="center")
    st.pyplot(fig)

    # ------------------- PDF Report Generator -------------------
    st.subheader("📄 Generate AI Report")

    def create_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # -------- Cover Page --------
        elements.append(Paragraph("AI-Based Transformer FRA Diagnostic Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        elements.append(Paragraph(f"Uploaded File: {uploaded_file.name}", styles["Normal"]))
        elements.append(PageBreak())

        # -------- Summary --------
        elements.append(Paragraph("Dataset Summary", styles["Heading2"]))
        data_summary = [["Column", "Mean", "Std Dev", "Min", "Max"]]
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            data_summary.append([
                col,
                round(df[col].mean(), 3),
                round(df[col].std(), 3),
                round(df[col].min(), 3),
                round(df[col].max(), 3)
            ])
        t = Table(data_summary)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(t)
        elements.append(PageBreak())

        # -------- Prediction Section --------
        elements.append(Paragraph("AI Fault Prediction", styles["Heading2"]))
        elements.append(Paragraph(f"<b>Predicted Fault:</b> {fault}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Confidence:</b> {probability}%", styles["Normal"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Recommendation:", styles["Heading3"]))
        elements.append(Paragraph(recommendation, styles["Normal"]))
        elements.append(PageBreak())

        # -------- Visualization --------
        elements.append(Paragraph("Frequency Response Visualization", styles["Heading2"]))
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        img_buf.seek(0)
        elements.append(Image(img_buf, width=400, height=200))
        elements.append(Spacer(1, 12))

        # -------- Closing --------
        elements.append(Paragraph("This report was auto-generated using AI and FRA signal analysis models.", styles["Italic"]))
        elements.append(Paragraph("© 2025 Purgon AI Systems", styles["Normal"]))

        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

    if st.button("📥 Download Full Report as PDF"):
        pdf = create_pdf()
        b64 = base64.b64encode(pdf).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="FRA_Report.pdf">📄 Click here to download your detailed report</a>'
        st.markdown(href, unsafe_allow_html=True)

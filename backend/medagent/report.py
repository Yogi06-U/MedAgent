"""Clinical report generation (Markdown & ReportLab PDF)."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from medagent.config import settings
from medagent.schemas import (
    CaseInput,
    ImagingReport,
    LabReport,
    RagReport,
    ClinicalDraft,
    DoctorDecision,
    FinalReport,
    CLINICAL_DISCLAIMER,
)


def generate_markdown_report(
    case_input: CaseInput,
    draft: ClinicalDraft,
    decision: DoctorDecision,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
) -> str:
    """Format full markdown representation of the case and decision."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_text = decision.edited_summary if decision.edited_summary else draft.summary

    md_lines = [
        f"# MedAgent Clinical Decision-Support Report",
        f"**Case ID:** `{case_input.case_id}` | **Status:** {decision.decision.upper()} | **Date:** {now_str}",
        f"**Attending Physician / Reviewer:** {decision.reviewer}",
        "",
        f"> **LEGAL & REGULATORY NOTICE:** {CLINICAL_DISCLAIMER}",
        "",
        "## 1. Patient Profile & Intake",
        f"- **Age:** {case_input.patient_age if case_input.patient_age is not None else 'Unspecified'}",
        f"- **Sex:** {case_input.patient_sex if case_input.patient_sex else 'Unspecified'}",
        f"- **Clinical Notes:** {case_input.clinical_text}",
        "",
        "## 2. Clinical Impression / Summary",
        f"{summary_text}",
        "",
        "## 3. Ranked Differential Diagnoses",
    ]

    for i, diff in enumerate(draft.differentials, 1):
        md_lines.append(f"### {i}. {diff.condition} (Likelihood: {diff.likelihood.upper()})")
        md_lines.append(f"- **Rationale:** {diff.rationale}")
        if diff.supporting_findings:
            md_lines.append(f"- **Supporting Findings:** {', '.join(diff.supporting_findings)}")
        if diff.against:
            md_lines.append(f"- **Discordant / Against:** {diff.against}")
        if diff.evidence_ids:
            md_lines.append(f"- **Literature Citations:** {', '.join(diff.evidence_ids)}")
        md_lines.append("")

    if imaging_report and imaging_report.status == "ok":
        md_lines.append("## 4. Radiographic Findings (Chest X-Ray)")
        md_lines.append(f"- **Model:** {imaging_report.model_name}")
        for f in imaging_report.findings:
            md_lines.append(
                f"- **{f.finding}:** Probability {f.probability:.1%}, Location: {f.location}, Severity: {f.severity}, Confidence: {f.confidence}"
            )
        if imaging_report.image_quality_notes:
            md_lines.append(f"- **Quality Notes:** {'; '.join(imaging_report.image_quality_notes)}")
        md_lines.append("")

    if lab_report and lab_report.status == "ok":
        md_lines.append("## 5. Laboratory Findings & Patterns")
        if lab_report.patterns:
            md_lines.append("**Synthesized Patterns:**")
            for pat in lab_report.patterns:
                md_lines.append(f"- [{pat.urgency.upper()}] **{pat.name}**: {pat.description}")
        md_lines.append("")

    if draft.suggested_workup:
        md_lines.append("## 6. Suggested Diagnostic Workup")
        for item in draft.suggested_workup:
            md_lines.append(f"- {item}")
        md_lines.append("")

    if draft.red_flags:
        md_lines.append("## 7. Red Flags & Urgent Considerations")
        for rf in draft.red_flags:
            md_lines.append(f"- ⚠️ {rf}")
        md_lines.append("")

    md_lines.append("## 8. Physician Gate Review")
    md_lines.append(f"- **Decision:** `{decision.decision}`")
    md_lines.append(f"- **Reviewer Notes:** {decision.notes or 'None'}")
    md_lines.append(f"- **Attestation:** Reviewed and approved for clinical release by {decision.reviewer}.")

    return "\n".join(md_lines)


def generate_pdf_report(
    case_input: CaseInput,
    draft: ClinicalDraft,
    decision: DoctorDecision,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
) -> str:
    """Compile PDF document using ReportLab and return absolute file path."""
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"report_{case_input.case_id}.pdf"
    pdf_path = settings.REPORTS_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0E3B43"),
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0E3B43"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A202C"),
    )
    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#C0392B"),
        backColor=colors.HexColor("#FDF2E9"),
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=10,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("MedAgent Clinical Decision Support", title_style))
    story.append(Spacer(1, 4))
    
    meta_text = (
        f"<b>Case ID:</b> {case_input.case_id} &nbsp;|&nbsp; "
        f"<b>Status:</b> {decision.decision.upper()} &nbsp;|&nbsp; "
        f"<b>Physician:</b> {decision.reviewer} &nbsp;|&nbsp; "
        f"<b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 6))

    # Mandatory Disclaimer
    story.append(Paragraph(f"<b>CRITICAL DISCLAIMER:</b> {CLINICAL_DISCLAIMER}", disclaimer_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1F8A9A"), spaceAfter=8))

    # Patient info
    patient_info = (
        f"<b>Patient:</b> Age: {case_input.patient_age or 'N/A'}, Sex: {case_input.patient_sex or 'N/A'}<br/>"
        f"<b>Notes:</b> {case_input.clinical_text}"
    )
    story.append(Paragraph("Patient Intake & Presentation", h2_style))
    story.append(Paragraph(patient_info, body_style))

    # Clinical Impression
    summary_text = decision.edited_summary if decision.edited_summary else draft.summary
    story.append(Paragraph("Clinical Summary / Impression", h2_style))
    story.append(Paragraph(summary_text, body_style))

    # Differentials
    story.append(Paragraph("Ranked Differential Diagnoses", h2_style))
    diff_data = [["#", "Condition", "Likelihood", "Key Supporting Findings", "Citations"]]
    for idx, d in enumerate(draft.differentials, 1):
        diff_data.append([
            str(idx),
            d.condition,
            d.likelihood.capitalize(),
            Paragraph(", ".join(d.supporting_findings) or "Clinical context", body_style),
            ", ".join(d.evidence_ids) or "None",
        ])

    diff_table = Table(diff_data, colWidths=[20, 160, 65, 230, 55])
    diff_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E3B43")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(diff_table)

    # Imaging Findings
    if imaging_report and imaging_report.status == "ok" and imaging_report.findings:
        story.append(Paragraph("Radiographic Findings (Chest X-Ray)", h2_style))
        img_data = [["Pathology", "Probability", "Location", "Indicative Severity", "Confidence"]]
        for f in imaging_report.findings:
            img_data.append([
                f.finding,
                f"{f.probability:.1%}",
                f.location,
                f.severity.capitalize(),
                f.confidence.capitalize(),
            ])
        img_table = Table(img_data, colWidths=[140, 80, 150, 90, 70])
        img_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F8A9A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(img_table)

    # Laboratory Findings
    if lab_report and lab_report.status == "ok" and lab_report.results:
        story.append(Paragraph("Laboratory Findings & Identified Patterns", h2_style))
        if lab_report.patterns:
            patterns_text = "<b>Identified Clinical Patterns:</b> " + "; ".join(
                [f"<b>{p.name}</b> ({p.urgency})" for p in lab_report.patterns]
            )
            story.append(Paragraph(patterns_text, body_style))
            story.append(Spacer(1, 4))

    # Physician Attestation
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceAfter=8))
    attestation_text = (
        f"<b>Physician Attestation:</b> I, Dr. {decision.reviewer}, hereby confirm that I have reviewed, "
        f"verified, and adjusted this decision-support synthesis. Decision: <b>{decision.decision.upper()}</b>.<br/>"
        f"<b>Reviewer Notes:</b> {decision.notes or 'No specific notes recorded.'}<br/>"
        f"<b>Signature:</b> ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; <b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    story.append(Paragraph(attestation_text, body_style))

    doc.build(story)
    return str(pdf_path)

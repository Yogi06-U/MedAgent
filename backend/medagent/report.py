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
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"report_{case_input.case_id}.pdf"
    pdf_path = settings.REPORTS_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Medical theme color palette
    c_primary = colors.HexColor("#0E3B43")
    c_secondary = colors.HexColor("#1F8A9A")
    c_dark = colors.HexColor("#1E293B")
    c_border = colors.HexColor("#CBD5E1")
    c_grid = colors.HexColor("#E2E8F0")
    c_alt_row = colors.HexColor("#F8FAFC")
    c_danger_text = colors.HexColor("#991B1B")
    c_danger_bg = colors.HexColor("#FEE2E2")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        textColor=c_primary,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_primary,
        spaceBefore=7,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=c_dark,
    )
    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9.5,
        textColor=c_danger_text,
        backColor=c_danger_bg,
        borderPadding=4,
        spaceBefore=3,
        spaceAfter=5,
    )

    # Reusable table cell typography
    th_style = ParagraphStyle(
        "TableTH",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.whitesmoke,
        alignment=TA_LEFT,
    )
    th_center_style = ParagraphStyle(
        "TableTHCenter",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.whitesmoke,
        alignment=TA_CENTER,
    )
    td_style = ParagraphStyle(
        "TableTD",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=c_dark,
        alignment=TA_LEFT,
    )
    td_bold = ParagraphStyle(
        "TableTDBold",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=c_primary,
        alignment=TA_LEFT,
    )
    td_center = ParagraphStyle(
        "TableTDCenter",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=c_dark,
        alignment=TA_CENTER,
    )
    td_center_bold = ParagraphStyle(
        "TableTDCenterBold",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=c_primary,
        alignment=TA_CENTER,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("MedAgent Clinical Decision Support", title_style))
    story.append(Spacer(1, 2))

    meta_text = (
        f"<b>Case ID:</b> {case_input.case_id} &nbsp;|&nbsp; "
        f"<b>Status:</b> {decision.decision.upper()} &nbsp;|&nbsp; "
        f"<b>Physician:</b> {decision.reviewer} &nbsp;|&nbsp; "
        f"<b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 3))

    # Mandatory Regulatory Disclaimer
    story.append(Paragraph(f"<b>CRITICAL DISCLAIMER:</b> {CLINICAL_DISCLAIMER}", disclaimer_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=4))

    # Patient Intake & Presentation
    story.append(Paragraph("Patient Intake & Presentation", h2_style))
    patient_info = (
        f"<b>Patient:</b> Age: {case_input.patient_age if case_input.patient_age is not None else 'Unspecified'}, "
        f"Sex: {case_input.patient_sex.capitalize() if case_input.patient_sex else 'Unspecified'}<br/>"
        f"<b>Notes & History:</b> {case_input.clinical_text}"
    )
    story.append(Paragraph(patient_info, body_style))

    # Clinical Summary / Impression
    summary_text = decision.edited_summary if decision.edited_summary else draft.summary
    story.append(Paragraph("Clinical Summary / Impression", h2_style))
    story.append(Paragraph(summary_text, body_style))

    # Red Flags & Urgent Alerts
    if draft.red_flags:
        rf_items = "<br/>".join([f"• <b>{rf}</b>" for rf in draft.red_flags])
        rf_box = ParagraphStyle(
            "DocRedFlags",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=c_danger_text,
            backColor=c_danger_bg,
            borderPadding=4,
            spaceBefore=3,
            spaceAfter=4,
        )
        story.append(Paragraph(f"<b>[URGENT CLINICAL ALERT] RED FLAGS IDENTIFIED:</b><br/>{rf_items}", rf_box))

    # Ranked Differential Diagnoses Table
    story.append(Paragraph("Ranked Differential Diagnoses", h2_style))
    diff_data = [[
        Paragraph("#", th_center_style),
        Paragraph("Condition", th_style),
        Paragraph("Likelihood", th_center_style),
        Paragraph("Key Supporting Findings", th_style),
        Paragraph("Citations", th_center_style),
    ]]

    for idx, d in enumerate(draft.differentials, 1):
        lk = d.likelihood.lower()
        if lk == "high":
            lk_color = "#991B1B"
        elif lk == "moderate":
            lk_color = "#B45309"
        else:
            lk_color = "#475569"
        lk_html = f'<font color="{lk_color}"><b>{d.likelihood.upper()}</b></font>'

        supp_text = ", ".join(d.supporting_findings) if d.supporting_findings else "Clinical context"
        cit_text = ", ".join(d.evidence_ids) if d.evidence_ids else "None"

        diff_data.append([
            Paragraph(str(idx), td_center_bold),
            Paragraph(d.condition, td_bold),
            Paragraph(lk_html, td_center),
            Paragraph(supp_text, td_style),
            Paragraph(cit_text, td_center),
        ])

    diff_table = Table(diff_data, colWidths=[22, 145, 68, 240, 65])
    diff_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_alt_row]),
        ("BOX", (0, 0), (-1, -1), 0.5, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_grid),
    ]))
    story.append(diff_table)

    # Radiographic Findings (Chest X-Ray) Table
    if imaging_report and imaging_report.status == "ok":
        story.append(Paragraph("Radiographic Findings (Chest X-Ray)", h2_style))
        if imaging_report.findings:
            img_data = [[
                Paragraph("Pathology / Finding", th_style),
                Paragraph("Probability", th_center_style),
                Paragraph("Location", th_style),
                Paragraph("Indicative Severity", th_center_style),
                Paragraph("Confidence", th_center_style),
            ]]
            for f in imaging_report.findings:
                img_data.append([
                    Paragraph(f.finding, td_bold),
                    Paragraph(f"<b>{f.probability:.1%}</b>", td_center),
                    Paragraph(f.location, td_style),
                    Paragraph(f.severity.capitalize(), td_center),
                    Paragraph(f.confidence.capitalize(), td_center),
                ])
            img_table = Table(img_data, colWidths=[140, 68, 182, 80, 70])
            img_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_alt_row]),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, c_grid),
            ]))
            story.append(img_table)
        else:
            story.append(Paragraph("<b>Unremarkable Radiograph:</b> No acute cardiopulmonary abnormalities detected above operating threshold.", body_style))

        if imaging_report.image_quality_notes:
            notes_str = "; ".join(imaging_report.image_quality_notes)
            story.append(Paragraph(f"<i>Quality Notes:</i> {notes_str}", body_style))

    # Laboratory Findings & Identified Patterns
    if lab_report and lab_report.status == "ok":
        has_results = bool(lab_report.results)
        has_patterns = bool(lab_report.patterns)
        if has_results or has_patterns:
            story.append(Paragraph("Laboratory Findings & Identified Patterns", h2_style))

            if has_results:
                lab_data = [[
                    Paragraph("Analyte / Test", th_style),
                    Paragraph("Result", th_center_style),
                    Paragraph("Unit", th_center_style),
                    Paragraph("Reference Interval", th_center_style),
                    Paragraph("Flag", th_center_style),
                ]]

                for r in lab_report.results:
                    flag_val = r.flag.lower()
                    if "critical" in flag_val:
                        flag_html = f'<font color="#991B1B"><b>{r.flag.upper()}</b></font>'
                    elif flag_val in ("high", "low"):
                        flag_html = f'<font color="#B45309"><b>{r.flag.upper()}</b></font>'
                    elif flag_val == "normal":
                        flag_html = f'<font color="#065F46">NORMAL</font>'
                    else:
                        flag_html = f'<font color="#64748B">{r.flag.upper()}</font>'

                    ref_str = (
                        f"{r.ref_low:g} – {r.ref_high:g}"
                        if r.ref_low is not None and r.ref_high is not None
                        else "—"
                    )

                    lab_data.append([
                        Paragraph(r.test or r.raw_name, td_bold),
                        Paragraph(f"<b>{r.value:g}</b>", td_center),
                        Paragraph(r.unit, td_center),
                        Paragraph(ref_str, td_center),
                        Paragraph(flag_html, td_center),
                    ])

                lab_table = Table(lab_data, colWidths=[165, 70, 75, 130, 100])
                lab_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_alt_row]),
                    ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, c_grid),
                ]))
                story.append(lab_table)
                story.append(Spacer(1, 3))

            if lab_report.inferred_diseases:
                inf_lines = []
                for d in lab_report.inferred_diseases:
                    conf_txt = f" [{d.confidence_score * 100.0:.1f}% Neural Confidence]" if getattr(d, "confidence_score", None) is not None else ""
                    if d.disease == "Normal / Unremarkable Laboratory Profile":
                        inf_lines.append(f"• <b>{d.disease}</b>{conf_txt}: {d.clinical_rationale}")
                        continue
                    sev_col = "#991B1B" if d.severity in ("critical", "severe") else "#B45309" if d.severity == "moderate" else "#065F46"
                    sev_badge = f'<font color="{sev_col}"><b>[{d.severity.upper()}]</b></font>'
                    inf_lines.append(f"• {sev_badge} <b>{d.disease}</b>{conf_txt} (Likelihood: {d.likelihood.upper()}): {d.clinical_rationale} <i>Pathophysiology:</i> {d.pathophysiology}")
                story.append(Paragraph(f"<b>Inferred Diagnoses (ClinicalLabNet Neural Classifier):</b><br/>{'<br/>'.join(inf_lines)}", body_style))
                story.append(Spacer(1, 3))

            if has_patterns:
                pat_lines = []
                for p in lab_report.patterns:
                    urg_color = "#991B1B" if p.urgency == "urgent" else "#B45309" if p.urgency == "soon" else "#065F46"
                    urg_badge = f'<font color="{urg_color}"><b>[{p.urgency.upper()}]</b></font>'
                    pat_lines.append(f"• {urg_badge} <b>{p.name}:</b> {p.description}")
                patterns_text = "<br/>".join(pat_lines)
                story.append(Paragraph(f"<b>Synthesized Clinical Patterns:</b><br/>{patterns_text}", body_style))

    # Suggested Diagnostic Workup
    if draft.suggested_workup:
        story.append(Paragraph("Suggested Diagnostic Workup", h2_style))
        workup_items = "<br/>".join([f"• {item}" for item in draft.suggested_workup])
        story.append(Paragraph(workup_items, body_style))

    # Physician Attestation & Signature
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceAfter=6))

    doc_name = decision.reviewer.strip()
    if not doc_name.lower().startswith("dr.") and not doc_name.lower().startswith("dr "):
        doc_display = f"Dr. {doc_name}"
    else:
        doc_display = doc_name

    attestation_text = (
        f"<b>Physician Attestation:</b> I, {doc_display}, hereby confirm that I have reviewed, "
        f"verified, and adjusted this decision-support synthesis. Decision: <b>{decision.decision.upper()}</b>.<br/>"
        f"<b>Reviewer Notes:</b> {decision.notes or 'No specific notes recorded.'}<br/>"
        f"<b>Signature:</b> ___________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        f"<b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    story.append(Paragraph(attestation_text, body_style))

    doc.build(story)
    return str(pdf_path)

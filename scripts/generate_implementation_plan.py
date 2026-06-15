"""
PyFlowML Implementation Plan PDF Generator

Generates a professional PDF document outlining the v1.0.4 → v2.0 implementation roadmap.
Outputs to: ./reports/PyFlowML_Implementation_Plan.pdf

Usage:
    python scripts/generate_implementation_plan.py

Requirements:
    - reportlab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Flowable
from datetime import date
import os

# ── Colour palette ────────────────────────────────────────────────────────────
DARK       = colors.HexColor("#1a1a2e")
MID        = colors.HexColor("#16213e")
ACCENT     = colors.HexColor("#0f3460")
HIGHLIGHT  = colors.HexColor("#e94560")
TEAL       = colors.HexColor("#0d7377")
GOLD       = colors.HexColor("#f5a623")
LIGHT_BG   = colors.HexColor("#f7f8fc")
CARD_BG    = colors.HexColor("#eef0f8")
TEXT_DARK  = colors.HexColor("#1c1c2e")
TEXT_MID   = colors.HexColor("#444466")
TEXT_LIGHT = colors.HexColor("#6b7280")
WHITE      = colors.white
GREEN      = colors.HexColor("#16a34a")
RED        = colors.HexColor("#dc2626")
ORANGE     = colors.HexColor("#ea580c")
BLUE       = colors.HexColor("#2563eb")

PAGE_W, PAGE_H = A4
M = 18 * mm   # margin

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

cover_title  = S("CoverTitle",  fontName="Helvetica-Bold",   fontSize=32, textColor=WHITE,     spaceAfter=6,  alignment=TA_CENTER, leading=38)
cover_sub    = S("CoverSub",    fontName="Helvetica",        fontSize=13, textColor=colors.HexColor("#c0c8e8"), spaceAfter=4, alignment=TA_CENTER, leading=18)
cover_badge  = S("CoverBadge",  fontName="Helvetica-Bold",   fontSize=10, textColor=colors.HexColor("#a0aacc"), alignment=TA_CENTER)

h1           = S("H1",          fontName="Helvetica-Bold",   fontSize=18, textColor=DARK,      spaceBefore=14, spaceAfter=6,  leading=22)
h2           = S("H2",          fontName="Helvetica-Bold",   fontSize=13, textColor=ACCENT,    spaceBefore=10, spaceAfter=4,  leading=17)
h3           = S("H3",          fontName="Helvetica-Bold",   fontSize=11, textColor=TEAL,      spaceBefore=6,  spaceAfter=3,  leading=15)
body         = S("Body",        fontName="Helvetica",        fontSize=9.5,textColor=TEXT_DARK, spaceBefore=2,  spaceAfter=3,  leading=14)
body_sm      = S("BodySm",      fontName="Helvetica",        fontSize=8.5,textColor=TEXT_MID,  spaceAfter=2,   leading=12)
code_style   = S("Code",        fontName="Courier",          fontSize=8,  textColor=colors.HexColor("#1e3a5f"), backColor=CARD_BG, spaceAfter=2, leading=11, leftIndent=8, borderPadding=4)
label_style  = S("Label",       fontName="Helvetica-Bold",   fontSize=8,  textColor=WHITE,     alignment=TA_CENTER)
toc_style    = S("TOC",         fontName="Helvetica",        fontSize=10, textColor=TEXT_DARK, spaceAfter=3,   leading=14)
toc_num      = S("TOCNum",      fontName="Helvetica-Bold",   fontSize=10, textColor=HIGHLIGHT, spaceAfter=3,   leading=14)
section_hdr  = S("SectionHdr",  fontName="Helvetica-Bold",   fontSize=10, textColor=WHITE,     alignment=TA_CENTER)
note_style   = S("Note",        fontName="Helvetica-Oblique",fontSize=8.5,textColor=TEXT_MID,  spaceAfter=2,   leading=12, leftIndent=10)

# ── Custom Flowables ──────────────────────────────────────────────────────────
class ColorRect(Flowable):
    def __init__(self, w, h, color, radius=4):
        super().__init__()
        self.w, self.h, self.color, self.radius = w, h, color, radius
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.w, self.h, self.radius, fill=1, stroke=0)

class FullWidthRect(Flowable):
    def __init__(self, h, color):
        super().__init__()
        self.h, self.color = h, color
        self.width = PAGE_W - 2 * M
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.h, fill=1, stroke=0)

class Pill(Flowable):
    """Coloured pill badge inline-ish — used as a standalone line."""
    def __init__(self, text, bg, fg=WHITE, w=60, h=14):
        super().__init__()
        self.text, self.bg, self.fg = text, bg, fg
        self.w, self.h = w, h
    def wrap(self, *a): return self.w, self.h + 4
    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.roundRect(0, 2, self.w, self.h, self.h / 2, fill=1, stroke=0)
        c.setFillColor(self.fg)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(self.w / 2, 2 + self.h / 2 - 3.5, self.text)

def priority_color(p):
    return {
        "CRITICAL": RED,
        "HIGH":     ORANGE,
        "MEDIUM":   GOLD,
        "LOW":      TEAL,
    }.get(p, BLUE)

def phase_color(ph):
    return {1: HIGHLIGHT, 2: TEAL, 3: BLUE, 4: GOLD}.get(ph, ACCENT)

def section_banner(title, subtitle=""):
    data = [[Paragraph(title, section_hdr)]]
    t = Table(data, colWidths=[PAGE_W - 2 * M])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
    ]))
    elems = [Spacer(1, 6), t]
    if subtitle:
        elems.append(Paragraph(subtitle, body_sm))
    elems.append(Spacer(1, 4))
    return elems

def info_card(items, title=None):
    """Shaded card with bullet rows."""
    rows = []
    if title:
        rows.append([Paragraph(f"<b>{title}</b>", h3), ""])
    for label, val in items:
        rows.append([
            Paragraph(f"<b>{label}</b>", body_sm),
            Paragraph(val, body_sm)
        ])
    t = Table(rows, colWidths=[55*mm, PAGE_W - 2*M - 55*mm - 4])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
        ("ROUNDEDCORNERS",[6]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return [t, Spacer(1, 6)]

def upgrade_card(num, title, priority, phase, module, effort, what, why, how, changes):
    pc = priority_color(priority)
    phc = phase_color(phase)

    header_data = [[
        Paragraph(f"<b>#{num}  {title}</b>", h2),
        Paragraph(f"<b>{priority}</b>", label_style),
        Paragraph(f"Phase {phase}", label_style),
    ]]
    header = Table(header_data, colWidths=[PAGE_W - 2*M - 68*mm, 34*mm, 30*mm])
    header.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), LIGHT_BG),
        ("BACKGROUND",    (1, 0), (1, 0), pc),
        ("BACKGROUND",    (2, 0), (2, 0), phc),
        ("ROUNDEDCORNERS",[6]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (0, 0), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    meta = Table([
        [Paragraph(f"<b>Module:</b> {module}", body_sm),
         Paragraph(f"<b>Effort:</b> {effort}", body_sm)]
    ], colWidths=[(PAGE_W - 2*M)/2]*2)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))

    def row3(a, b, c):
        t = Table([[
            Paragraph(f"<b>{a[0]}</b><br/>{a[1]}", body_sm),
            Paragraph(f"<b>{b[0]}</b><br/>{b[1]}", body_sm),
            Paragraph(f"<b>{c[0]}</b><br/>{c[1]}", body_sm),
        ]], colWidths=[(PAGE_W - 2*M)/3]*3)
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d4e8")),
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ]))
        return t

    body_tbl = row3(
        ("What", what),
        ("Why", why),
        ("How", how),
    )

    ch_rows = [[Paragraph("<b>File / Location</b>", body_sm),
                Paragraph("<b>Change required</b>", body_sm)]]
    for f, ch in changes:
        ch_rows.append([Paragraph(f"<font face='Courier' size='7.5'>{f}</font>", body_sm),
                        Paragraph(ch, body_sm)])
    ch_tbl = Table(ch_rows, colWidths=[68*mm, PAGE_W - 2*M - 68*mm])
    ch_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, CARD_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d4e8")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))

    return KeepTogether([
        header, Spacer(1, 2), meta, Spacer(1, 3),
        body_tbl, Spacer(1, 3),
        Paragraph("<b>Files to change</b>", body_sm),
        ch_tbl, Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d0d4e8")),
        Spacer(1, 8),
    ])

# ── Cover page ────────────────────────────────────────────────────────────────
def build_cover(doc):
    story = []

    # dark banner
    story.append(FullWidthRect(52, DARK))
    story.append(Spacer(1, -52))
    story.append(Paragraph("PyFlowML", cover_title))
    story.append(Paragraph("v1.0.4  →  v2.0  Implementation Plan", cover_sub))
    story.append(Paragraph(f"Prepared {date.today().strftime('%B %d, %Y')}  ·  Confidential", cover_badge))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HIGHLIGHT))
    story.append(Spacer(1, 10))

    # Intro paragraph
    story.append(Paragraph(
        "This document is the complete planning reference for the next major version of PyFlowML. "
        "It covers every gap identified in the v1.0.4 audit, the nine upgrade ideas proposed, "
        "and the exact files, APIs, and code changes required. "
        "Upgrades are grouped into four sequential phases so the team can ship value incrementally "
        "without breaking existing users.",
        body
    ))
    story.append(Spacer(1, 8))

    # Key numbers
    kn = Table([
        [Paragraph("<b>9</b>", h1), Paragraph("<b>4</b>", h1),
         Paragraph("<b>12</b>", h1), Paragraph("<b>~6 wks</b>", h1)],
        [Paragraph("Upgrades", body_sm), Paragraph("Phases", body_sm),
         Paragraph("Files changed", body_sm), Paragraph("Total effort", body_sm)],
    ], colWidths=[(PAGE_W - 2*M)/4]*4)
    kn.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS",[8]),
    ]))
    story.append(kn)
    story.append(Spacer(1, 12))

    # Priority legend
    story.append(Paragraph("<b>Priority legend</b>", h3))
    leg = Table([[
        Paragraph(p, label_style)
        for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ]], colWidths=[(PAGE_W - 2*M)/4]*4)
    leg.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), RED),
        ("BACKGROUND",    (1, 0), (1, 0), ORANGE),
        ("BACKGROUND",    (2, 0), (2, 0), GOLD),
        ("BACKGROUND",    (3, 0), (3, 0), TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS",[6]),
    ]))
    story.append(leg)
    story.append(PageBreak())
    return story

# ── Table of contents ─────────────────────────────────────────────────────────
def build_toc():
    story = []
    story += section_banner("Table of Contents")
    items = [
        ("1", "Project Overview & Audit Summary",    "3"),
        ("2", "Phase Map",                           "4"),
        ("3", "Upgrade #1 — ModelSaver.load()",      "5"),
        ("4", "Upgrade #2 — Fix GitHub URL",         "6"),
        ("5", "Upgrade #3 — Reporter.regression()",  "7"),
        ("6", "Upgrade #4 — Python 3.12/3.13 Support", "8"),
        ("7", "Upgrade #5 — CrossValidator docs",    "9"),
        ("8", "Upgrade #6 — DataCleaner ordering",   "10"),
        ("9", "Upgrade #7 — SHAP feature importance","11"),
        ("10","Upgrade #8 — HTML report export",     "12"),
        ("11","Upgrade #9 — MixedPipeline",          "13"),
        ("12","Release checklist & versioning",      "14"),
    ]
    rows = []
    for num, title, pg in items:
        rows.append([
            Paragraph(f"<b>{num}</b>", toc_num),
            Paragraph(title, toc_style),
            Paragraph(pg, toc_style),
        ])
    t = Table(rows, colWidths=[10*mm, PAGE_W - 2*M - 24*mm, 10*mm])
    t.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (0, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, colors.HexColor("#d8dae8")),
        ("ALIGN",         (2, 0), (2, -1), "RIGHT"),
    ]))
    story.append(t)
    story.append(PageBreak())
    return story

# ── Section 1 — Overview ──────────────────────────────────────────────────────
def build_overview():
    story = []
    story += section_banner("1. Project Overview & Audit Summary",
                            "What we found in v1.0.4 and why it matters")

    story.append(Paragraph("Library summary", h2))
    story.append(Paragraph(
        "PyFlowML is an AutoML library that covers the full machine-learning lifecycle: "
        "data loading, cleaning, preprocessing, model training, evaluation, visualisation, "
        "and persistence. Released in April 2026, version 1.0.4 is the first stable beta. "
        "The architecture is sound — parallel training via joblib, a cached SmartPipeline, "
        "dtype-downcast memory savings, and a time-budgeted AutoClassifier are all "
        "industry-correct patterns.", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Audit findings", h2))
    findings = [
        ("PASS",   "One-liner PyFlowEngine",     "Clean API, works on a single DataFrame"),
        ("PASS",   "MemoryOptimizer.reduce()",   "40–60% RAM savings via dtype downcasting"),
        ("PASS",   "SmartPipeline caching",      "No data leakage — fits on train set only"),
        ("PASS",   "time_limit in AutoClassifier","Budget-aware training prevents runaway jobs"),
        ("REVIEW", "DataCleaner chain order",    "Order-dependent; docs don't specify correct sequence"),
        ("REVIEW", "AutoClusterer + target col", "Unsupervised models shouldn't receive a target — no warning"),
        ("REVIEW", "NLP pipeline integration",   "TextCleaner/Vectorizer not wired into SmartPipeline"),
        ("GAP",    "ModelSaver.load() missing",  "Save exists, load does not — models can't be reused"),
        ("GAP",    "GitHub URL is placeholder",  "yourusername/pyflowml breaks bug tracker & source access"),
        ("GAP",    "Reporter.regression()",      "No RMSE/MAE/R² path for AutoRegressor users"),
        ("GAP",    "Python 3.12/3.13 classifiers","Not declared — users on modern Python have no guarantee"),
        ("GAP",    "CrossValidator undocumented", "Module exists but never shown in examples"),
    ]
    status_colors = {"PASS": GREEN, "REVIEW": GOLD, "GAP": RED}
    rows = [[
        Paragraph("<b>Status</b>", label_style),
        Paragraph("<b>Area</b>",   label_style),
        Paragraph("<b>Finding</b>",label_style),
    ]]
    for status, area, finding in findings:
        rows.append([
            Paragraph(status, label_style),
            Paragraph(area,   body_sm),
            Paragraph(finding,body_sm),
        ])
    t = Table(rows, colWidths=[20*mm, 52*mm, PAGE_W - 2*M - 76*mm])
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), DARK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, CARD_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d4e8")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, (status, _, _) in enumerate(findings, start=1):
        style_cmds.append(("BACKGROUND", (0, i), (0, i), status_colors[status]))
    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(PageBreak())
    return story

# ── Section 2 — Phase map ─────────────────────────────────────────────────────
def build_phasemap():
    story = []
    story += section_banner("2. Phase Map", "Four sequential releases — ship value incrementally")

    phases = [
        (1, HIGHLIGHT, "Phase 1 — Critical fixes",  "Week 1–2",
         ["#1 ModelSaver.load()", "#2 Fix GitHub URL", "#3 Reporter.regression()", "#4 Python 3.12/3.13 classifiers"],
         "v1.1.0"),
        (2, TEAL,      "Phase 2 — Documentation & DX","Week 2–3",
         ["#5 CrossValidator integration docs", "#6 DataCleaner ordering + strict_order flag"],
         "v1.2.0"),
        (3, BLUE,      "Phase 3 — New features",      "Week 3–5",
         ["#7 SHAP feature importance (ModelViz)", "#8 HTML report export (Reporter.export_html)"],
         "v1.3.0"),
        (4, GOLD,      "Phase 4 — Advanced pipeline",  "Week 5–6",
         ["#9 MixedPipeline for tabular + text"],
         "v2.0.0"),
    ]

    for ph, col, title, timeline, items, ver in phases:
        rows = [[Paragraph(f"<b>{title}</b>", section_hdr),
                 Paragraph(timeline, label_style),
                 Paragraph(ver, label_style)]]
        t_hdr = Table(rows, colWidths=[PAGE_W - 2*M - 44*mm, 24*mm, 16*mm])
        t_hdr.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), col),
            ("BACKGROUND", (1, 0), (1, 0), ACCENT),
            ("BACKGROUND", (2, 0), (2, 0), DARK),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (0, 0), 12),
            ("ROUNDEDCORNERS",[6]),
        ]))
        item_rows = [[Paragraph(f"• {i}", body)] for i in items]
        t_items = Table(item_rows, colWidths=[PAGE_W - 2*M])
        t_items.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ]))
        story.append(t_hdr)
        story.append(t_items)
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    return story

# ── Upgrade detail pages ──────────────────────────────────────────────────────
def build_upgrades():
    story = []
    story += section_banner("3–11. Upgrade Detail Cards",
                            "Every upgrade: what, why, how, and which files to touch")

    upgrades = [
        dict(
            num=1, title="ModelSaver.load() — complete the save/load cycle",
            priority="CRITICAL", phase=1,
            module="pyflowml/utils/model_saver.py", effort="~1 day",
            what="Add ModelSaver.load(name) that restores a saved model and its metadata dict.",
            why="Without load(), saved models are write-only. Production inference is impossible.",
            how="Use joblib.load() mirroring the save() pattern. Return (model, metadata) tuple.",
            changes=[
                ("pyflowml/utils/model_saver.py", "Add load() classmethod alongside save()"),
                ("pyflowml/utils/__init__.py",     "Export load in public API"),
                ("tests/test_model_saver.py",      "Add round-trip save→load test"),
                ("README.md / docs quickstart",    "Show load() usage after save() example"),
            ]
        ),
        dict(
            num=2, title="Fix GitHub URL — replace placeholder with real repo",
            priority="CRITICAL", phase=1,
            module="pyproject.toml / setup.cfg", effort="30 min",
            what="Replace yourusername/pyflowml with the real GitHub username in all project metadata.",
            why="PyPI shows a dead link. Users cannot file bugs or read source code.",
            how="Update project.urls in pyproject.toml, re-publish. Also update README badge links.",
            changes=[
                ("pyproject.toml",  "Set [project.urls] Homepage/Repository to real URL"),
                ("setup.cfg",       "If used — update url= field"),
                ("README.md",       "Update badge URLs and repo links"),
            ]
        ),
        dict(
            num=3, title="Reporter.regression() — evaluation for AutoRegressor",
            priority="HIGH", phase=1,
            module="pyflowml/evaluation/reporter.py", effort="~1.5 days",
            what="Add Reporter.regression(clf, X_test, y_test) that prints RMSE, MAE, R² and plots residuals.",
            why="AutoRegressor exists but there is no documented evaluation path for regression tasks.",
            how="Compute sklearn metrics. Use ModelViz to render residual scatter + predicted-vs-actual chart.",
            changes=[
                ("pyflowml/evaluation/reporter.py",   "Add regression() classmethod"),
                ("pyflowml/visualization/model_viz.py","Add residual_plot() and predicted_vs_actual()"),
                ("tests/test_reporter.py",             "Add regression reporter tests on synthetic data"),
                ("README.md",                          "Add regression quickstart block"),
            ]
        ),
        dict(
            num=4, title="Python 3.12 / 3.13 classifiers in package metadata",
            priority="HIGH", phase=1,
            module="pyproject.toml", effort="2 hours",
            what="Add Python 3.12 and 3.13 to the Programming Language classifiers after running the test suite.",
            why="These are now the dominant Python versions. Missing classifiers signal neglect and hide the package in searches.",
            how="Run pytest on 3.12 and 3.13 in CI (tox or GitHub Actions matrix). If green, add classifiers and bump patch version.",
            changes=[
                ("pyproject.toml",           "Add Python :: 3.12 and Python :: 3.13 classifiers"),
                (".github/workflows/ci.yml", "Add 3.12, 3.13 to the python-version matrix"),
                ("tox.ini (if present)",     "Add py312, py313 environments"),
            ]
        ),
        dict(
            num=5, title="CrossValidator — document integration with AutoClassifier",
            priority="HIGH", phase=2,
            module="pyflowml/evaluation/cross_validator.py + docs", effort="~1 day",
            what="Show how CrossValidator plugs in after fit(). Add a quickstart block and docstrings.",
            why="The module exists but is invisible. Users don't know when or how to use it alongside the engine.",
            how="Add CrossValidator.score(clf, X, y, cv=5) example to README and to PyFlowEngine internals as an optional step.",
            changes=[
                ("pyflowml/evaluation/cross_validator.py","Add/improve docstrings and score() method"),
                ("pyflowml/core/engine.py",               "Call CrossValidator optionally when cv=True is passed to run()"),
                ("README.md",                             "Add CV quickstart block after AutoClassifier example"),
                ("tests/test_cross_validator.py",         "Add integration test"),
            ]
        ),
        dict(
            num=6, title="DataCleaner — explicit ordering docs + strict_order flag",
            priority="MEDIUM", phase=2,
            module="pyflowml/data/data_cleaner.py", effort="~4 hours",
            what="Document the recommended chain order (duplicates → nulls → outliers). Add strict_order=True that raises if called out of order.",
            why="Method chaining is order-dependent and produces different results depending on sequence. Silent wrong results are the worst failure mode.",
            how="Track call order internally with a list. In strict mode, assert sequence matches recommended order and raise ValueError with a helpful message.",
            changes=[
                ("pyflowml/data/data_cleaner.py","Add _call_order tracking; strict_order param to __init__"),
                ("README.md",                    "Document recommended chain order with a note"),
                ("tests/test_data_cleaner.py",   "Add strict_order=True tests and wrong-order test"),
            ]
        ),
        dict(
            num=7, title="SHAP feature importance — ModelViz.feature_importance()",
            priority="MEDIUM", phase=3,
            module="pyflowml/visualization/model_viz.py", effort="~2 days",
            what="Add ModelViz.feature_importance(clf, X, method='shap') that renders a bar chart of global importances.",
            why="Feature importance is the #1 most-requested capability in AutoML tools. SHAP is the industry standard.",
            how="Try shap.Explainer first; fall back to clf.feature_importances_ for tree models and permutation importance for others. Render with matplotlib using the existing dark theme.",
            changes=[
                ("pyflowml/visualization/model_viz.py","Add feature_importance() classmethod"),
                ("setup.cfg / pyproject.toml",         "Add shap as optional dependency in [project.optional-dependencies] boosting"),
                ("README.md",                          "Add feature importance usage example"),
                ("tests/test_model_viz.py",            "Add smoke test — check plot produced without exception"),
            ]
        ),
        dict(
            num=8, title="HTML report export — Reporter.export_html()",
            priority="MEDIUM", phase=3,
            module="pyflowml/evaluation/reporter.py", effort="~2 days",
            what="Add Reporter.export_html(clf, X_test, y_test, path='report.html') that bundles metrics, confusion matrix, ROC curve, and feature importances into one shareable file.",
            why="Stakeholders rarely have Python. A single HTML file they can open in a browser is a major value-add and differentiator.",
            how="Render all plots to base64 PNG strings; embed in a Jinja2 (or f-string) HTML template. Output a self-contained file with no external dependencies.",
            changes=[
                ("pyflowml/evaluation/reporter.py",    "Add export_html() classmethod"),
                ("pyflowml/evaluation/templates/",     "Add report.html Jinja2 template (or inline f-string)"),
                ("setup.cfg / pyproject.toml",         "Add jinja2 as optional dep in [dev] or [all]"),
                ("tests/test_reporter.py",             "Add test that output file is valid HTML"),
                ("README.md",                          "Add export_html() example"),
            ]
        ),
        dict(
            num=9, title="MixedPipeline — tabular + text columns in one pipeline",
            priority="LOW", phase=4,
            module="pyflowml/preprocessing/mixed_pipeline.py", effort="~4 days",
            what="Create MixedPipeline that accepts a DataFrame with numeric and text columns, applies SmartPipeline to numeric and Vectorizer to text, then merges features before model training.",
            why="Real-world datasets almost always mix structured and unstructured data. No existing AutoML tool in the library handles this case.",
            how="Use sklearn ColumnTransformer under the hood. Auto-detect column types (is_object → text branch; is_numeric → numeric branch). Expose the same fit_transform/transform API as SmartPipeline.",
            changes=[
                ("pyflowml/preprocessing/mixed_pipeline.py","New file — MixedPipeline class"),
                ("pyflowml/preprocessing/__init__.py",      "Export MixedPipeline"),
                ("pyflowml/nlp/vectorizer.py",              "Ensure Vectorizer returns a sparse/dense compatible with ColumnTransformer"),
                ("tests/test_mixed_pipeline.py",            "Test on synthetic DataFrame with text + numeric cols"),
                ("README.md",                               "Add MixedPipeline quickstart example"),
            ]
        ),
    ]

    for upg in upgrades:
        story.append(upgrade_card(**upg))

    return story

# ── Section 12 — Release checklist ───────────────────────────────────────────
def build_checklist():
    story = []
    story.append(PageBreak())
    story += section_banner("12. Release Checklist & Versioning")

    story.append(Paragraph("Versioning strategy", h2))
    story.append(Paragraph(
        "PyFlowML follows Semantic Versioning (SemVer). Bug fixes and metadata-only changes "
        "increment the patch (1.0.x). New backward-compatible features increment the minor (1.x.0). "
        "Breaking API changes increment the major (x.0.0). The MixedPipeline in Phase 4 warrants "
        "a 2.0.0 bump if it changes any existing public interface.", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Per-phase checklist", h2))
    phases_chk = [
        ("Phase 1 — v1.1.0", [
            "ModelSaver.load() implemented and tested",
            "GitHub URL updated in pyproject.toml + README",
            "Reporter.regression() + residual plots done",
            "CI matrix includes Python 3.12 and 3.13",
            "All existing tests pass on 3.8–3.13",
            "Changelog updated",
            "pypi publish via twine / trusted publishing",
        ]),
        ("Phase 2 — v1.2.0", [
            "CrossValidator.score() documented and tested",
            "PyFlowEngine accepts cv=True kwarg",
            "DataCleaner strict_order flag added",
            "Ordering docs in README",
            "All tests green",
        ]),
        ("Phase 3 — v1.3.0", [
            "ModelViz.feature_importance() with SHAP fallback",
            "Reporter.export_html() producing valid single-file HTML",
            "shap added as optional dep",
            "Smoke tests for both new visualisations",
        ]),
        ("Phase 4 — v2.0.0", [
            "MixedPipeline class implemented",
            "Vectorizer compatible with ColumnTransformer",
            "Integration tests on synthetic mixed DataFrame",
            "README quickstart for MixedPipeline",
            "CHANGELOG notes breaking changes if any",
            "Version bump to 2.0.0",
        ]),
    ]
    for phase_title, items in phases_chk:
        story.append(Paragraph(phase_title, h3))
        chk_rows = [[
            Paragraph("☐", body),
            Paragraph(item, body_sm)
        ] for item in items]
        t = Table(chk_rows, colWidths=[8*mm, PAGE_W - 2*M - 8*mm])
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (0, -1), 4),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.3, colors.HexColor("#e0e4f0")),
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This document will be updated as implementation progresses. "
        "Each upgrade card maps directly to a GitHub issue and PR.",
        note_style
    ))
    return story

# ── Page template (header/footer) ─────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(DARK)
    canvas.rect(M, PAGE_H - 14*mm, PAGE_W - 2*M, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawString(M + 4, PAGE_H - 10.5*mm, "PyFlowML — Implementation Plan v1.0.4 → v2.0")
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(PAGE_W - M - 4, PAGE_H - 10.5*mm, f"Page {doc.page}")
    # Footer
    canvas.setFillColor(ACCENT)
    canvas.rect(M, 8*mm, PAGE_W - 2*M, 1.5*mm, fill=1, stroke=0)
    canvas.setFillColor(TEXT_LIGHT)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(M, 5*mm, f"Confidential  ·  {date.today().strftime('%Y-%m-%d')}")
    canvas.drawRightString(PAGE_W - M, 5*mm, "github.com/yourusername/pyflowml")
    canvas.restoreState()

# ── Assemble and build ────────────────────────────────────────────────────────
def main():
    # Use relative path from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, "PyFlowML_Implementation_Plan.pdf")
    
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=16*mm, bottomMargin=16*mm,
        title="PyFlowML Implementation Plan",
        author="PyFlowML Contributors",
    )

    story = []
    story += build_cover(doc)
    story += build_toc()
    story += build_overview()
    story += build_phasemap()
    story += build_upgrades()
    story += build_checklist()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"✓ PDF created successfully: {out_path}")

if __name__ == "__main__":
    main()

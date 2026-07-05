# -*- coding: utf-8 -*-
"""Export internship resume PDF to Desktop."""
from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from resume_content import RESUME  # noqa: E402

DESKTOP = Path.home() / "Desktop"
OUTPUT = DESKTOP / "实习简历-AI Agent方向.pdf"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
]


def find_font() -> Path:
    for p in FONT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("未找到中文字体")


class ResumePDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("resume", size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"- {self.page_no()} -", align="C")


def section_title(pdf: ResumePDF, title: str) -> None:
    pdf.ln(3)
    pdf.set_font("resume", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(60, 60, 60)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(4)


def body_text(pdf: ResumePDF, text: str) -> None:
    pdf.set_font("resume", size=10.5)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 6, text)


def bullet(pdf: ResumePDF, text: str, prefix: str = "") -> None:
    pdf.set_font("resume", size=10.5)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(pdf.l_margin)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(width, 6, f"{prefix}{text}")


def build_pdf() -> Path:
    pdf = ResumePDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("resume", "", str(find_font()))

    pdf.set_font("resume", size=18)
    pdf.cell(0, 10, RESUME["name"], new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("resume", size=10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5.5, RESUME["contact"], align="C")
    pdf.ln(1)
    pdf.multi_cell(0, 5.5, RESUME["meta"], align="C")
    pdf.ln(2)

    section_title(pdf, "教育背景")
    for line in RESUME["education"]:
        body_text(pdf, line)

    section_title(pdf, "专业技能")
    for line in RESUME["skills"]:
        body_text(pdf, line)

    section_title(pdf, "项目经历")
    pdf.set_font("resume", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, RESUME["project_title"], new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("resume", size=10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, RESUME["project_time"], new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, RESUME["project_stack"], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("resume", size=10.5)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 6, "项目描述：" + RESUME["project_desc"])
    pdf.ln(1)
    pdf.cell(0, 6, "主要工作：", new_x="LMARGIN", new_y="NEXT")
    for i, item in enumerate(RESUME["project_work"], 1):
        bullet(pdf, item, prefix=f"{i}. ")
    pdf.ln(1)
    pdf.multi_cell(0, 6, "个人收获：" + RESUME["project_gain"])
    pdf.ln(1)
    pdf.set_text_color(60, 60, 120)
    pdf.multi_cell(0, 6, RESUME["project_link"])

    section_title(pdf, "荣誉奖项及其它经历")
    for line in RESUME["honors"]:
        body_text(pdf, line)

    DESKTOP.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    path = build_pdf()
    print(f"已生成: {path}")

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "docx_to_pdf"))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from build_docx_lajss import fill
from docx_to_pdf import convert

SOURCE = os.path.join(HERE, "cover_letter_9169.md")
OUT = os.path.join(HERE, "cover_letter_9169.docx")
FONT = "Calibri"
COMMENT = re.compile(r'^(Comment \w+\.) "(.*)"$')
SIGNATURE_START = "Sincerely,"


def finish(paragraph, size=11):
    for run in paragraph.runs:
        run.font.name = FONT
        run.font.size = Pt(size)
    return paragraph


def build():
    with open(SOURCE, encoding="utf-8") as fh:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", fh.read()) if b.strip()]
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Cm(2.5)
    section.top_margin = section.bottom_margin = Cm(2.5)
    signing = False
    for i, block in enumerate(blocks):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        if i == 0:
            finish(fill(p, block), 12).runs[0].bold = True
            continue
        if block in ("Response to the reviewer", "Other corrections"):
            run = p.add_run(block)
            run.bold = True
            finish(p, 12 if i == 1 else 11)
            continue
        if block == SIGNATURE_START:
            signing = True
        comment = COMMENT.match(block)
        if comment:
            label = p.add_run(comment.group(1) + " ")
            label.bold = True
            quoted = p.add_run('"' + comment.group(2) + '"')
            quoted.italic = True
            p.paragraph_format.space_after = Pt(4)
            finish(p)
            continue
        if signing:
            for j, line in enumerate(block.splitlines()):
                if j:
                    p.add_run().add_break()
                p.add_run(line)
            finish(p)
            continue
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        finish(fill(p, block))
    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Wrote {convert(OUT, os.path.splitext(OUT)[0] + '.pdf')}")


if __name__ == "__main__":
    build()

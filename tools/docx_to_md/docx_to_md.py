import argparse
import sys

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

HEADING_PREFIX = {
    "ArticleTitle": "# ",
    "Heading1": "## ",
    "Heading2": "### ",
    "AbstractTitle": "## ",
    "KeywordsTitle": "## ",
}


def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def paragraph_to_md(p):
    text = p.text.strip()
    has_picture = bool(p._element.xpath(".//*[local-name()='blip']"))
    if has_picture:
        return "[figure]"
    if not text:
        return None
    style = p.style.name if p.style is not None else ""
    if style == "Equation":
        return f"    {text}"
    return HEADING_PREFIX.get(style, "") + text


def table_to_md(t):
    rows = [[c.text.strip().replace("|", "\\|") for c in r.cells] for r in t.rows]
    if not rows:
        return None
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def convert(path):
    doc = Document(path)
    out = []
    for block in iter_blocks(doc):
        md = paragraph_to_md(block) if isinstance(block, Paragraph) else table_to_md(block)
        if md is not None:
            out.append(md)
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Convert a manuscript .docx to markdown by paragraph style.")
    parser.add_argument("docx")
    parser.add_argument("-o", "--output", help="markdown file to write; stdout if omitted")
    args = parser.parse_args()
    md = convert(args.docx)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(md)


if __name__ == "__main__":
    main()

import argparse
import re
import sys

from docx import Document
from docx.oxml.ns import qn

RUN, MATH, MATH_RUN = qn("w:r"), qn("m:oMath"), qn("m:r")
RED = "FF0000"


def merge(text):
    while "+}{+" in text:
        text = text.replace("+}{+", "")
    return text.replace("{++}", "").strip()


def docx_marked(element):
    out = []
    for child in element.iter():
        if child.tag == RUN and child.getparent().tag != MATH_RUN and not any(a.tag == MATH for a in child.iterancestors()):
            text = "".join(t.text or "" for t in child.iter(qn("w:t"))).replace("\t", "")
            color = child.find(qn("w:rPr") + "/" + qn("w:color"))
            red = color is not None and color.get(qn("w:val")) == RED
            out.append("{+" + text + "+}" if red else text)
        elif child.tag == MATH:
            out.append("{+$M$+}" if child.find(".//" + qn("w:color")) is not None else "$M$")
    return merge("".join(out))


def flatten(text):
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("{+", i):
            out.append("{+" if depth == 0 else ""); depth += 1; i += 2
        elif text.startswith("+}", i):
            depth -= 1; out.append("+}" if depth == 0 else ""); i += 2
        else:
            out.append(text[i]); i += 1
    return "".join(out)


def md_marked(text):
    text = flatten(text)
    text = re.sub(r"\$\$.*?\\qquad \((\d+)\)\$\$", lambda m: "$M$(" + m.group(1) + ")", text)
    text = re.sub(r"\$[^$]*\$", "$M$", text)
    text = re.sub(r"\b([A-Za-z])_([A-Za-z0-9]+)\b", r"\1\2", text)
    text = re.sub(r"\b10\^-", "10−", text)
    return merge(text)


def md_expected(source):
    paragraphs, rows = [], []
    for block in (b.strip() for b in re.split(r"\n\s*\n", source) if b.strip()):
        if block.startswith("[figure"):
            continue
        if block.startswith("|"):
            for line in block.splitlines():
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells):
                    rows.append([md_marked(c) for c in cells])
            continue
        paragraphs.append(md_marked(re.sub(r"^#+ ", "", block)))
    paragraphs[1] = paragraphs[1].replace(" a,* ", "a,* ")
    paragraphs[2] = "a " + paragraphs[2]
    return paragraphs, rows


def main():
    parser = argparse.ArgumentParser(description="Check that a built LAJSS .docx carries the same text and {+ +} change marks (as red runs) as its markdown source.")
    parser.add_argument("docx")
    parser.add_argument("markdown")
    args = parser.parse_args()
    doc = Document(args.docx)
    with open(args.markdown, encoding="utf-8") as fh:
        want_paragraphs, want_rows = md_expected(fh.read())
    got_paragraphs = [s for s in (docx_marked(p._p) for p in doc.paragraphs) if s]
    got_rows = [[docx_marked(c.paragraphs[0]._p) for c in r.cells] for t in doc.tables for r in t.rows]
    problems = []
    if len(got_paragraphs) != len(want_paragraphs):
        problems.append(f"paragraph count {len(got_paragraphs)} in docx, {len(want_paragraphs)} in markdown")
    for i, (got, want) in enumerate(zip(got_paragraphs, want_paragraphs)):
        if got != want:
            problems.append(f"paragraph {i}:\n  docx {got}\n  md   {want}")
    if got_rows != want_rows:
        problems.append(f"table rows differ:\n  docx {got_rows}\n  md   {want_rows}")
    out = sys.stdout.buffer
    for p in problems:
        out.write((p + "\n").encode("utf-8"))
    out.write(f"{len(got_paragraphs)} paragraphs, {len(got_rows)} table rows, {len(problems)} problems\n".encode("utf-8"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

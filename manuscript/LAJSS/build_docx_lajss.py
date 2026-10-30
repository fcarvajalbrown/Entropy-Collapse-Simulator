import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANU = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(MANU, ".."))
sys.path.insert(0, MANU)
sys.path.insert(0, os.path.join(ROOT, "tools", "latex_to_omml"))

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from latex_to_omml import latex_to_omath, latex_to_omath_para
from results import compute_results

TEMPLATE = os.path.join(HERE, "LAJSS - Template.docx")
SOURCE = os.path.join(HERE, "revision_9169.md")
FIG = os.path.join(MANU, "figures")
OUT = os.path.join(HERE, "Carvajal_LAJSS_9169_revised.docx")

ABSTRACT_MAX_CHARS = 1140
RED_HEX = "FF0000"
RED = RGBColor(0xFF, 0x00, 0x00)
FIGURE_WIDTH = Cm(13.0)
WIDE_FIGURES = {"fig_benchmark_geometry.png": Cm(17.0)}
TEXT_WIDTH_CM = 17.0
EQUATION_NUMBER_CM = 1.5
EQUATION_TAG = "equation"
STATEMENT_FONT = "Calibri"
STATEMENT_SIZE = Pt(10.5)

FIGURES_IN_ORDER = [
    "fig_robustness_ranking.png",
    "fig_redundancy_sweep.png",
    "fig_importance_scatter.png",
    "fig_importance_ensemble.png",
    "fig_entropy_trajectory.png",
    "fig_criteria_comparison.png",
]

STATEMENT_STYLES = [
    ("Author's Contributions:", "fnAuthor-notesContributions"),
    ("Data Availability:", "dataStatement"),
    ("Conflicts of Interest:", "dataStatement"),
    ("Funding:", "dataStatement"),
    ("Editor:", "fnEdited-by"),
]

SYMBOL = re.compile(r"\b([A-Za-z])_([A-Za-z0-9]+)\b|\b(10)\^(-?\d+)|\b(H)(0)\b")
CAPTION_LABEL = re.compile(r"^(Figure|Table) \d+")
EQUATION_NUMBER = re.compile(r"\\qquad\s*\((\d+)\)\s*$")


def strip_marks(text):
    return text.replace("{+", "").replace("+}", "")


def plain(text):
    return re.sub(r"\$([^$]*)\$", r"\1", strip_marks(text))


def segments(text):
    out, depth, i, buf = [], 0, 0, []

    def flush():
        if buf:
            out.append(("text", "".join(buf), depth > 0))
            buf.clear()

    while i < len(text):
        if text.startswith("{+", i):
            flush(); depth += 1; i += 2
        elif text.startswith("+}", i):
            flush(); depth -= 1; i += 2
        elif text[i] == "$":
            flush()
            end = text.index("$", i + 1)
            out.append(("math", text[i + 1:end], depth > 0))
            i = end + 1
        else:
            buf.append(text[i]); i += 1
    flush()
    if depth != 0:
        raise SystemExit(f"Unbalanced change marks in: {text[:80]}")
    return out


def add_run(paragraph, text, red, bold=False, italic=False, vert=None):
    run = paragraph.add_run(text)
    if red:
        run.font.color.rgb = RED
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if vert == "sub":
        run.font.subscript = True
    elif vert == "sup":
        run.font.superscript = True
    return run


def add_text(paragraph, text, red, bold_until, offset):
    pos = 0
    pieces = []
    for match in SYMBOL.finditer(text):
        pieces.append((text[pos:match.start()], None, False))
        if match.group(1):
            pieces.append((match.group(1), None, True))
            pieces.append((match.group(2), "sub", True))
        elif match.group(3):
            pieces.append((match.group(3), None, False))
            pieces.append((match.group(4).replace("-", "\u2212"), "sup", False))
        else:
            pieces.append((match.group(5), None, True))
            pieces.append((match.group(6), "sub", False))
        pos = match.end()
    pieces.append((text[pos:], None, False))
    for piece, vert, italic in pieces:
        if not piece:
            continue
        cut = max(0, min(len(piece), bold_until - offset))
        if cut:
            add_run(paragraph, piece[:cut], red, True, italic, vert)
        if cut < len(piece):
            add_run(paragraph, piece[cut:], red, False, italic, vert)
        offset += len(piece)
    return offset


def fill(paragraph, text, bold_label=False):
    bold_until = 0
    if bold_label:
        label = CAPTION_LABEL.match(plain(text))
        bold_until = label.end() if label else 0
    offset = 0
    for kind, content, red in segments(text):
        if kind == "text":
            offset = add_text(paragraph, content, red, bold_until, offset)
        else:
            paragraph._p.append(latex_to_omath(content, RED_HEX if red else None))
            offset += len(content)
    return paragraph


def styled(doc, style):
    names = {s.name for s in doc.styles}
    return doc.add_paragraph(style=style if style in names else None)


def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}p") or child.tag.endswith("}tbl"):
            body.remove(child)


def borderless(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "nil")
        borders.append(el)
    later = [c for c in tbl_pr if c.tag in (qn("w:shd"), qn("w:tblLayout"), qn("w:tblCellMar"), qn("w:tblLook"))]
    if later:
        later[0].addprevious(borders)
    else:
        tbl_pr.append(borders)
    caption = OxmlElement("w:tblCaption")
    caption.set(qn("w:val"), EQUATION_TAG)
    tbl_pr.append(caption)


def add_equation(doc, block):
    red = block.startswith("{+") and block.endswith("+}")
    latex = strip_marks(block).strip()[2:-2].strip()
    number = EQUATION_NUMBER.search(latex)
    if not number:
        raise SystemExit(f"Display equation without a number: {latex}")
    latex = latex[:number.start()].strip()
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borderless(t)
    widths = (Cm(TEXT_WIDTH_CM - EQUATION_NUMBER_CM), Cm(EQUATION_NUMBER_CM))
    for cell, width in zip(t.rows[0].cells, widths):
        cell.width = width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for cell in t.rows[0].cells:
        fmt = cell.paragraphs[0]
        fmt.style = doc.styles["Equation"]
        fmt.paragraph_format.space_before = Pt(4)
        fmt.paragraph_format.space_after = Pt(4)
    math = t.rows[0].cells[0].paragraphs[0]
    math.paragraph_format.left_indent = Cm(1.25)
    math._p.append(latex_to_omath_para(latex, RED_HEX if red else None, "left"))
    tag = t.rows[0].cells[1].paragraphs[0]
    tag.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(tag, f"({number.group(1)})", red)


def add_author(doc, block):
    name, rest = block.split(" a,* ", 1)
    p = styled(doc, "Authors")
    add_run(p, name, False)
    add_run(p, "a,*", False, vert="sup")
    add_run(p, " " + rest, False)


def add_affiliation(doc, block):
    p = styled(doc, "Affiliations")
    add_run(p, "a", False, vert="sup")
    add_run(p, " ", False)
    fill(p, block)


def set_cell_borders(cell, edges):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement("w:" + edge)
        if edge in edges:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), edges[edge]); el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    tc_pr.append(borders)


def table_rows(block):
    rows = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"-+", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def column_widths(rows):
    longest = [max(len(plain(row[j])) for row in rows) for j in range(len(rows[0]))]
    weights = [max(8, min(n, 40)) for n in longest]
    total = sum(weights)
    return [Cm(TEXT_WIDTH_CM * w / total) for w in weights]


def add_table(doc, block):
    rows = table_rows(block)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = column_widths(rows)
    for row in t.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    last = len(rows) - 1
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = t.rows[i].cells[j]
            p = cell.paragraphs[0]
            p.style = doc.styles["TableHead" if i == 0 else "TableBody"]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fill(p, value)
            if i == 0:
                for run in p.runs:
                    run.bold = True
            edges = {}
            if i == 0:
                edges.update(top="12", bottom="6")
            if i == last:
                edges["bottom"] = "12"
            set_cell_borders(cell, edges)
    doc.add_paragraph()
    return rows


def add_figure(doc, filename):
    path = os.path.join(FIG, filename)
    if not os.path.exists(path):
        raise SystemExit(f"Missing figure {path}")
    p = styled(doc, "Figure")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=WIDE_FIGURES.get(filename, FIGURE_WIDTH))


def add_statement(doc, block):
    prefix, style = next((prefix, style) for prefix, style in STATEMENT_STYLES if block.startswith(prefix))
    p = styled(doc, style)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    add_run(p, prefix, False, bold=True)
    fill(p, block[len(prefix):])
    for run in p.runs:
        run.font.name = STATEMENT_FONT
        run.font.size = STATEMENT_SIZE


def blocks_of(text):
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def render(doc, blocks):
    figures = iter(FIGURES_IN_ORDER)
    tables = {}
    mode = None
    pending_table_title = None
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if i == 0 and block.startswith("# "):
            fill(styled(doc, "ArticleTitle"), block[2:])
        elif i == 1:
            add_author(doc, block)
        elif i == 2:
            add_affiliation(doc, block)
        elif block == "* Corresponding author":
            fill(styled(doc, "Correspondence"), block)
        elif block in ("## Abstract", "## Keywords"):
            fill(styled(doc, block[3:] + "Title"), block[3:])
            mode = block[3:]
        elif mode in ("Abstract", "Keywords"):
            fill(styled(doc, mode), block)
            mode = None
        elif block.startswith("### "):
            fill(styled(doc, "Heading2"), block[4:])
        elif block.startswith("## "):
            fill(styled(doc, "Heading1"), block[3:])
        elif strip_marks(block).startswith("$$"):
            add_equation(doc, block)
        elif block.startswith("|"):
            tables[pending_table_title] = add_table(doc, block)
        elif re.match(r"^Table \d+", plain(block)):
            fill(styled(doc, "TableTitle"), block, bold_label=True)
            pending_table_title = re.match(r"^Table \d+", plain(block)).group(0)
        elif block.startswith("[figure"):
            named = re.match(r"\[figure: (.+)\]", block)
            add_figure(doc, named.group(1) if named else next(figures))
            i += 1
            fill(styled(doc, "FigureCaption"), blocks[i], bold_label=True)
        elif block == "References":
            fill(styled(doc, "ReferenceHead"), block)
            mode = "References"
        elif mode == "References":
            fill(styled(doc, "Reference"), block)
        elif any(block.startswith(prefix) for prefix, _ in STATEMENT_STYLES):
            add_statement(doc, block)
        else:
            p = fill(styled(doc, "Paragraph"), block)
            if plain(block).startswith("where ") and strip_marks(blocks[i - 1]).startswith("$$"):
                p.paragraph_format.first_line_indent = Cm(0)
        i += 1
    if next(figures, None) is not None:
        raise SystemExit("Fewer [figure] slots in the source than FIGURES_IN_ORDER")
    return tables


def abstract_text(blocks):
    return plain(blocks[blocks.index("## Abstract") + 1])


def sci(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa} \u00d7 10^{int(exponent)} %"


def expect(condition, message, failures):
    if not condition:
        failures.append(message)


def check_numbers(source, tables, R):
    failures = []
    text = plain(source)
    f2, f3 = (lambda x: f"{x:.2f}"), (lambda x: f"{x:.3f}")
    pct = lambda x: f"{x:.0%}"

    t1 = [[plain(c) for c in row] for row in tables["Table 1"]][1:]
    expected_t1 = [["Analytical", r["case"], sci(r["delta_err_pct"]), sci(r["U_err_pct"])] for r in R["analytical"]]
    independent = [r for r in R["independent"] if "Vogel" not in r["frame"]]
    expected_t1 += [["Independent", r["frame"], sci(r["disp_err_pct"]), sci(r["U_err_pct"])] for r in independent]
    n_pass = sum(1 for c in R["index_checks"] if c["pass"])
    expected_t1.append(["Index checks", f"distribution / ALP / failure / stability ({n_pass}/{len(R['index_checks'])} pass)", "-", "-"])
    expect(t1 == expected_t1, f"Table 1 differs from live results:\n  md   {t1}\n  live {expected_t1}", failures)
    worst = max(max(r["disp_err_pct"], r["U_err_pct"]) for r in independent + [
        dict(disp_err_pct=r["delta_err_pct"], U_err_pct=r["U_err_pct"]) for r in R["analytical"]])
    expect(worst < 4e-13, f"Largest Table 1 error {worst:.2e} is not below the 4e-13 % the text claims", failures)

    t2 = [[plain(c) for c in row] for row in tables["Table 2"]][1:]
    expected_t2 = [[name, f3(a.intact_entropy_norm), f3(a.robustness_index), f3(a.worst_case_entropy_norm), pct(a.mechanism_fraction)]
                   for name, a in (("Single-span beam", R["beam"]), ("Moment frame (2-bay, 3-story)", R["frame"]),
                                   ("Truss bridge (X-braced)", R["truss"]))]
    expect(t2 == expected_t2, f"Table 2 differs from live results:\n  md   {t2}\n  live {expected_t2}", failures)

    t3 = [[plain(c) for i, c in enumerate(row) if i != 1] for row in tables["Table 3"]][1:]
    expected_t3 = [[str(p.dsi), str(p.n_members), f3(p.robustness_index), pct(p.mechanism_fraction)] for p in R["sweep"]]
    expect(t3 == expected_t3, f"Table 3 differs from live results:\n  md   {t3}\n  live {expected_t3}", failures)

    tr = R["triggers"]
    t4 = [plain(row[1]) for row in tables["Table 4"]][1:]
    expected_t4 = [f2(tr[k]) for k in ("displacement", "dcr", "energy", "entropy")]
    expect(t4 == expected_t4, f"Table 4 differs from live results: md {t4} live {expected_t4}", failures)

    ens = R["ensemble"]
    rhos = [e.spearman_rho for e in ens]
    phrases = [
        f"R_S reads {f2(R['beam'].robustness_index)}, {f2(R['frame'].robustness_index)}, and {f2(R['truss'].robustness_index)}",
        f"Spearman rank correlation of {R['imp_frame'].spearman_rho:+.2f}".replace("+-", "-"),
        f"{R['imp_truss'].spearman_rho:+.2f} for the truss",
        f"({R['large_n'][0]} nodes, {R['large_n'][1]} members)",
        f"(R_S = {f2(R['large'].robustness_index)})",
        f"ensemble of {len(ens)} frames",
        f"ranges from {min(rhos):+.2f} to {max(rhos):+.2f} with mean {sum(rhos) / len(rhos):+.2f}",
        f"negative for {sum(1 for r in rhos if r < 0)} of the {len(ens)} frames",
        f"H_k = {f3(R['frame'].worst_case_entropy_norm)}",
        f"(H_k = {f3(R['truss'].worst_case_entropy_norm)})",
        f"energy criterion ({f2(tr['energy'])}) and one load step after the DCR criterion ({f2(tr['dcr'])})",
        f"near load factor {f2(tr['dcr'])} to {f2(tr['energy'])}",
    ]
    for phrase in phrases:
        expect(phrase in text, f"Live value not found in the source: {phrase!r}", failures)
    if failures:
        raise SystemExit("Source and live results disagree:\n" + "\n".join(failures))


def build(skip_check=False):
    with open(SOURCE, encoding="utf-8") as fh:
        source = fh.read()
    blocks = blocks_of(source)
    abstract = abstract_text(blocks)
    if len(abstract) > ABSTRACT_MAX_CHARS:
        raise SystemExit(f"Abstract is {len(abstract)} chars > LAJSS limit {ABSTRACT_MAX_CHARS}.")
    print(f"Abstract length: {len(abstract)} / {ABSTRACT_MAX_CHARS} chars with spaces")

    doc = Document(TEMPLATE)
    clear_body(doc)
    tables = render(doc, blocks)
    if not skip_check:
        check_numbers(source, tables, compute_results())
        print("Every checked number matches compute_results()")
    doc.save(OUT)
    print(f"Wrote {OUT}")


def main():
    parser = argparse.ArgumentParser(description="Build the LAJSS 9169 revised manuscript from revision_9169.md.")
    parser.add_argument("--skip-check", action="store_true", help="do not compare the source numbers with compute_results()")
    build(parser.parse_args().skip_check)


if __name__ == "__main__":
    main()

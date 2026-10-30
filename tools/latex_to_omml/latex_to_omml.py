import argparse
import sys
import xml.etree.ElementTree as ET

import latex2mathml.converter
from lxml import etree
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

MATHML = "{http://www.w3.org/1998/Math/MathML}"
NARY_CHARS = {"∑", "∏", "∫"}
FUNCTION_NAMES = {"ln", "log", "exp", "sin", "cos", "tan", "max", "min"}
NARY_STOPS = {"=", ",", "+", "−", "<", ">", "≤", "≥"}
BOLD_UPPER, BOLD_LOWER = 0x1D400, 0x1D41A


def local(el):
    return el.tag.replace(MATHML, "")


def m(tag, *kids, **attrs):
    el = OxmlElement("m:" + tag)
    for key, value in attrs.items():
        el.set(qn("m:" + key), value)
    for kid in kids:
        if isinstance(kid, list):
            for k in kid:
                el.append(k)
        elif kid is not None:
            el.append(kid)
    return el


def unbold(text):
    out, bold = [], False
    for ch in text:
        code = ord(ch)
        if BOLD_UPPER <= code < BOLD_UPPER + 26:
            out.append(chr(ord("A") + code - BOLD_UPPER)); bold = True
        elif BOLD_LOWER <= code < BOLD_LOWER + 26:
            out.append(chr(ord("a") + code - BOLD_LOWER)); bold = True
        else:
            out.append(ch)
    return "".join(out), bold


class Converter:
    def __init__(self, color=None, display=False):
        self.color = color
        self.display = display

    def run(self, text, style=None):
        r = m("r")
        if style:
            r.append(m("rPr", m("sty", val=style)))
        if self.color:
            rpr = OxmlElement("w:rPr")
            c = OxmlElement("w:color")
            c.set(qn("w:val"), self.color)
            rpr.append(c)
            r.append(rpr)
        t = m("t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        return r

    def leaf(self, el):
        tag, text = local(el), el.text or ""
        if tag == "mspace":
            return [self.run("  ")]
        if tag == "mi":
            text, bold = unbold(text)
            if bold:
                return [self.run(text, "b")]
            if len(text) > 1:
                return [self.run(text, "p")]
            return [self.run(text)]
        if tag == "mn":
            return [self.run(text, "p")]
        return [self.run(text, "p")]

    def arg(self, el):
        return self.node(el)

    def is_nary(self, el):
        return local(el) == "mo" and (el.text or "") in NARY_CHARS

    def is_function(self, el):
        return local(el) in ("mi", "mo") and (el.text or "") in FUNCTION_NAMES

    def sequence(self, kids):
        out, i = [], 0
        while i < len(kids):
            el = kids[i]
            tag = local(el)
            base = el[0] if tag in ("msub", "msup", "msubsup", "munder", "munderover", "mover") and len(el) else None
            if base is not None and self.is_nary(base):
                j = i + 1
                body = []
                while j < len(kids) and not self.stops_nary(kids[j]):
                    body.append(kids[j]); j += 1
                out.append(self.nary(el, body))
                i = j
                continue
            if self.is_function(el) and i + 1 < len(kids):
                out.append(m("func", m("fName", self.run(el.text, "p")), m("e", self.node(kids[i + 1]))))
                i += 2
                continue
            if base is not None and self.is_function(base) and tag == "msub" and i + 1 < len(kids):
                kind, slot = ("limLow", "lim") if self.display else ("sSub", "sub")
                limit = m(kind, m("e", self.run(base.text, "p")), m(slot, self.arg(el[1])))
                out.append(m("func", m("fName", limit), m("e", self.node(kids[i + 1]))))
                i += 2
                continue
            out.extend(self.node(el))
            i += 1
        return out

    def stops_nary(self, el):
        return local(el) == "mspace" or (local(el) == "mo" and (el.text or "") in NARY_STOPS)

    def nary(self, el, body):
        tag = local(el)
        sub = self.arg(el[1]) if len(el) > 1 else []
        sup = self.arg(el[2]) if tag in ("msubsup", "munderover") and len(el) > 2 else []
        props = [m("chr", val=el[0].text), m("limLoc", val="undOvr" if self.display else "subSup")]
        if not sup:
            props.append(m("supHide", val="1"))
        return m("nary", m("naryPr", props), m("sub", sub), m("sup", sup), m("e", self.sequence(body)))

    def node(self, el):
        tag = local(el)
        if tag in ("math", "mrow", "mstyle"):
            return self.sequence(list(el))
        if tag == "mfrac":
            return [m("f", m("num", self.arg(el[0])), m("den", self.arg(el[1])))]
        if tag == "msub":
            return [m("sSub", m("e", self.arg(el[0])), m("sub", self.arg(el[1])))]
        if tag == "msup":
            return [m("sSup", m("e", self.arg(el[0])), m("sup", self.arg(el[1])))]
        if tag == "msubsup":
            return [m("sSubSup", m("e", self.arg(el[0])), m("sub", self.arg(el[1])), m("sup", self.arg(el[2])))]
        return self.leaf(el)


def latex_to_omath(latex, color=None, display=False):
    tree = ET.fromstring(latex2mathml.converter.convert(latex))
    return m("oMath", Converter(color, display).node(tree))


def latex_to_omath_para(latex, color=None, justification="center"):
    return m("oMathPara", m("oMathParaPr", m("jc", val=justification)), latex_to_omath(latex, color, display=True))


def main():
    parser = argparse.ArgumentParser(description="Convert a LaTeX math string to a Word OMML <m:oMath> element.")
    parser.add_argument("latex")
    parser.add_argument("--color", default=None)
    parser.add_argument("--display", action="store_true", help="emit a display <m:oMathPara> instead of inline <m:oMath>")
    args = parser.parse_args()
    element = latex_to_omath_para(args.latex, args.color) if args.display else latex_to_omath(args.latex, args.color)
    sys.stdout.buffer.write(etree.tostring(element, pretty_print=True, encoding="utf-8"))


if __name__ == "__main__":
    main()

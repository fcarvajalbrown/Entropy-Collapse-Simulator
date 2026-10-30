import argparse
import os
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape

X2T = r"C:\Program Files\ONLYOFFICE\DesktopEditors\converter\x2t.exe"
ALL_FONTS = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ONLYOFFICE", "DesktopEditors", "data", "fonts", "AllFonts.js")
FONT_DIR = r"C:\Windows\Fonts"
PDF_FORMAT = 513


def convert(source, target, x2t=X2T, all_fonts=ALL_FONTS):
    for path, what in ((x2t, "ONLYOFFICE x2t.exe"), (all_fonts, "ONLYOFFICE AllFonts.js"), (source, "source file")):
        if not os.path.exists(path):
            raise SystemExit(f"{what} not found at {path}")
    source, target = os.path.abspath(source), os.path.abspath(target)
    with tempfile.TemporaryDirectory() as work:
        params = os.path.join(work, "params.xml")
        scratch = os.path.join(work, "tmp")
        os.makedirs(scratch)
        fields = [
            ("m_sFileFrom", source),
            ("m_sFileTo", target),
            ("m_nFormatTo", str(PDF_FORMAT)),
            ("m_sFontDir", FONT_DIR),
            ("m_sAllFontsPath", all_fonts),
            ("m_sTempDir", scratch),
        ]
        body = "".join(f"<{k}>{escape(v)}</{k}>" for k, v in fields)
        with open(params, "w", encoding="utf-8") as fh:
            fh.write(f'<?xml version="1.0" encoding="utf-8"?><TaskQueueDataConvert>{body}</TaskQueueDataConvert>')
        if os.path.exists(target):
            os.remove(target)
        result = subprocess.run([x2t, params], cwd=os.path.dirname(x2t), capture_output=True, text=True, errors="replace", timeout=600)
    if result.returncode != 0 or not os.path.exists(target):
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"x2t failed with exit code {result.returncode}")
    return target


def main():
    parser = argparse.ArgumentParser(description="Convert a .docx to PDF headless with ONLYOFFICE's x2t converter.")
    parser.add_argument("source")
    parser.add_argument("target", nargs="?", help="output PDF (default: next to the source)")
    args = parser.parse_args()
    target = args.target or os.path.splitext(args.source)[0] + ".pdf"
    print(f"Wrote {convert(args.source, target)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""md2pdf.py — chuyển markdown sang PDF không cần cài gì thêm.

Đường đi: markdown → HTML (thư viện `markdown`) + CSS in đẹp → Edge headless
"--print-to-pdf" (Edge có sẵn trên Windows). Dùng:
    python docs/md2pdf.py [file.md]      # mặc định: bài giảng TF-IDF
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', 'Times New Roman', sans-serif; font-size: 10pt;
       line-height: 1.5; color: #1c2733; margin: 0; }
h1 { font-size: 15.5pt; color: #123a5f; border-bottom: 2.5px solid #123a5f;
     padding-bottom: 4px; margin: 20px 0 10px; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; margin-top: 0; }
h2 { font-size: 12.5pt; color: #1a5276; margin: 16px 0 6px; border-bottom: 1px solid #cfd8e3;
     padding-bottom: 2px; }
h3 { font-size: 11pt; color: #21618c; margin: 12px 0 5px; }
p { margin: 5px 0; }
ul, ol { margin: 5px 0 5px 4px; padding-left: 20px; }
li { margin: 2.5px 0; }
strong { color: #0d2f4f; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt;
       background: #eef1f5; padding: 0.5px 3px; border-radius: 3px; }
pre { background: #f4f6f9; border: 1px solid #d5dce6; border-radius: 4px;
      padding: 8px 10px; margin: 7px 0; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.38; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.9pt; }
th, td { border: 1px solid #b8c4d0; padding: 3px 7px; text-align: left;
         vertical-align: top; }
th { background: #dbe7f3; color: #123a5f; }
tr:nth-child(even) td { background: #f6f9fc; }
blockquote { border-left: 3.5px solid #7fa8c9; background: #f2f7fb; margin: 8px 0;
             padding: 5px 12px; color: #33475b; }
hr { border: 0; border-top: 1px solid #c9d4e0; margin: 14px 0; }
a { color: #1a5276; text-decoration: none; }
"""

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
]


def find_edge() -> str | None:
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def md_to_html(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{md_path.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def html_to_pdf(html_path: Path, edge: str, retries: int = 2) -> Path:
    pdf = html_path.with_suffix(".pdf")
    profile = os.path.join(os.environ.get("TEMP", os.getcwd()), "dsh-edge-profile")
    for attempt in range(retries + 1):
        if pdf.exists():
            pdf.unlink()
        cmd = [edge, "--headless", "--no-sandbox", "--disable-crashpad", "--disable-gpu",
               "--no-first-run", "--no-pdf-header-footer", f"--user-data-dir={profile}",
               f"--print-to-pdf={pdf}", html_path.as_uri()]
        subprocess.run(cmd, capture_output=True, timeout=90)
        for _ in range(40):  # Edge viết file bất đồng bộ — đợi tối đa ~10s
            if pdf.exists() and pdf.stat().st_size > 10_000:
                return pdf
            time.sleep(0.25)
        # lần 2 thử chế độ headless mới (Edge mới đôi khi cần)
        cmd[1] = "--headless=new"
    raise RuntimeError(f"Edge không tạo được PDF: {pdf}")


def main() -> None:
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bai-giang-trich-xuat-skill-tfidf.md"
    md = md.resolve()
    if not md.exists():
        sys.exit(f"Không thấy file: {md}")
    edge = find_edge()
    if not edge:
        sys.exit("Không tìm thấy Microsoft Edge trên máy này.")
    html = md_to_html(md)
    pdf = html_to_pdf(html, edge)
    print(f"OK  {pdf.name}  {pdf.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

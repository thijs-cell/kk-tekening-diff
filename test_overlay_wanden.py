"""Test stap 6: wandmarkeringen in overlay voor Muiden D2.1 en 56 de Helling."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.diff_engine import run_diff
from app.overlay import generate_overlay_pdf

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")

PAREN = [
    (
        "muiden",
        os.path.join(BASE, "WT-PLG-D2.1_20250324_B.pdf"),
        os.path.join(BASE, "WT-PLG-D2.1_20260202_E.pdf"),
    ),
    (
        "helling",
        os.path.join(BASE, "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"),
        os.path.join(BASE, "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"),
    ),
]

os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)

for naam, oud, nieuw in PAREN:
    print(f"\n=== {naam.upper()} ===")
    print(f"  OUD : {os.path.basename(oud)}")
    print(f"  NIEUW: {os.path.basename(nieuw)}")

    diff = run_diff(oud, nieuw, pagina=0)
    if "error" in diff:
        print(f"  DIFF FOUT: {diff['error']}")
        continue

    diff["meta"] = diff.get("meta", {})
    diff["meta"]["nieuw_bestand"] = os.path.basename(nieuw)

    pdf_bytes = generate_overlay_pdf(oud, nieuw, diff, pagina=0)

    out_path = os.path.join(os.path.dirname(__file__), "output", f"overlay_{naam}_p1.pdf")
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"  -> {out_path} ({len(pdf_bytes) // 1024} KB)")

"""
Genereert overlay-PDFs voor alle drie tekeningparen.
oud-variant = samenvatting/rapport (A4)
nieuw-variant = gemarkeerde tekening

Run: python gen_overlays.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from app.overlay import generate_split_rapport
from app.config import DiffConfig
import fitz

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")
OUT  = r"C:\tmp"

os.makedirs(OUT, exist_ok=True)

cfg = DiffConfig()  # use_new_wand_diff=True, vision_per_segment_actief=False

PAREN = {
    "56helling": (
        os.path.join(BASE, "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"),
        os.path.join(BASE, "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"),
    ),
    "muiden": (
        os.path.join(BASE, "WT-PLG-D2.1_20250324_B.pdf"),
        os.path.join(BASE, "WT-PLG-D2.1_20260202_E.pdf"),
    ),
    "5102": (
        os.path.join(BASE, "5102_Eerste verdieping_17-01-2025_ (3).pdf"),
        os.path.join(BASE, "5102_Eerste verdieping_05-03-2025_ (2).pdf"),
    ),
}

for naam, (oud_path, nieuw_path) in PAREN.items():
    print(f"\n--- {naam} ---")
    if not os.path.exists(oud_path):
        print(f"  SKIP oud niet gevonden: {oud_path}")
        continue
    if not os.path.exists(nieuw_path):
        print(f"  SKIP nieuw niet gevonden: {nieuw_path}")
        continue

    doc = fitz.open(nieuw_path)
    n_pag = len(doc)
    doc.close()
    print(f"  pagina's: {n_pag}")

    t0 = time.perf_counter()
    try:
        rapport_bytes, tekening_bytes = generate_split_rapport(
            oud_path, nieuw_path,
            aantal_paginas=n_pag,
            oud_naam=os.path.basename(oud_path),
            nieuw_naam=os.path.basename(nieuw_path),
            config=cfg,
        )
    except Exception as e:
        import traceback
        print(f"  FOUT: {e}")
        traceback.print_exc()
        continue
    elapsed = time.perf_counter() - t0
    print(f"  runtime: {elapsed:.1f}s")

    oud_out  = os.path.join(OUT, f"overlay_{naam}oud.pdf")
    nieuw_out = os.path.join(OUT, f"overlay_{naam}nieuw.pdf")

    with open(oud_out, "wb") as f:
        f.write(rapport_bytes)
    with open(nieuw_out, "wb") as f:
        f.write(tekening_bytes)

    print(f"  Rapport  -> {oud_out}  ({len(rapport_bytes)//1024} KB)")
    print(f"  Tekening -> {nieuw_out}  ({len(tekening_bytes)//1024} KB)")

print("\nKlaar.")

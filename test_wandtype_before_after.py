"""
Test _WANDTYPE_TERMEN voor/na toevoeging van 'wand' als losse term.

Gebruik:
    cd kk-tekening-diff
    python test_wandtype_before_after.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import fitz
import app.tekening_profiel as tp
from app.tekening_profiel import detecteer_orientatie, vind_legenda

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")

PROJECTEN = [
    {
        "naam": "5102",
        "pdf": "5102_Eerste verdieping_05-03-2025_ (2).pdf",
        "pagina": 0,
    },
    {
        "naam": "56 de Helling",
        "pdf": "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf",
        "pagina": 0,
    },
]

REGEX_VOOR = re.compile(
    r"\b(kalkzandsteen|gibo|isolatie|metselwerk|beton|prefab|hsb|sandwich|pir|"
    r"rhombus|hardschuim|achterwand|voorzetwand|mato|stuc|biobased|gyproc|cellenbeton|"
    r"ytong|poriso|siporex|damwand|glaswand|systeemwand|scheidingswand|binnenwand|"
    r"buitenwand|draagwand|spouwwand|brandwand|staalstud)\b",
    re.I,
)

REGEX_NA = re.compile(
    r"\b(kalkzandsteen|gibo|isolatie|metselwerk|beton|prefab|hsb|sandwich|pir|"
    r"rhombus|hardschuim|achterwand|voorzetwand|mato|stuc|biobased|gyproc|cellenbeton|"
    r"ytong|poriso|siporex|damwand|glaswand|systeemwand|scheidingswand|binnenwand|"
    r"buitenwand|draagwand|spouwwand|brandwand|staalstud|wand)\b",
    re.I,
)

for proj in PROJECTEN:
    print(f"\n{'=' * 65}")
    print(f"PROJECT: {proj['naam']}")
    print("=" * 65)

    pdf_path = os.path.join(BASE, proj["pdf"])
    if not os.path.exists(pdf_path):
        print(f"  BESTAND NIET GEVONDEN: {pdf_path}")
        continue

    doc = fitz.open(pdf_path)
    page = doc[proj["pagina"]]
    ori = detecteer_orientatie(page)

    # --- VOOR (huidige regex, zonder 'wand') ---
    tp._WANDTYPE_TERMEN = REGEX_VOOR
    resultaat_voor = vind_legenda(page, ori)
    print(f"\n  [VOOR]  {len(resultaat_voor)} item(s)")
    for rgb, naam in resultaat_voor.items():
        print(f"    {rgb}  ->  {naam}")
    if not resultaat_voor:
        print("    (geen resultaten)")

    # --- NA (nieuwe regex, met 'wand') ---
    tp._WANDTYPE_TERMEN = REGEX_NA
    resultaat_na = vind_legenda(page, ori)
    print(f"\n  [NA]    {len(resultaat_na)} item(s)")
    for rgb, naam in resultaat_na.items():
        print(f"    {rgb}  ->  {naam}")
    if not resultaat_na:
        print("    (geen resultaten)")

    # --- Diff ---
    delta = len(resultaat_na) - len(resultaat_voor)
    if delta > 0:
        print(f"\n  + {delta} extra item(s) gevonden dankzij 'wand'")
    elif delta < 0:
        print(f"\n  REGRESSIE: {abs(delta)} item(s) minder dan voor! Onderzoek vereist.")
    else:
        print(f"\n  = Geen verschil in aantal")

    doc.close()

print(f"\n{'=' * 65}")
print("Klaar.")

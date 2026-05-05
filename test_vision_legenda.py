"""
Test Vision-gebaseerde legenda-lezer op alle 4 K&K projecten.

Gebruik:
    cd kk-tekening-diff
    python test_vision_legenda.py

Vereist: ANTHROPIC_API_KEY in omgeving.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import fitz

from app.tekening_profiel import (
    detecteer_orientatie,
    vind_legenda,
    vind_legenda_combined,
    vind_legenda_vision,
)

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")

PROJECTEN = [
    {
        "naam": "56 de Helling",
        "pdf": "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf",
        "pagina": 0,
        "verwacht": "vector en Vision moeten overeenkomen",
    },
    {
        "naam": "Muiden D2.1",
        "pdf": "WT-PLG-D2.1_20260202_E.pdf",
        "pagina": 0,
        "verwacht": "Vision geeft correcte labels (niet alles kalkzandsteen)",
    },
    {
        "naam": "5102",
        "pdf": "5102_Eerste verdieping_05-03-2025_ (2).pdf",
        "pagina": 0,
        "verwacht": "Vision vindt 'lichte binnenwand 70mm' die vector mist",
    },
    {
        "naam": "BD/BU",
        "pdf": "BU-N-101 - PLATTEGROND 1e VERDIEPING----V2 (1).pdf",
        "pagina": 0,
        "verwacht": "Vision leest legenda correct ondanks slecht revisie-paar",
    },
]

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("WAARSCHUWING: ANTHROPIC_API_KEY niet ingesteld — Vision wordt overgeslagen\n")

for proj in PROJECTEN:
    print(f"\n{'=' * 65}")
    print(f"PROJECT: {proj['naam']}")
    print(f"Verwacht: {proj['verwacht']}")
    print("=" * 65)

    pdf_path = os.path.join(BASE, proj["pdf"])
    if not os.path.exists(pdf_path):
        print(f"  BESTAND NIET GEVONDEN: {pdf_path}")
        continue

    doc = fitz.open(pdf_path)
    page = doc[proj["pagina"]]
    ori = detecteer_orientatie(page)

    print(f"  Bestand : {proj['pdf']}")
    print(f"  Rotatie : {ori['rotation']}°  |  "
          f"Display: {ori['display_width']:.0f} × {ori['display_height']:.0f} pt")

    # --- VECTOR ---
    print("\n  [VECTOR]")
    vector = vind_legenda(page, ori)
    if vector:
        for rgb, naam in vector.items():
            print(f"    {rgb}  ->  {naam}")
    else:
        print("    (geen resultaten)")

    # --- VISION ---
    print("\n  [VISION]")
    vision_raw: dict = {}
    if api_key:
        vision_raw = vind_legenda_vision(page, ori, api_key)
        if vision_raw:
            for rgb, info in vision_raw.items():
                rel = "KK" if info["relevant_voor_kk"] else "  "
                print(f"    [{rel}] {rgb}  ->  {info['naam']}  [{info['arcering']}]")
        else:
            print("    (geen resultaten)")
    else:
        print("    (overgeslagen — geen API key)")

    # --- GECOMBINEERD ---
    print("\n  [GECOMBINEERD]")
    combined = vind_legenda_combined(page, ori, api_key)
    for rgb, naam in combined.items():
        print(f"    {rgb}  ->  {naam}")

    # --- VERGELIJKING ---
    if api_key and vision_raw:
        vector_namen = set(vector.values())
        vision_namen = {info["naam"] for info in vision_raw.values()}
        alleen_vision = vision_namen - vector_namen
        alleen_vector = vector_namen - vision_namen
        print()
        if alleen_vision:
            print(f"  + Vision extra (niet in vector): {alleen_vision}")
        if alleen_vector:
            print(f"  - Vector extra (niet in Vision): {alleen_vector}")
        if not alleen_vision and not alleen_vector and vector and vision_raw:
            print("  ✓ Vector en Vision zijn consistent")

    doc.close()

print(f"\n{'=' * 65}")
print("Klaar.")

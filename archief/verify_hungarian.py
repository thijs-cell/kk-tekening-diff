"""
Stap 4: Verify Hongaarse pipeline draait correct op 56 de Helling p.1.
Gebruik: python verify_hungarian.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import fitz
from app.config import DiffConfig
from app.wand_diff import bereken_wand_diff
from app.tekening_profiel import detecteer_orientatie, vind_legenda_combined

OLD = "../Karregat & Koning MVP/56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"
NEW = "../Karregat & Koning MVP/56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"

cfg = DiffConfig()
assert not cfg.use_vision_pipeline, "Flag moet False zijn voor deze test"

oud_doc   = fitz.open(OLD)
nieuw_doc = fitz.open(NEW)
oud_page  = oud_doc[0]
nieuw_page = nieuw_doc[0]

oud_ori  = detecteer_orientatie(oud_page)
nieuw_ori = detecteer_orientatie(nieuw_page)
legenda   = vind_legenda_combined(nieuw_page, nieuw_ori, api_key=None)

t0 = time.time()
resultaten = bereken_wand_diff(oud_page, nieuw_page, oud_ori, nieuw_ori, legenda, cfg=cfg)
elapsed = time.time() - t0

oud_doc.close()
nieuw_doc.close()

nieuw   = [r for r in resultaten if r["type"] == "nieuw"]
verdwn  = [r for r in resultaten if r["type"] == "verdwenen"]
gewij   = [r for r in resultaten if r["type"] == "gewijzigd"]

print("=== Stap 4: Hungarian pipeline verify ===")
print(f"Totaal wijzigingen : {len(resultaten)}")
print(f"  Nieuw            : {len(nieuw)}")
print(f"  Verdwenen        : {len(verdwn)}")
print(f"  Gewijzigd        : {len(gewij)}")
print(f"Runtime            : {elapsed:.2f}s")
print(f"Flag use_vision    : {cfg.use_vision_pipeline}  (moet False zijn)")
print()
if resultaten:
    print("Eerste 5 resultaten:")
    for r in resultaten[:5]:
        print(f"  {r['type']:10s}  wandtype={r.get('wandtype','')!r:20s}  pos={[round(x,1) for x in r['positie']]}")

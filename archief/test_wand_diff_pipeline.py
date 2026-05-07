"""
Integratietest: wand_diff pipeline op echte K&K tekeningparen.
Run: python test_wand_diff_pipeline.py
"""

import sys
import time
import os

# Pad naar app-module
sys.path.insert(0, os.path.dirname(__file__))

import fitz
from app.wand_diff import bereken_wand_diff
from app.config import DiffConfig
from app.tekening_profiel import detecteer_orientatie, vind_legenda_combined
from app.overlay import _collect_wanden_profiel

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")

PAREN = {
    "56_de_Helling": (
        os.path.join(BASE, "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"),
        os.path.join(BASE, "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"),
    ),
    "5102": (
        os.path.join(BASE, "5102_Eerste verdieping_17-01-2025_ (3).pdf"),
        os.path.join(BASE, "5102_Eerste verdieping_05-03-2025_ (2).pdf"),
    ),
    "Muiden_D2.1": (
        os.path.join(BASE, "WT-PLG-D2.1_20250324_B.pdf"),
        os.path.join(BASE, "WT-PLG-D2.1_20260202_E.pdf"),
    ),
}

PAGINA = 0
SEP = "-" * 60


def run_test(naam: str, oud_path: str, nieuw_path: str, cfg: DiffConfig):
    print(f"\n{SEP}")
    print(f"TEST: {naam}  (pagina {PAGINA + 1})")
    print(SEP)

    if not os.path.exists(oud_path):
        print(f"  SKIP — oud bestand niet gevonden: {oud_path}")
        return
    if not os.path.exists(nieuw_path):
        print(f"  SKIP — nieuw bestand niet gevonden: {nieuw_path}")
        return

    t0 = time.perf_counter()
    try:
        oud_doc = fitz.open(oud_path)
        nieuw_doc = fitz.open(nieuw_path)
        oud_page = oud_doc[PAGINA]
        nieuw_page = nieuw_doc[PAGINA]
        oud_ori = detecteer_orientatie(oud_page)
        nieuw_ori = detecteer_orientatie(nieuw_page)
        legenda = vind_legenda_combined(nieuw_page, nieuw_ori, api_key=None)
        resultaten = bereken_wand_diff(oud_page, nieuw_page, oud_ori, nieuw_ori, legenda, cfg=cfg)
    except Exception as e:
        print(f"  FOUT: {e}")
        import traceback; traceback.print_exc()
        return
    finally:
        try: oud_doc.close()
        except: pass
        try: nieuw_doc.close()
        except: pass

    elapsed = time.perf_counter() - t0

    nieuw = [r for r in resultaten if r["type"] == "nieuw"]
    verdwenen = [r for r in resultaten if r["type"] == "verdwenen"]
    onbekend = [r for r in resultaten if r["wandtype"] == "type onbekend"]
    types_gevonden = sorted(set(r["wandtype"] for r in resultaten))

    print(f"  Legenda-types gevonden : {len(legenda)} — {list(legenda.values())}")
    print(f"  Wandsegmenten totaal   : {len(resultaten)}")
    print(f"    Nieuw                : {len(nieuw)}")
    print(f"    Verdwenen            : {len(verdwenen)}")
    print(f"    'type onbekend'      : {len(onbekend)} van {len(resultaten)}")
    print(f"  Geïdentificeerde types : {types_gevonden}")
    print(f"  Runtime                : {elapsed:.2f}s")

    if nieuw:
        print(f"\n  [nieuw details]")
        for r in nieuw[:8]:
            print(f"    wandtype={r['wandtype']!r:35s}  pos={[round(v) for v in r['positie']]}")
        if len(nieuw) > 8:
            print(f"    ... en {len(nieuw) - 8} meer")

    if verdwenen:
        print(f"\n  [verdwenen details]")
        for r in verdwenen[:8]:
            print(f"    wandtype={r['wandtype']!r:35s}  pos={[round(v) for v in r['positie']]}")
        if len(verdwenen) > 8:
            print(f"    ... en {len(verdwenen) - 8} meer")


def run_regressie(naam: str, oud_path: str, nieuw_path: str):
    print(f"\n{SEP}")
    print(f"REGRESSIE (use_new_wand_diff=False): {naam}  (pagina {PAGINA + 1})")
    print(SEP)

    if not os.path.exists(oud_path):
        print(f"  SKIP — oud bestand niet gevonden")
        return
    if not os.path.exists(nieuw_path):
        print(f"  SKIP — nieuw bestand niet gevonden")
        return

    cfg_oud = DiffConfig(use_new_wand_diff=False)

    t0 = time.perf_counter()
    try:
        # Gebruik _collect_wanden_profiel met het oude pad
        import fitz as _fitz
        _doc = _fitz.open(nieuw_path)
        _page = _doc[PAGINA]
        pw, ph = _page.rect.width, _page.rect.height
        _doc.close()

        nieuw_items, verdwenen_items, materiaal_items, rij_items = _collect_wanden_profiel(
            oud_path, nieuw_path, PAGINA, pw, ph, layout=None, cfg=cfg_oud
        )
    except Exception as e:
        print(f"  FOUT: {e}")
        import traceback; traceback.print_exc()
        return

    elapsed = time.perf_counter() - t0

    print(f"  Nieuwe wand-items      : {len(nieuw_items)}")
    print(f"  Verdwenen wand-items   : {len(verdwenen_items)}")
    print(f"  Materiaalwissel-items  : {len(materiaal_items)}")
    print(f"  Rij-items              : {len(rij_items)}")
    print(f"  Runtime                : {elapsed:.2f}s")

    if nieuw_items:
        print(f"\n  [nieuw labels (oud pad)]")
        for it in nieuw_items[:6]:
            print(f"    beschrijving={it['beschrijving']!r}  rect={it['rect']}")
    if verdwenen_items:
        print(f"\n  [verdwenen labels (oud pad)]")
        for it in verdwenen_items[:6]:
            print(f"    beschrijving={it['beschrijving']!r}  rect={it['rect']}")

    print(f"\n  Cirkels zijn vaste r=8 (niet ovals) — backward compat OK als rect w/h ~16pt:")
    for it in (nieuw_items + verdwenen_items)[:3]:
        r = it["rect"]
        print(f"    w={r.x1-r.x0:.1f}pt  h={r.y1-r.y0:.1f}pt  (verwacht ~16×16)")


# ---------------------------------------------------------------------------
# Hoofdprogramma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = DiffConfig()  # use_new_wand_diff=True, vision_per_segment_actief=False

    oud_h, nieuw_h = PAREN["56_de_Helling"]
    oud_5, nieuw_5 = PAREN["5102"]
    oud_m, nieuw_m = PAREN["Muiden_D2.1"]

    print("=" * 60)
    print("K&K WAND_DIFF INTEGRATIETESTS")
    print("=" * 60)

    # Test 1
    run_test("Test 1 — 56 de Helling", oud_h, nieuw_h, cfg)

    # Test 2
    run_test("Test 2 — 5102", oud_5, nieuw_5, cfg)

    # Test 3
    run_test("Test 3 — Muiden D2.1", oud_m, nieuw_m, cfg)

    # Test 4 — Regressie
    run_regressie("Test 4 — 56 de Helling (oud pad)", oud_h, nieuw_h)

    print(f"\n{'=' * 60}")
    print("ALLE TESTS KLAAR")

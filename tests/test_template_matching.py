"""
Template-matching test op data/helling cluster_0_deel4.

Pipeline:
1. Render deel-4 zone (3335,225,3779,1156) op 200 DPI uit beide PDFs.
2. Laad alle templates uit references/helling/templates/<wandtype>.png als grayscale.
3. Multi-scale + multi-rotatie cross-correlatie via cv2.matchTemplate (TM_CCOEFF_NORMED).
4. Vergelijk oud-matches met nieuw-matches per wandtype:
   - alleen-in-oud   -> verdwenen
   - alleen-in-nieuw -> toegevoegd
   - in beide met andere schaal -> mogelijk dikte-wijziging
5. Visualisaties per wandtype in <tmp>/helling_template_test/.

GT-validatie:
- GT-1: Gibo zwaar dikte 70 -> 100mm  (schaalverhouding ~1.43)
- GT-2: Hardschuimisolatie toegevoegd

Standalone, geen Vision, geen wijzigingen aan app/.
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROJECT = "helling"
PAGINA = 0
DEEL4_BBOX_PT = (3335.0, 225.0, 3779.0, 1156.0)
RENDER_DPI = 200
PT_PX = RENDER_DPI / 72.0  # ~2.778

TEMPLATES_DIR = ROOT / "references" / PROJECT / "templates"
PDF_OUD = ROOT / "data" / PROJECT / "oud.pdf"
PDF_NIEUW = ROOT / "data" / PROJECT / "nieuw.pdf"
OUT_DIR = Path(tempfile.gettempdir()) / "helling_template_test"

SCALES = [0.75, 1.0, 1.25, 1.5]
ROTATIONS = [0, 90]
THRESHOLD = 0.85
NMS_DIST_PX = 40
LOC_MATCH_PT = 30
LOC_MATCH_PX = LOC_MATCH_PT * PT_PX
DIKTE_RATIO_LO, DIKTE_RATIO_HI = 1.30, 1.60  # 70->100mm = 1.43

GIBO_KEYS = ("gibo_zwaar",)
HARD_KEYS = ("hardschuim",)


def _safe(naam: str) -> str:
    return re.sub(r"[^\w-]", "_", naam.lower())


def render_zone(pdf_path: Path, bbox_pt: tuple, dpi: int, out_path: Path) -> tuple[int, int]:
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[PAGINA]
        clip = fitz.Rect(*bbox_pt) & page.rect
        if clip.is_empty:
            raise RuntimeError(f"clip leeg voor {pdf_path}")
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        return pix.width, pix.height
    finally:
        doc.close()


def load_templates(templates_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(templates_dir.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out[p.stem] = img
    return out


def match_template_multi(zone_gray, tpl_gray, scales, rotations, threshold):
    """Geeft lijst (score, x, y, w, h, scale, rot) — top-left in zone-pixels."""
    H, W = zone_gray.shape
    results = []
    for rot in rotations:
        k = rot // 90
        base = np.rot90(tpl_gray, k=k) if k else tpl_gray
        for s in scales:
            new_w = max(1, int(round(base.shape[1] * s)))
            new_h = max(1, int(round(base.shape[0] * s)))
            if new_w < 4 or new_h < 4 or new_w >= W or new_h >= H:
                continue
            tpl_s = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)
            try:
                res = cv2.matchTemplate(zone_gray, tpl_s, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            ys, xs = np.where(res >= threshold)
            for x, y in zip(xs.tolist(), ys.tolist()):
                results.append((float(res[y, x]), x, y, new_w, new_h, s, rot))
    return results


def nms(matches, min_dist_px):
    if not matches:
        return []
    sorted_m = sorted(matches, key=lambda m: -m[0])
    kept = []
    for m in sorted_m:
        score, x, y, w, h, s, rot = m
        cx, cy = x + w / 2, y + h / 2
        ok = True
        for k in kept:
            kx, ky = k[1] + k[3] / 2, k[2] + k[4] / 2
            if (cx - kx) ** 2 + (cy - ky) ** 2 < min_dist_px ** 2:
                ok = False
                break
        if ok:
            kept.append(m)
    return kept


def vergelijk_matches(oud_matches, nieuw_matches, loc_dist_px):
    """Geeft (toegevoegd, verdwenen, mogelijk_dikte_shift)."""
    paired_n = set()
    paired_pairs = []
    verdwenen = []

    for o in oud_matches:
        ox, oy = o[1] + o[3] / 2, o[2] + o[4] / 2
        beste_idx = None
        beste_d = float("inf")
        for ni, n in enumerate(nieuw_matches):
            if ni in paired_n:
                continue
            nx, ny = n[1] + n[3] / 2, n[2] + n[4] / 2
            d = math.hypot(ox - nx, oy - ny)
            if d < beste_d and d <= loc_dist_px:
                beste_d = d
                beste_idx = ni
        if beste_idx is not None:
            paired_n.add(beste_idx)
            paired_pairs.append((o, nieuw_matches[beste_idx], beste_d))
        else:
            verdwenen.append(o)

    toegevoegd = [n for ni, n in enumerate(nieuw_matches) if ni not in paired_n]

    dikte_shift = []
    for o, n, d in paired_pairs:
        ratio = n[5] / o[5] if o[5] > 0 else 0
        if DIKTE_RATIO_LO <= ratio <= DIKTE_RATIO_HI:
            dikte_shift.append({"oud": o, "nieuw": n, "afstand_px": d, "ratio": round(ratio, 2)})

    return toegevoegd, verdwenen, dikte_shift, paired_pairs


def draw_matches(img_color, matches, color, label_prefix=""):
    canvas = img_color.copy()
    for m in matches:
        score, x, y, w, h, s, rot = m
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        text = f"{label_prefix}s{s:.2f} {score:.2f}"
        cv2.putText(canvas, text, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def main():
    if not (PDF_OUD.exists() and PDF_NIEUW.exists() and TEMPLATES_DIR.exists()):
        print("FAIL: PDFs of templates ontbreken")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Stap 1
    print(f"[1] Render zone bbox={DEEL4_BBOX_PT} @ {RENDER_DPI} DPI")
    oud_zone_path = OUT_DIR / "oud_zone.png"
    nieuw_zone_path = OUT_DIR / "nieuw_zone.png"
    ow, oh = render_zone(PDF_OUD, DEEL4_BBOX_PT, RENDER_DPI, oud_zone_path)
    nw, nh = render_zone(PDF_NIEUW, DEEL4_BBOX_PT, RENDER_DPI, nieuw_zone_path)
    print(f"    oud_zone.png   {ow}x{oh}px")
    print(f"    nieuw_zone.png {nw}x{nh}px")
    print(f"    -> {OUT_DIR}")

    oud_gray = cv2.imread(str(oud_zone_path), cv2.IMREAD_GRAYSCALE)
    nieuw_gray = cv2.imread(str(nieuw_zone_path), cv2.IMREAD_GRAYSCALE)
    oud_color = cv2.imread(str(oud_zone_path), cv2.IMREAD_COLOR)
    nieuw_color = cv2.imread(str(nieuw_zone_path), cv2.IMREAD_COLOR)

    # Stap 2
    print(f"\n[2] Templates uit {TEMPLATES_DIR.relative_to(ROOT)}")
    templates = load_templates(TEMPLATES_DIR)
    print(f"    {len(templates)} templates geladen")
    for nm, t in templates.items():
        print(f"    {nm:<45}  {t.shape[1]}x{t.shape[0]}px")

    # Stap 3
    print(f"\n[3] Multi-scale matching: scales={SCALES} rotations={ROTATIONS} threshold={THRESHOLD}")
    rapporten = {}
    t_match = time.time()
    for naam, tpl in templates.items():
        oud_raw = match_template_multi(oud_gray, tpl, SCALES, ROTATIONS, THRESHOLD)
        nieuw_raw = match_template_multi(nieuw_gray, tpl, SCALES, ROTATIONS, THRESHOLD)
        oud_matches = nms(oud_raw, NMS_DIST_PX)
        nieuw_matches = nms(nieuw_raw, NMS_DIST_PX)
        rapporten[naam] = {
            "oud": oud_matches,
            "nieuw": nieuw_matches,
            "oud_raw": len(oud_raw),
            "nieuw_raw": len(nieuw_raw),
        }
    print(f"    matching duur: {time.time() - t_match:.1f}s")

    # Stap 4
    print(f"\n[4] Vergelijk per wandtype (loc-drempel {LOC_MATCH_PT}pt = {LOC_MATCH_PX:.0f}px, "
          f"dikte-ratio {DIKTE_RATIO_LO}..{DIKTE_RATIO_HI})")
    print(f"    {'wandtype':<45} {'oud_raw':>7} {'oud':>5} {'nw_raw':>7} {'nw':>4} {'toeg':>5} {'verd':>5} {'shift':>5}")
    samenvatting = {}
    for naam, r in rapporten.items():
        toeg, verd, shift, paired = vergelijk_matches(r["oud"], r["nieuw"], LOC_MATCH_PX)
        samenvatting[naam] = {
            "n_oud": len(r["oud"]), "n_nieuw": len(r["nieuw"]),
            "n_oud_raw": r["oud_raw"], "n_nieuw_raw": r["nieuw_raw"],
            "toegevoegd": toeg, "verdwenen": verd,
            "dikte_shift": shift, "paired": paired,
        }
        print(f"    {naam:<45} {r['oud_raw']:>7} {len(r['oud']):>5} "
              f"{r['nieuw_raw']:>7} {len(r['nieuw']):>4} "
              f"{len(toeg):>5} {len(verd):>5} {len(shift):>5}")

    # Stap 5: visualisaties (alleen waar minstens iets gevonden is)
    print(f"\n[5] Visualisaties opslaan -> {OUT_DIR}")
    saved = 0
    for naam, r in rapporten.items():
        if not r["oud"] and not r["nieuw"]:
            continue
        oud_img = draw_matches(oud_color, r["oud"], (0, 0, 255))
        nieuw_img = draw_matches(nieuw_color, r["nieuw"], (0, 200, 0))
        cv2.imwrite(str(OUT_DIR / f"matches_{_safe(naam)}_oud.png"), oud_img)
        cv2.imwrite(str(OUT_DIR / f"matches_{_safe(naam)}_nieuw.png"), nieuw_img)
        saved += 2
    print(f"    {saved} PNGs opgeslagen")

    # Stap 6: GT-validatie
    print(f"\n=== GT-VALIDATIE ===")
    gibo_naam = next((n for n in samenvatting if any(k in n.lower() for k in GIBO_KEYS)), None)
    hard_naam = next((n for n in samenvatting if any(k in n.lower() for k in HARD_KEYS)), None)

    print(f"  GT-1: Gibo zwaar dikte-shift 70->100mm (~ratio 1.43)")
    if gibo_naam and samenvatting[gibo_naam]["dikte_shift"]:
        for s in samenvatting[gibo_naam]["dikte_shift"][:5]:
            o = s["oud"]; n = s["nieuw"]
            print(f"        GEVONDEN — oud_scale={o[5]:.2f} (score={o[0]:.2f}, "
                  f"@{o[1]},{o[2]}) | nieuw_scale={n[5]:.2f} (score={n[0]:.2f}, @{n[1]},{n[2]}) "
                  f"| ratio={s['ratio']}")
        gibo_ok = True
    elif gibo_naam:
        toeg = len(samenvatting[gibo_naam]["toegevoegd"])
        verd = len(samenvatting[gibo_naam]["verdwenen"])
        n_o = samenvatting[gibo_naam]["n_oud"]
        n_n = samenvatting[gibo_naam]["n_nieuw"]
        print(f"        NIET GEVONDEN — geen scale-shift in ratio {DIKTE_RATIO_LO}..{DIKTE_RATIO_HI}")
        print(f"          (template '{gibo_naam}': oud={n_o} matches, nieuw={n_n}, +{toeg}/-{verd})")
        gibo_ok = False
    else:
        print(f"        NIET GEVONDEN — geen Gibo-zwaar template")
        gibo_ok = False

    print(f"  GT-2: Hardschuimisolatie toegevoegd")
    if hard_naam and samenvatting[hard_naam]["toegevoegd"]:
        n_t = len(samenvatting[hard_naam]["toegevoegd"])
        n_v = len(samenvatting[hard_naam]["verdwenen"])
        print(f"        GEVONDEN — {n_t} toegevoegde locaties, {n_v} verdwenen "
              f"(template '{hard_naam}')")
        for m in samenvatting[hard_naam]["toegevoegd"][:5]:
            print(f"          score={m[0]:.2f} scale={m[5]:.2f} rot={m[6]} bbox=({m[1]},{m[2]},+{m[3]},+{m[4]})")
        hard_ok = True
    elif hard_naam:
        n_o = samenvatting[hard_naam]["n_oud"]
        n_n = samenvatting[hard_naam]["n_nieuw"]
        print(f"        NIET GEVONDEN — geen 'toegevoegd' classificatie")
        print(f"          (template '{hard_naam}': oud={n_o} matches, nieuw={n_n})")
        hard_ok = False
    else:
        print(f"        NIET GEVONDEN — geen hardschuim template")
        hard_ok = False

    # Top 10 hoogst-scorende matches per zone
    print(f"\n=== TOP 10 HOOGST-SCOREND ===")
    for label, key in (("oud", "oud"), ("nieuw", "nieuw")):
        all_matches = []
        for naam, r in rapporten.items():
            for m in r[key]:
                all_matches.append((m, naam))
        all_matches.sort(key=lambda x: -x[0][0])
        print(f"  {label}:")
        for m, naam in all_matches[:10]:
            score, x, y, w, h, s, rot = m
            print(f"    {score:.3f}  scale={s:.2f} rot={rot:>3}  "
                  f"bbox=({x},{y},+{w},+{h})  -> {naam}")

    # Stats + eindoordeel
    totaal_oud_raw = sum(s["n_oud_raw"] for s in samenvatting.values())
    totaal_nieuw_raw = sum(s["n_nieuw_raw"] for s in samenvatting.values())
    totaal_oud = sum(s["n_oud"] for s in samenvatting.values())
    totaal_nieuw = sum(s["n_nieuw"] for s in samenvatting.values())
    totaal_toeg = sum(len(s["toegevoegd"]) for s in samenvatting.values())
    totaal_verd = sum(len(s["verdwenen"]) for s in samenvatting.values())
    totaal_shift = sum(len(s["dikte_shift"]) for s in samenvatting.values())

    print(f"\n=== STATS ===")
    print(f"  Templates verwerkt:   {len(templates)}")
    print(f"  Schalen / rotaties:   {SCALES} / {ROTATIONS}")
    print(f"  Drempel / NMS:        {THRESHOLD} / {NMS_DIST_PX}px")
    print(f"  Matches raw:          oud={totaal_oud_raw}  nieuw={totaal_nieuw_raw}")
    print(f"  Matches na NMS:       oud={totaal_oud}  nieuw={totaal_nieuw}")
    if totaal_oud_raw > 0:
        print(f"  Reductie raw->NMS:    {totaal_oud_raw / max(totaal_oud, 1):.1f}x (oud)  "
              f"{totaal_nieuw_raw / max(totaal_nieuw, 1):.1f}x (nieuw)")
    print(f"  Toegevoegd / verdwenen / dikte-shift:  {totaal_toeg} / {totaal_verd} / {totaal_shift}")
    print(f"  Visualisaties:        {saved // 2} wandtype-paren")
    print(f"  Recall (GT):          {(1 if gibo_ok else 0) + (1 if hard_ok else 0)}/2")
    print(f"  Tijd totaal:          {time.time() - t0:.1f}s")
    print(f"  Vision-kosten:        $0.00")
    return 0


if __name__ == "__main__":
    sys.exit(main())

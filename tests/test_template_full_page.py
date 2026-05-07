"""
Pagina-brede template-matching test op data/helling pagina 1.

Doel: meten of ronde-0 parameters schalen naar de hele pagina (zonder zone-restrictie).
Geen GT-validatie, alleen schaalbaarheid: tijd, geheugen, match-aantallen,
ruis-zones via spatiale binning.

Parameters: scales [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
rotations [0, 90, 180, 270], threshold 0.70, NMS 25px.
"""
from __future__ import annotations

import gc
import math
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil
import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_template_matching import (
    match_template_multi,
    nms,
    load_templates,
    _safe,
)

PROJECT = "helling"
PAGINA = 0
RENDER_DPI = 200
PT_PX = RENDER_DPI / 72.0

PDF_OUD = ROOT / "data" / PROJECT / "oud.pdf"
PDF_NIEUW = ROOT / "data" / PROJECT / "nieuw.pdf"
TEMPLATES_DIR = ROOT / "references" / PROJECT / "templates"
OUT_DIR = Path(tempfile.gettempdir()) / "helling_template_full_page"

# Ronde-0 parameters
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
ROTATIONS = [0, 90, 180, 270]
THRESHOLD = 0.70
NMS_DIST_PX = 25

BIN_SIZE_PX = 200  # 200x200px tiles voor ruis-zone analyse


def proc_mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def render_full_page(pdf_path: Path, dpi: int, out_path: Path) -> tuple[int, int, float]:
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[PAGINA]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        return pix.width, pix.height, time.time() - t0
    finally:
        doc.close()


def bin_matches(matches_with_naam, bin_size_px, img_w, img_h):
    """Tel matches per BIN_SIZExBIN_SIZE tile. Geeft sortable lijst."""
    bins = defaultdict(lambda: {"count": 0, "templates": defaultdict(int)})
    for m, naam in matches_with_naam:
        score, x, y, w, h, s, rot = m
        cx = x + w / 2
        cy = y + h / 2
        bx = int(cx // bin_size_px)
        by = int(cy // bin_size_px)
        bins[(bx, by)]["count"] += 1
        bins[(bx, by)]["templates"][naam] += 1
    return bins


def main():
    if not (PDF_OUD.exists() and PDF_NIEUW.exists() and TEMPLATES_DIR.exists()):
        print("FAIL: input ontbreekt")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    mem_start = proc_mem_mb()
    print(f"[start] mem={mem_start:.0f}MB")

    # 1. Render full page
    oud_path = OUT_DIR / "oud_full.png"
    nieuw_path = OUT_DIR / "nieuw_full.png"
    print(f"[1] Render volledige pagina @ {RENDER_DPI} DPI")
    ow, oh, t_oud_render = render_full_page(PDF_OUD, RENDER_DPI, oud_path)
    print(f"    oud:   {ow}x{oh}px in {t_oud_render:.1f}s, mem={proc_mem_mb():.0f}MB")
    nw, nh, t_nieuw_render = render_full_page(PDF_NIEUW, RENDER_DPI, nieuw_path)
    print(f"    nieuw: {nw}x{nh}px in {t_nieuw_render:.1f}s, mem={proc_mem_mb():.0f}MB")
    print(f"    totaal pixels per zone: {ow * oh / 1e6:.1f}M")

    # Load images as grayscale
    print(f"\n[2] Inladen grayscale images")
    t_load = time.time()
    oud_gray = cv2.imread(str(oud_path), cv2.IMREAD_GRAYSCALE)
    nieuw_gray = cv2.imread(str(nieuw_path), cv2.IMREAD_GRAYSCALE)
    print(f"    geladen in {time.time() - t_load:.1f}s, mem={proc_mem_mb():.0f}MB")
    print(f"    image bytes per zone: {oud_gray.nbytes / 1e6:.0f}MB grayscale")

    # Estimate result-map memory per call
    H, W = oud_gray.shape
    max_result_mb = (W * H * 4) / 1e6
    print(f"    max result-map per matchTemplate call: ~{max_result_mb:.0f}MB float32")

    # 3. Templates
    templates = load_templates(TEMPLATES_DIR)
    print(f"\n[3] Templates: {len(templates)} geladen, mem={proc_mem_mb():.0f}MB")

    # 4. Matching loop met progress + memory peak tracking
    print(f"\n[4] Matching: scales={SCALES} rotations={ROTATIONS} threshold={THRESHOLD}")
    print(f"    Total calls per zone: {len(templates)} * {len(SCALES)} * {len(ROTATIONS)} = "
          f"{len(templates) * len(SCALES) * len(ROTATIONS)}")
    mem_peak = proc_mem_mb()

    def track_peak():
        nonlocal mem_peak
        m = proc_mem_mb()
        if m > mem_peak:
            mem_peak = m
        return m

    rapporten = {}
    t_matching = time.time()

    for i, (naam, tpl) in enumerate(templates.items(), 1):
        t_tpl = time.time()
        oud_raw = match_template_multi(oud_gray, tpl, SCALES, ROTATIONS, THRESHOLD)
        track_peak()
        nieuw_raw = match_template_multi(nieuw_gray, tpl, SCALES, ROTATIONS, THRESHOLD)
        track_peak()
        oud_nms = nms(oud_raw, NMS_DIST_PX)
        nieuw_nms = nms(nieuw_raw, NMS_DIST_PX)

        rapporten[naam] = {
            "oud_raw": len(oud_raw),
            "nieuw_raw": len(nieuw_raw),
            "oud_nms": oud_nms,
            "nieuw_nms": nieuw_nms,
        }
        dt = time.time() - t_tpl
        m = track_peak()
        print(f"    [{i:>2}/{len(templates)}] {naam:<45} "
              f"raw oud={len(oud_raw):>6} nieuw={len(nieuw_raw):>6}  "
              f"NMS oud={len(oud_nms):>5} nieuw={len(nieuw_nms):>5}  "
              f"{dt:>5.1f}s  mem={m:.0f}MB")
        gc.collect()

    t_matching_total = time.time() - t_matching
    print(f"    matching klaar: {t_matching_total:.1f}s, peak mem={mem_peak:.0f}MB")

    # 5. Aggregeer + ruis-zones
    print(f"\n[5] Spatiale ruis-analyse: bin size = {BIN_SIZE_PX}px "
          f"({BIN_SIZE_PX / PT_PX:.0f}pt = {BIN_SIZE_PX * 25.4 / RENDER_DPI:.0f}mm)")
    oud_alle_matches = [(m, naam) for naam, r in rapporten.items() for m in r["oud_nms"]]
    nieuw_alle_matches = [(m, naam) for naam, r in rapporten.items() for m in r["nieuw_nms"]]

    oud_bins = bin_matches(oud_alle_matches, BIN_SIZE_PX, ow, oh)
    nieuw_bins = bin_matches(nieuw_alle_matches, BIN_SIZE_PX, nw, nh)

    def print_top_bins(label, bins):
        sorted_bins = sorted(bins.items(), key=lambda kv: -kv[1]["count"])[:20]
        print(f"\n  Top 20 zones ({label}):")
        print(f"    {'#':>3}  {'tile_x':>7} {'tile_y':>7}  {'pt_x':>5} {'pt_y':>5}  "
              f"{'mm_x':>5} {'mm_y':>5}  {'matches':>7}  top-3 templates")
        for rank, ((bx, by), info) in enumerate(sorted_bins, 1):
            px_x = bx * BIN_SIZE_PX
            px_y = by * BIN_SIZE_PX
            pt_x = px_x / PT_PX
            pt_y = px_y / PT_PX
            mm_x = px_x * 25.4 / RENDER_DPI
            mm_y = px_y * 25.4 / RENDER_DPI
            top3 = sorted(info["templates"].items(), key=lambda kv: -kv[1])[:3]
            top3_str = ", ".join(f"{nm}({cnt})" for nm, cnt in top3)
            print(f"    {rank:>3}  {bx:>7} {by:>7}  {pt_x:>5.0f} {pt_y:>5.0f}  "
                  f"{mm_x:>5.0f} {mm_y:>5.0f}  {info['count']:>7}  {top3_str}")

    print_top_bins("oud", oud_bins)
    print_top_bins("nieuw", nieuw_bins)

    # 6. Per-template tabel
    print(f"\n[6] Per-template totalen")
    print(f"    {'wandtype':<45} {'oud_raw':>8} {'oud_nms':>8} {'nw_raw':>8} {'nw_nms':>8}")
    sorted_temps = sorted(rapporten.items(), key=lambda kv: -(kv[1]['oud_raw'] + kv[1]['nieuw_raw']))
    for naam, r in sorted_temps:
        print(f"    {naam:<45} {r['oud_raw']:>8} {len(r['oud_nms']):>8} "
              f"{r['nieuw_raw']:>8} {len(r['nieuw_nms']):>8}")

    # 7. Eindstats
    totaal_oud_raw = sum(r["oud_raw"] for r in rapporten.values())
    totaal_nieuw_raw = sum(r["nieuw_raw"] for r in rapporten.values())
    totaal_oud_nms = sum(len(r["oud_nms"]) for r in rapporten.values())
    totaal_nieuw_nms = sum(len(r["nieuw_nms"]) for r in rapporten.values())

    print(f"\n=== EINDSTATS ===")
    print(f"  Pagina:               {ow}x{oh}px = {ow * oh / 1e6:.1f}M pixels")
    print(f"  Templates:            {len(templates)}")
    print(f"  Calls per zone:       {len(templates) * len(SCALES) * len(ROTATIONS)}")
    print(f"  Tijd render:          oud={t_oud_render:.1f}s  nieuw={t_nieuw_render:.1f}s  "
          f"totaal={t_oud_render + t_nieuw_render:.1f}s")
    print(f"  Tijd matching:        {t_matching_total:.1f}s")
    print(f"  Tijd totaal:          {time.time() - t_start:.1f}s")
    print(f"  Geheugen start:       {mem_start:.0f}MB")
    print(f"  Geheugen peak:        {mem_peak:.0f}MB  (delta {mem_peak - mem_start:+.0f}MB)")
    print(f"  Matches raw:          oud={totaal_oud_raw}  nieuw={totaal_nieuw_raw}")
    print(f"  Matches NMS:          oud={totaal_oud_nms}  nieuw={totaal_nieuw_nms}")
    if totaal_oud_raw > 0:
        print(f"  Reductie raw->NMS:    oud={totaal_oud_raw / max(totaal_oud_nms,1):.1f}x  "
              f"nieuw={totaal_nieuw_raw / max(totaal_nieuw_nms,1):.1f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Diagnose: zijn de twee ground-truth wijzigingen detecteerbaar uit de huidige
pixel-diff output van data/helling?

GT-1: Gibo zwaar muur, dikte 70mm -> 100mm
GT-2: Hardschuimisolatie toegevoegd
Locatie: cluster_0_deel4 = rechter kwart van cluster 0 (grootste BFS-cluster)

Geen Vision, alleen analyse.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from app.diff_engine import (
    strip_annotations,
    _extract_lijnen,
    _vergelijk_fills,
    _vergelijk_lijnen,
)
from test_kleur_dikte_matching import (
    bfs_cluster,
    cluster_bbox,
    to_255,
    best_kleur_match,
    rgb_dist,
    parallel_perp_afstand,
)

PT_MM = 25.4 / 72.0
MM_PT = 72.0 / 25.4

PROJECT = "helling"
PAGINA = 0
PDF_OUD = ROOT / "data" / PROJECT / "oud.pdf"
PDF_NIEUW = ROOT / "data" / PROJECT / "nieuw.pdf"
SIGS_PATH = ROOT / "references" / PROJECT / "signatures.json"

CLUSTER_AFSTAND_PT = 80.0
KLEUR_DREMPEL = 30.0

# Verwachte muurdiktes (in punten via mm)
TOL_DIKTE_MM = 15.0
GIBO_OUD_PT = 70 * MM_PT       # ~198.4
GIBO_NIEUW_PT = 100 * MM_PT    # ~283.5
TOL_DIKTE_PT = TOL_DIKTE_MM * MM_PT


def in_zone(pos, zone):
    x, y = pos[0], pos[1]
    return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]


def bbox_in_zone(bbox, zone):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def find_parallel_pairs(lijnen, zone, dikte_min_pt, dikte_max_pt, max_n=2000):
    in_lijnen = [
        l for l in lijnen
        if "van" in l
        and zone[0] <= l["van"][0] <= zone[2]
        and zone[1] <= l["van"][1] <= zone[3]
    ][:max_n]
    pairs = []
    n = len(in_lijnen)
    for i in range(n):
        for j in range(i + 1, n):
            d = parallel_perp_afstand(in_lijnen[i], in_lijnen[j])
            if d is None:
                continue
            if dikte_min_pt <= d <= dikte_max_pt:
                a = in_lijnen[i]
                b = in_lijnen[j]
                mx = (a["van"][0] + a["naar"][0] + b["van"][0] + b["naar"][0]) / 4
                my = (a["van"][1] + a["naar"][1] + b["van"][1] + b["naar"][1]) / 4
                pairs.append({
                    "afstand_pt": round(d, 1),
                    "afstand_mm": round(d * PT_MM, 1),
                    "mid": (round(mx, 1), round(my, 1)),
                    "stroke_a": a.get("stroke"),
                    "stroke_b": b.get("stroke"),
                })
    return pairs, len(in_lijnen)


def main():
    if not (PDF_OUD.exists() and PDF_NIEUW.exists() and SIGS_PATH.exists()):
        print("FAIL: input ontbreekt")
        return 1

    sigs = json.loads(SIGS_PATH.read_text(encoding="utf-8"))["wandtypes"]
    sig_by_name = {s["naam"]: s for s in sigs}

    GIBO_ZWAAR = sig_by_name.get("Gibo zwaar 70mm", {}).get("dominante_rgb")
    HARDSCHUIM = sig_by_name.get("hardschuimisolatie", {}).get("dominante_rgb")
    print(f"[load] {len(sigs)} signatures")
    print(f"  Gibo zwaar 70mm dominante_rgb: {GIBO_ZWAAR}")
    print(f"  hardschuimisolatie dominante_rgb: {HARDSCHUIM}")

    # 0. Pipeline: extract + diff
    oud_clean = strip_annotations(str(PDF_OUD))
    nieuw_clean = strip_annotations(str(PDF_NIEUW))
    oud_doc = fitz.open(oud_clean)
    nieuw_doc = fitz.open(nieuw_clean)
    try:
        oud_page = oud_doc[PAGINA]
        nieuw_page = nieuw_doc[PAGINA]
        page_w = nieuw_page.rect.width
        page_h = nieuw_page.rect.height
        print(f"[page] {page_w:.1f} x {page_h:.1f} pt = {page_w*PT_MM:.0f} x {page_h*PT_MM:.0f} mm")

        oud_items = _extract_lijnen(oud_page)
        nieuw_items = _extract_lijnen(nieuw_page)
        fill_g, fills_to, fills_ve = _vergelijk_fills(oud_items, nieuw_items)
        _, _, lijnen_to, lijnen_ve = _vergelijk_lijnen(oud_items, nieuw_items)
        oud_lijnen_alle = [i for i in oud_items if "van" in i]
        nieuw_lijnen_alle = [i for i in nieuw_items if "van" in i]

        # 1. Bouw wijzigingen + clusters om cluster 0 bbox te vinden
        wijzigingen = []
        for f in fills_to:
            wijzigingen.append({"kind": "fill_to", "pos": tuple(f["pos"]),
                                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0]+1, f["pos"][1]+1]),
                                "rgb255": to_255(f["rgb"])})
        for f in fills_ve:
            wijzigingen.append({"kind": "fill_ve", "pos": tuple(f["pos"]),
                                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0]+1, f["pos"][1]+1]),
                                "rgb255": to_255(f["rgb"])})
        for f in fill_g:
            wijzigingen.append({"kind": "fill_ge", "pos": tuple(f["pos"]),
                                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0]+1, f["pos"][1]+1]),
                                "rgb255_oud": to_255(f["oud_rgb"]),
                                "rgb255_nieuw": to_255(f["nieuw_rgb"])})
        for l in lijnen_to:
            wijzigingen.append({"kind": "lijn_to", "pos": tuple(l["van"]),
                                "bbox": [min(l["van"][0], l["naar"][0]), min(l["van"][1], l["naar"][1]),
                                         max(l["van"][0], l["naar"][0]), max(l["van"][1], l["naar"][1])],
                                "stroke": l.get("stroke")})
        for l in lijnen_ve:
            wijzigingen.append({"kind": "lijn_ve", "pos": tuple(l["van"]),
                                "bbox": [min(l["van"][0], l["naar"][0]), min(l["van"][1], l["naar"][1]),
                                         max(l["van"][0], l["naar"][0]), max(l["van"][1], l["naar"][1])],
                                "stroke": l.get("stroke")})

        clusters = bfs_cluster(wijzigingen, CLUSTER_AFSTAND_PT)
        clusters = [c for c in clusters if len(c) >= 2]
        clusters.sort(key=len, reverse=True)
        print(f"[pipeline] {len(wijzigingen)} wijzigingen, {len(clusters)} clusters")

        c0 = clusters[0]
        c0_bbox = cluster_bbox(wijzigingen, c0)
        c0_w = c0_bbox[2] - c0_bbox[0]
        c0_h = c0_bbox[3] - c0_bbox[1]
        print(f"\n=== STAP 1: cluster 0 + deel 4 ===")
        print(f"  cluster 0: n={len(c0)}, bbox={[round(v, 1) for v in c0_bbox]}")
        print(f"             {c0_w:.0f} x {c0_h:.0f} pt = {c0_w*PT_MM:.0f} x {c0_h*PT_MM:.0f} mm")

        # Deel 4 = rechter kwart van cluster 0
        deel4 = [
            c0_bbox[0] + 0.75 * c0_w,
            c0_bbox[1],
            c0_bbox[2],
            c0_bbox[3],
        ]
        print(f"  deel 4 (rechter 1/4): bbox={[round(v, 1) for v in deel4]}")
        print(f"             {deel4[2]-deel4[0]:.0f} x {deel4[3]-deel4[1]:.0f} pt = "
              f"{(deel4[2]-deel4[0])*PT_MM:.0f} x {(deel4[3]-deel4[1])*PT_MM:.0f} mm")

        # 2. Pixel-verschillen binnen deel 4
        print(f"\n=== STAP 2: pixel-verschillen in deel 4 ===")
        zone_fills_to = [f for f in fills_to if bbox_in_zone(f.get("bbox", [0]*4), deel4)]
        zone_fills_ve = [f for f in fills_ve if bbox_in_zone(f.get("bbox", [0]*4), deel4)]
        zone_fills_ge = [f for f in fill_g if bbox_in_zone(f.get("bbox", [0]*4), deel4)]
        zone_lijnen_to = [l for l in lijnen_to if in_zone(l["van"], deel4) or in_zone(l["naar"], deel4)]
        zone_lijnen_ve = [l for l in lijnen_ve if in_zone(l["van"], deel4) or in_zone(l["naar"], deel4)]
        print(f"  fills:  {len(zone_fills_to)} toegevoegd, {len(zone_fills_ve)} verdwenen, {len(zone_fills_ge)} gewijzigd")
        print(f"  lijnen: {len(zone_lijnen_to)} toegevoegd, {len(zone_lijnen_ve)} verdwenen")

        # 3. Kruis alle fills met signatures
        print(f"\n=== STAP 3: kleur-matching alle fills in zone ===")
        rgb_counter = defaultdict(int)
        match_counter = defaultdict(int)
        for f in zone_fills_to + zone_fills_ve:
            rgb = to_255(f["rgb"])
            rgb_counter[rgb] += 1
            sig, d = best_kleur_match(rgb, sigs, drempel=KLEUR_DREMPEL)
            if sig:
                match_counter[sig["naam"]] += 1
        for f in zone_fills_ge:
            for kk in ("oud_rgb", "nieuw_rgb"):
                rgb = to_255(f[kk])
                rgb_counter[rgb] += 1
                sig, d = best_kleur_match(rgb, sigs, drempel=KLEUR_DREMPEL)
                if sig:
                    match_counter[sig["naam"]] += 1

        print(f"  unieke fill-RGBs in zone: {len(rgb_counter)}")
        print(f"  top 8 RGB-frequenties:")
        for rgb, cnt in sorted(rgb_counter.items(), key=lambda x: -x[1])[:8]:
            sig, d = best_kleur_match(rgb, sigs, drempel=999)
            naam = sig["naam"] if sig and d <= KLEUR_DREMPEL else f"(geen match, dichtst: {sig['naam'] if sig else '?'} d={d:.0f})"
            print(f"    {cnt:>4}  rgb={rgb}  -> {naam}")
        print(f"\n  match-tellingen (binnen drempel {KLEUR_DREMPEL}):")
        for naam, cnt in sorted(match_counter.items(), key=lambda x: -x[1]):
            print(f"    {cnt:>4}  {naam}")

        # Kleuren van lijnen
        print(f"\n  unieke lijnkleuren in zone (alleen niet-zwart/wit/grijs):")
        lijn_kleur_counter = defaultdict(int)
        for l in zone_lijnen_to + zone_lijnen_ve:
            stroke = l.get("stroke")
            if not stroke:
                continue
            rgb = to_255(stroke)
            r, g, b = rgb
            if abs(r - g) < 5 and abs(g - b) < 5 and abs(r - b) < 5:
                continue
            lijn_kleur_counter[rgb] += 1
        for rgb, cnt in sorted(lijn_kleur_counter.items(), key=lambda x: -x[1])[:8]:
            sig, d = best_kleur_match(rgb, sigs, drempel=999)
            naam = sig["naam"] if sig and d <= KLEUR_DREMPEL else f"(geen match, dichtst: {sig['naam'] if sig else '?'} d={d:.0f})"
            print(f"    {cnt:>4}  rgb={rgb}  -> {naam}")

        # 4. Specifiek zoeken naar GT-wandtypes
        print(f"\n=== STAP 4: specifiek GT-wandtype-zoek ===")

        def near_gt(rgb, target, tol=KLEUR_DREMPEL):
            return rgb_dist(rgb, target) <= tol

        if GIBO_ZWAAR:
            gibo_fills = []
            for f in zone_fills_to:
                if near_gt(to_255(f["rgb"]), GIBO_ZWAAR):
                    gibo_fills.append(("fill_to", f))
            for f in zone_fills_ve:
                if near_gt(to_255(f["rgb"]), GIBO_ZWAAR):
                    gibo_fills.append(("fill_ve", f))
            for f in zone_fills_ge:
                if near_gt(to_255(f["oud_rgb"]), GIBO_ZWAAR) or near_gt(to_255(f["nieuw_rgb"]), GIBO_ZWAAR):
                    gibo_fills.append(("fill_ge", f))
            gibo_lijnen = []
            for l in zone_lijnen_to + zone_lijnen_ve:
                stroke = l.get("stroke")
                if stroke and near_gt(to_255(stroke), GIBO_ZWAAR):
                    gibo_lijnen.append(l)
            print(f"  Gibo zwaar [62,30,9] ± 30 in zone:")
            print(f"    fills:  {len(gibo_fills)}")
            print(f"    lijnen: {len(gibo_lijnen)}")
            for kind, f in gibo_fills[:5]:
                print(f"      {kind} bbox={f.get('bbox')}  rgb={to_255(f['rgb']) if 'rgb' in f else 'gewijzigd'}")
            for l in gibo_lijnen[:5]:
                print(f"      lijn van={l['van']} naar={l['naar']} stroke={to_255(l['stroke'])}")

        if HARDSCHUIM:
            hard_fills = []
            for f in zone_fills_to:
                if near_gt(to_255(f["rgb"]), HARDSCHUIM):
                    hard_fills.append(("fill_to", f))
            for f in zone_fills_ve:
                if near_gt(to_255(f["rgb"]), HARDSCHUIM):
                    hard_fills.append(("fill_ve", f))
            for f in zone_fills_ge:
                if near_gt(to_255(f["oud_rgb"]), HARDSCHUIM) or near_gt(to_255(f["nieuw_rgb"]), HARDSCHUIM):
                    hard_fills.append(("fill_ge", f))
            hard_lijnen = []
            for l in zone_lijnen_to + zone_lijnen_ve:
                stroke = l.get("stroke")
                if stroke and near_gt(to_255(stroke), HARDSCHUIM):
                    hard_lijnen.append(l)
            print(f"  hardschuimisolatie [131,134,125] ± 30 in zone:")
            print(f"    fills:  {len(hard_fills)}")
            print(f"    lijnen: {len(hard_lijnen)}")
            for kind, f in hard_fills[:5]:
                print(f"      {kind} bbox={f.get('bbox')}")
            for l in hard_lijnen[:5]:
                print(f"      lijn van={l['van']} naar={l['naar']} stroke={to_255(l['stroke'])}")

        # 5. Geometrische check op dikte-wijziging
        print(f"\n=== STAP 5: parallelle lijnpaar-diktes in zone ===")
        print(f"  doelranges: 70mm = {GIBO_OUD_PT:.1f}pt ± {TOL_DIKTE_PT:.1f}pt   "
              f"100mm = {GIBO_NIEUW_PT:.1f}pt ± {TOL_DIKTE_PT:.1f}pt")

        oud_70mm_pairs, oud_n = find_parallel_pairs(
            oud_lijnen_alle, deel4, GIBO_OUD_PT - TOL_DIKTE_PT, GIBO_OUD_PT + TOL_DIKTE_PT
        )
        oud_100mm_pairs, _ = find_parallel_pairs(
            oud_lijnen_alle, deel4, GIBO_NIEUW_PT - TOL_DIKTE_PT, GIBO_NIEUW_PT + TOL_DIKTE_PT
        )
        nieuw_70mm_pairs, nieuw_n = find_parallel_pairs(
            nieuw_lijnen_alle, deel4, GIBO_OUD_PT - TOL_DIKTE_PT, GIBO_OUD_PT + TOL_DIKTE_PT
        )
        nieuw_100mm_pairs, _ = find_parallel_pairs(
            nieuw_lijnen_alle, deel4, GIBO_NIEUW_PT - TOL_DIKTE_PT, GIBO_NIEUW_PT + TOL_DIKTE_PT
        )

        print(f"  alle lijnen in deel 4 zone: oud={oud_n}, nieuw={nieuw_n}")
        print(f"  parallel pairs ~70mm:  oud={len(oud_70mm_pairs)}  nieuw={len(nieuw_70mm_pairs)}")
        print(f"  parallel pairs ~100mm: oud={len(oud_100mm_pairs)} nieuw={len(nieuw_100mm_pairs)}")

        # Zoek paren waarvan midden in oud bij geen-overeenkomst zit in nieuw met andere dikte
        # i.e. oud-pair op locatie X met afstand 70mm, nieuw-pair op locatie X met afstand 100mm
        kandidaten_dikteshift = []
        for op in oud_70mm_pairs:
            for np_ in nieuw_100mm_pairs:
                dx = op["mid"][0] - np_["mid"][0]
                dy = op["mid"][1] - np_["mid"][1]
                if math.hypot(dx, dy) <= 50.0:
                    kandidaten_dikteshift.append({
                        "oud_mid": op["mid"], "oud_dikte_mm": op["afstand_mm"],
                        "nieuw_mid": np_["mid"], "nieuw_dikte_mm": np_["afstand_mm"],
                        "afstand_paren_pt": round(math.hypot(dx, dy), 1),
                    })
        print(f"  kandidaten 70->100mm shift (oud-pair binnen 50pt van nieuw-pair):  {len(kandidaten_dikteshift)}")
        for k in kandidaten_dikteshift[:8]:
            print(f"    oud_mid={k['oud_mid']}  d_oud={k['oud_dikte_mm']}mm   "
                  f"nieuw_mid={k['nieuw_mid']}  d_nieuw={k['nieuw_dikte_mm']}mm")

        # 6. Eindrapport
        print(f"\n=== EINDRAPPORT ===")
        gibo_signal = (GIBO_ZWAAR is not None and (len(gibo_fills) > 0 or len(gibo_lijnen) > 0)) or len(kandidaten_dikteshift) > 0
        hard_signal = HARDSCHUIM is not None and (len(hard_fills) > 0 or len(hard_lijnen) > 0)
        totaal_pixel_diff = (len(zone_fills_to) + len(zone_fills_ve) + len(zone_fills_ge) +
                             len(zone_lijnen_to) + len(zone_lijnen_ve))
        print(f"  Pixel-verschillen totaal in deel 4 zone: {totaal_pixel_diff}")
        print(f"  Fill/lijn met Gibo-zwaar kleur:          {len(gibo_fills)} fills, {len(gibo_lijnen)} lijnen")
        print(f"  Parallel-paar dikte-shift kandidaten:    {len(kandidaten_dikteshift)}")
        print(f"  Fill/lijn met hardschuim kleur:          {len(hard_fills)} fills, {len(hard_lijnen)} lijnen")
        print()
        print(f"  GT-1 (Gibo zwaar dikte-shift) detecteerbaar?  "
              f"{'JA' if gibo_signal else 'NEE'}")
        print(f"  GT-2 (Hardschuimisolatie toegevoegd) detecteerbaar?  "
              f"{'JA' if hard_signal else 'NEE'}")

        # Conclusie
        print()
        if gibo_signal and hard_signal:
            print("  CONCLUSIE: beide GT-wijzigingen zijn aanwezig in pixel-diff data, "
                  "maar de kleurkeuzes uit signatures.json zijn ongeschikt voor matching.")
        elif gibo_signal or hard_signal:
            print("  CONCLUSIE: slechts één van beide GT-wijzigingen is detecteerbaar uit "
                  "huidige pixel-diff + signatures combinatie.")
        else:
            print("  CONCLUSIE: GT-wijzigingen zijn met huidige signatures-kleuren NIET "
                  "te detecteren in pixel-diff data van deel 4 zone. Andere features nodig "
                  "(arcering-patroon, geometrie, of grotere zoekzone).")

        return 0
    finally:
        oud_doc.close()
        nieuw_doc.close()
        for p in (oud_clean, nieuw_clean):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())

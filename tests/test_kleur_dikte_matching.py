"""
Stap 2 + 3 van het wand-detectie plan, alleen Helling.

Pipeline:
1. Pixel-diff via app.diff_engine private functies (alle gewijzigde fills + lijnen).
2. BFS-clustering op gewijzigde items (spatial grid, threshold 80pt).
3. Per cluster:
   - kleur-match tegen references/helling/signatures.json (Euclidisch in 0-255 RGB, drempel 30)
   - dikte uit parallelle lijnparen
4. Validatie tegen ground truth: 2 verwachte wijzigingen.

Crops van top 10 clusters in <tmp>/helling_kleur_dikte/.

Geen Vision-calls.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz

from app.diff_engine import (
    strip_annotations,
    _extract_lijnen,
    _vergelijk_fills,
    _vergelijk_lijnen,
)

PROJECT = "helling"
PAGINA = 0
PDF_OUD = ROOT / "data" / PROJECT / "oud.pdf"
PDF_NIEUW = ROOT / "data" / PROJECT / "nieuw.pdf"
SIGNATURES_PATH = ROOT / "references" / PROJECT / "signatures.json"

CLUSTER_AFSTAND_PT = 80.0
KLEUR_DREMPEL = 30.0
DIKTE_MIN_PT, DIKTE_MAX_PT = 5.0, 300.0
PT_MM = 25.4 / 72.0
TOP_N_CROPS = 10
CROP_DIR = Path(tempfile.gettempdir()) / "helling_kleur_dikte"


def to_255(rgb_01):
    return tuple(int(round(c * 255)) for c in rgb_01[:3])


def rgb_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def best_kleur_match(rgb255, signatures, drempel=KLEUR_DREMPEL):
    beste = None
    beste_d = float("inf")
    for sig in signatures:
        d = rgb_dist(rgb255, sig["dominante_rgb"])
        if d < beste_d:
            beste_d = d
            beste = sig
    if beste and beste_d <= drempel:
        return beste, beste_d
    return None, beste_d


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def line_length(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def parallel_perp_afstand(l1, l2):
    """Perpendiculaire afstand tussen twee parallelle lijnsegmenten.
    Geeft None als de lijnen niet (vrijwel) parallel zijn of niet overlappen
    in projectie-richting."""
    a1, a2 = l1["van"], l1["naar"]
    b1, b2 = l2["van"], l2["naar"]
    da = (a2[0] - a1[0], a2[1] - a1[1])
    db = (b2[0] - b1[0], b2[1] - b1[1])
    la = math.hypot(*da)
    lb = math.hypot(*db)
    if la < 1.0 or lb < 1.0:
        return None
    cross = da[0] * db[1] - da[1] * db[0]
    if abs(cross) / (la * lb) > 0.07:
        return None
    nx, ny = -da[1] / la, da[0] / la
    perp = abs((b1[0] - a1[0]) * nx + (b1[1] - a1[1]) * ny)
    tx, ty = da[0] / la, da[1] / la
    a_t0, a_t1 = 0.0, la
    b_t0 = (b1[0] - a1[0]) * tx + (b1[1] - a1[1]) * ty
    b_t1 = (b2[0] - a1[0]) * tx + (b2[1] - a1[1]) * ty
    bt_lo, bt_hi = min(b_t0, b_t1), max(b_t0, b_t1)
    overlap = min(a_t1, bt_hi) - max(a_t0, bt_lo)
    if overlap < 5.0:
        return None
    return perp


def cluster_dikte_pt(lijnen):
    if len(lijnen) < 2:
        return None
    diktes = []
    n = min(len(lijnen), 60)
    for i in range(n):
        for j in range(i + 1, n):
            d = parallel_perp_afstand(lijnen[i], lijnen[j])
            if d is None:
                continue
            if DIKTE_MIN_PT <= d <= DIKTE_MAX_PT:
                diktes.append(d)
    if not diktes:
        return None
    diktes.sort()
    return diktes[len(diktes) // 2]


def grid_key(x, y, cell):
    return (int(x // cell), int(y // cell))


def bfs_cluster(items, cluster_afstand=CLUSTER_AFSTAND_PT):
    """BFS-clustering op spatial grid. Items moeten 'pos' (x, y) hebben."""
    if not items:
        return []
    cell = cluster_afstand
    grid = defaultdict(list)
    for i, it in enumerate(items):
        gk = grid_key(it["pos"][0], it["pos"][1], cell)
        grid[gk].append(i)

    verwerkt = [False] * len(items)
    clusters = []
    for start in range(len(items)):
        if verwerkt[start]:
            continue
        cluster = [start]
        verwerkt[start] = True
        frontier = [start]
        while frontier:
            cur = frontier.pop()
            cx, cy = items[cur]["pos"]
            gk = grid_key(cx, cy, cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((gk[0] + dx, gk[1] + dy), ()):
                        if verwerkt[j]:
                            continue
                        jx, jy = items[j]["pos"]
                        if math.hypot(jx - cx, jy - cy) <= cluster_afstand:
                            verwerkt[j] = True
                            cluster.append(j)
                            frontier.append(j)
        clusters.append(cluster)
    return clusters


def cluster_bbox(items, indices):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i in indices:
        b = items[i].get("bbox")
        if b and len(b) == 4:
            xs0.append(b[0]); ys0.append(b[1]); xs1.append(b[2]); ys1.append(b[3])
        else:
            x, y = items[i]["pos"]
            xs0.append(x); ys0.append(y); xs1.append(x); ys1.append(y)
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def main():
    t_start = time.time()

    # 1. Inputs
    if not PDF_OUD.exists() or not PDF_NIEUW.exists():
        print(f"FAIL: data PDFs ontbreken in {PDF_OUD.parent}")
        return 1
    if not SIGNATURES_PATH.exists():
        print(f"FAIL: {SIGNATURES_PATH} ontbreekt — run tools/build_signatures.py {PROJECT}")
        return 1

    signatures = json.loads(SIGNATURES_PATH.read_text(encoding="utf-8"))["wandtypes"]
    print(f"[load] {len(signatures)} signatures uit {SIGNATURES_PATH.name}")

    # 2. Pixel-diff
    print(f"[diff] {PDF_OUD.name} vs {PDF_NIEUW.name}, pagina {PAGINA + 1}")
    oud_clean = strip_annotations(str(PDF_OUD))
    nieuw_clean = strip_annotations(str(PDF_NIEUW))
    oud_doc = fitz.open(oud_clean)
    nieuw_doc = fitz.open(nieuw_clean)
    try:
        oud_page = oud_doc[PAGINA]
        nieuw_page = nieuw_doc[PAGINA]
        oud_items = _extract_lijnen(oud_page)
        nieuw_items = _extract_lijnen(nieuw_page)

        fill_gewijzigd, fills_toegevoegd, fills_verdwenen = _vergelijk_fills(oud_items, nieuw_items)
        _, _, lijnen_toegevoegd, lijnen_verdwenen = _vergelijk_lijnen(oud_items, nieuw_items)

        oud_lijnen_alle = [i for i in oud_items if "van" in i]
        nieuw_lijnen_alle = [i for i in nieuw_items if "van" in i]

        print(f"  fills: {len(fill_gewijzigd)} gewijzigd, {len(fills_toegevoegd)} toegevoegd, {len(fills_verdwenen)} verdwenen")
        print(f"  lijnen: {len(lijnen_toegevoegd)} toegevoegd, {len(lijnen_verdwenen)} verdwenen")
        print(f"  lijnen totaal: {len(oud_lijnen_alle)} oud, {len(nieuw_lijnen_alle)} nieuw")

        # 3. Verzamel wijzigingen als items met 'pos' en 'bbox'
        wijzigingen = []
        for f in fills_toegevoegd:
            wijzigingen.append({
                "kind": "fill_toegevoegd",
                "pos": tuple(f["pos"]),
                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0] + 1, f["pos"][1] + 1]),
                "rgb255": to_255(f["rgb"]),
            })
        for f in fills_verdwenen:
            wijzigingen.append({
                "kind": "fill_verdwenen",
                "pos": tuple(f["pos"]),
                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0] + 1, f["pos"][1] + 1]),
                "rgb255": to_255(f["rgb"]),
            })
        for f in fill_gewijzigd:
            wijzigingen.append({
                "kind": "fill_gewijzigd",
                "pos": tuple(f["pos"]),
                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0] + 1, f["pos"][1] + 1]),
                "rgb255_oud": to_255(f["oud_rgb"]),
                "rgb255_nieuw": to_255(f["nieuw_rgb"]),
            })
        for l in lijnen_toegevoegd:
            wijzigingen.append({
                "kind": "lijn_toegevoegd",
                "pos": tuple(l["van"]),
                "bbox": [min(l["van"][0], l["naar"][0]), min(l["van"][1], l["naar"][1]),
                         max(l["van"][0], l["naar"][0]), max(l["van"][1], l["naar"][1])],
                "lijn": l,
            })
        for l in lijnen_verdwenen:
            wijzigingen.append({
                "kind": "lijn_verdwenen",
                "pos": tuple(l["van"]),
                "bbox": [min(l["van"][0], l["naar"][0]), min(l["van"][1], l["naar"][1]),
                         max(l["van"][0], l["naar"][0]), max(l["van"][1], l["naar"][1])],
                "lijn": l,
            })

        if not wijzigingen:
            print("[result] geen wijzigingen gedetecteerd")
            return 0

        # 4. BFS-clustering
        t0 = time.time()
        clusters = bfs_cluster(wijzigingen, CLUSTER_AFSTAND_PT)
        clusters = [c for c in clusters if len(c) >= 2]
        clusters.sort(key=len, reverse=True)
        print(f"[cluster] {len(wijzigingen)} wijzigingen -> {len(clusters)} clusters in {time.time() - t0:.1f}s")

        # 5. Analyse per cluster
        resultaten = []
        for cid, idx_list in enumerate(clusters):
            bbox = cluster_bbox(wijzigingen, idx_list)
            kinds = [wijzigingen[i]["kind"] for i in idx_list]

            kleur_kandidaten = {}  # naam -> aantal hits
            kleur_summen = {}      # naam -> som van afstand
            for i in idx_list:
                w = wijzigingen[i]
                if w["kind"] in ("fill_toegevoegd", "fill_verdwenen"):
                    sig, d = best_kleur_match(w["rgb255"], signatures)
                    if sig:
                        n = sig["naam"]
                        kleur_kandidaten[n] = kleur_kandidaten.get(n, 0) + 1
                        kleur_summen[n] = kleur_summen.get(n, 0) + d
                elif w["kind"] == "fill_gewijzigd":
                    for kk in ("rgb255_oud", "rgb255_nieuw"):
                        sig, d = best_kleur_match(w[kk], signatures)
                        if sig:
                            n = sig["naam"]
                            kleur_kandidaten[n] = kleur_kandidaten.get(n, 0) + 1
                            kleur_summen[n] = kleur_summen.get(n, 0) + d
            wandtype = "onbekend"
            kleur_score = None
            if kleur_kandidaten:
                wandtype = max(kleur_kandidaten, key=lambda k: (kleur_kandidaten[k], -kleur_summen[k]))
                kleur_score = round(kleur_summen[wandtype] / kleur_kandidaten[wandtype], 1)

            x0, y0, x1, y1 = bbox
            oud_in = [l for l in oud_lijnen_alle
                      if x0 - 5 <= l["van"][0] <= x1 + 5 and y0 - 5 <= l["van"][1] <= y1 + 5]
            nieuw_in = [l for l in nieuw_lijnen_alle
                        if x0 - 5 <= l["van"][0] <= x1 + 5 and y0 - 5 <= l["van"][1] <= y1 + 5]
            d_oud = cluster_dikte_pt(oud_in)
            d_nieuw = cluster_dikte_pt(nieuw_in)

            n_to = sum(1 for k in kinds if k.endswith("toegevoegd"))
            n_ve = sum(1 for k in kinds if k.endswith("verdwenen"))
            n_ge = sum(1 for k in kinds if k.endswith("gewijzigd"))
            if n_ge > 0 and n_ge >= max(n_to, n_ve):
                wijzig_type = "gewijzigd"
            elif n_to > n_ve * 2:
                wijzig_type = "toegevoegd"
            elif n_ve > n_to * 2:
                wijzig_type = "verdwenen"
            else:
                wijzig_type = "gemengd"

            if wandtype != "onbekend" and (d_oud is not None or d_nieuw is not None):
                confidence = "hoog"
            elif wandtype != "onbekend":
                confidence = "midden"
            else:
                confidence = "laag"

            resultaten.append({
                "cluster_id": cid,
                "n_items": len(idx_list),
                "bbox": [round(v, 1) for v in bbox],
                "wijzig_type": wijzig_type,
                "wandtype": wandtype,
                "kleur_score": kleur_score,
                "dikte_oud_mm": round(d_oud * PT_MM, 1) if d_oud else None,
                "dikte_nieuw_mm": round(d_nieuw * PT_MM, 1) if d_nieuw else None,
                "confidence": confidence,
            })

        # 6. Crops top N
        CROP_DIR.mkdir(parents=True, exist_ok=True)
        for r in resultaten[:TOP_N_CROPS]:
            x0, y0, x1, y1 = r["bbox"]
            pad = 20
            for label, page in (("oud", oud_page), ("nieuw", nieuw_page)):
                clip = fitz.Rect(max(0, x0 - pad), max(0, y0 - pad),
                                 min(page.rect.width, x1 + pad), min(page.rect.height, y1 + pad)) & page.rect
                if clip.is_empty:
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
                pix.save(str(CROP_DIR / f"cluster_{r['cluster_id']:03d}_{label}.png"))
        print(f"[crops] top {min(TOP_N_CROPS, len(resultaten))} cluster-paren -> {CROP_DIR}")

        # 7. Print top resultaten
        print("\n--- top 15 clusters ---")
        for r in resultaten[:15]:
            print(f"  c{r['cluster_id']:>3}  n={r['n_items']:>4}  {r['wijzig_type']:<10}  "
                  f"{r['wandtype']:<35}  d_oud={str(r['dikte_oud_mm']):>6}  d_nieuw={str(r['dikte_nieuw_mm']):>6}  "
                  f"conf={r['confidence']}")

        # 8. Ground truth validatie
        print("\n=== GROUND TRUTH VALIDATIE ===")
        gt = [
            {"naam": "Gibo zwaar dikte 70mm -> 100mm", "type": "dikte_gewijzigd",
             "wandtype_match": "gibo zwaar"},
            {"naam": "Hardschuimisolatie toegevoegd", "type": "toegevoegd",
             "wandtype_match": "hardschuimisolatie"},
        ]

        # 1: Gibo zwaar — cluster met wandtype bevattend "gibo zwaar" en dikte-verandering
        gibo_clusters = [r for r in resultaten if "gibo zwaar" in r["wandtype"].lower()]
        gibo_dikte_change = [
            r for r in gibo_clusters
            if r["dikte_oud_mm"] is not None and r["dikte_nieuw_mm"] is not None
            and abs(r["dikte_oud_mm"] - r["dikte_nieuw_mm"]) >= 10
        ]
        gibo_any_dikte = [r for r in gibo_clusters if r["dikte_oud_mm"] or r["dikte_nieuw_mm"]]

        # 2: Hardschuimisolatie — toegevoegd cluster met dat wandtype
        hard_clusters = [r for r in resultaten if "hardschuimisolatie" in r["wandtype"].lower()]
        hard_toegevoegd = [r for r in hard_clusters if r["wijzig_type"] in ("toegevoegd", "gemengd")]

        gevonden = 0

        print(f"  GT-1: Gibo zwaar dikte-wijziging (70mm -> 100mm)")
        if gibo_dikte_change:
            print(f"        WEL gevonden — {len(gibo_dikte_change)} cluster(s):")
            for r in gibo_dikte_change[:3]:
                print(f"          c{r['cluster_id']}  d_oud={r['dikte_oud_mm']}mm  d_nieuw={r['dikte_nieuw_mm']}mm  bbox={r['bbox']}")
            gevonden += 1
        elif gibo_any_dikte:
            print(f"        DEELS — {len(gibo_clusters)} Gibo zwaar cluster(s) maar geen duidelijke dikte-shift")
        elif gibo_clusters:
            print(f"        DEELS — {len(gibo_clusters)} Gibo zwaar cluster(s) maar geen dikte gemeten")
        else:
            print(f"        NIET gevonden — geen Gibo zwaar cluster (kleur ~[62,30,9])")

        print(f"  GT-2: Hardschuimisolatie toegevoegd")
        if hard_toegevoegd:
            print(f"        WEL gevonden — {len(hard_toegevoegd)} cluster(s):")
            for r in hard_toegevoegd[:3]:
                print(f"          c{r['cluster_id']}  type={r['wijzig_type']}  n={r['n_items']}  bbox={r['bbox']}")
            gevonden += 1
        elif hard_clusters:
            print(f"        DEELS — {len(hard_clusters)} hardschuimisolatie cluster(s) maar niet als 'toegevoegd' geclassificeerd")
        else:
            print(f"        NIET gevonden — geen hardschuimisolatie cluster (kleur ~[131,134,125])")

        # 9. Stats
        totaal = len(resultaten)
        per_type = defaultdict(int)
        for r in resultaten:
            per_type[r["wandtype"]] += 1

        print(f"\n=== STATS ===")
        print(f"  Recall:        {gevonden}/{len(gt)} ground truth wijzigingen gevonden")
        print(f"  Totaal clusters: {totaal} ({sum(1 for r in resultaten if r['wandtype'] != 'onbekend')} met wandtype-match)")
        print(f"  Top wandtypes (cluster-count):")
        for naam, cnt in sorted(per_type.items(), key=lambda x: -x[1])[:8]:
            print(f"    {cnt:>4}  {naam}")
        print(f"  Tijd:          {time.time() - t_start:.1f}s")
        print(f"  Vision-kosten: $0.00 (signatures gecached)")
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

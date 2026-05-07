"""
Diagnostiek wand_diff pipeline — geen code-wijzigingen, alleen data.
Drie onderzoeksvragen op 56 de Helling en Muiden D2.1.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import fitz
from app.wand_diff import (
    _extraheer_segmenten, _bouw_cost_matrix, _hungarian_match,
    _cluster_segmenten, _match_wandtype, _kleur_afstand, _rnd,
)
from app.tekening_profiel import detecteer_orientatie, vind_legenda_combined
from app.config import DiffConfig

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")
SEP = "=" * 64

def open_pagina(pad, pnr=0):
    doc = fitz.open(pad)
    page = doc[pnr]
    ori = detecteer_orientatie(page)
    return doc, page, ori


# ============================================================
# VRAAG 1 — RGB-afstanden "type onbekend" vs legenda (56 de Helling)
# ============================================================
def vraag1():
    print(f"\n{SEP}")
    print("VRAAG 1 — RGB-afstanden 'type onbekend' vs legenda (56 de Helling)")
    print(SEP)

    oud_path  = os.path.join(BASE, "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf")
    nieuw_path = os.path.join(BASE, "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf")

    oud_doc, oud_page, oud_ori = open_pagina(oud_path)
    nieuw_doc, nieuw_page, nieuw_ori = open_pagina(nieuw_path)
    legenda = vind_legenda_combined(nieuw_page, nieuw_ori, api_key=None)
    nieuw_segs = _extraheer_segmenten(nieuw_page, nieuw_ori)
    oud_segs   = _extraheer_segmenten(oud_page, oud_ori)
    oud_doc.close(); nieuw_doc.close()

    cfg = DiffConfig()
    tol = cfg.wand_kleur_tolerantie

    # Alle onbekende segmenten (beide tekeningen samen)
    alle_segs = nieuw_segs + oud_segs
    onbekend = [s for s in alle_segs if _match_wandtype(s, legenda, tol) == "type onbekend"]

    print(f"\nLegenda ({len(legenda)} entries):")
    for rgb, naam in legenda.items():
        print(f"  RGB={rgb}  →  {naam!r}")

    print(f"\nAantal segmenten totaal (nieuw+oud): {len(alle_segs)}")
    print(f"Onbekend (match faalt bij tol={tol}): {len(onbekend)}")
    print(f"\nEerste 12 onbekende segmenten — RGB + kleinste afstand tot legenda:")
    print(f"  {'Seg-RGB':>40}  {'min-dist':>8}  {'dichtstbijzijnde legenda-kleur':>42}  naam")
    print(f"  {'-'*40}  {'-'*8}  {'-'*42}  ----")

    for seg in onbekend[:12]:
        seg_rgb = seg.get("kleur")
        if seg_rgb is None:
            print(f"  (geen kleur)")
            continue
        min_dist = float("inf")
        best_rgb = None
        best_naam = ""
        for leg_rgb, naam in legenda.items():
            d = _kleur_afstand(seg_rgb, leg_rgb)
            if d < min_dist:
                min_dist = d
                best_rgb = leg_rgb
                best_naam = naam
        print(f"  {str(seg_rgb):>40}  {min_dist:>8.4f}  {str(best_rgb):>42}  {best_naam!r}")

    # Ook: hoe groot is de afstandsverdeling voor ALLE onbekende?
    dists = []
    for seg in onbekend:
        seg_rgb = seg.get("kleur")
        if not seg_rgb:
            continue
        dists.append(min(_kleur_afstand(seg_rgb, leg_rgb) for leg_rgb in legenda))

    if dists:
        dists.sort()
        print(f"\nAfstandsverdeling alle {len(dists)} onbekende segmenten:")
        print(f"  min={dists[0]:.4f}  p25={dists[len(dists)//4]:.4f}  "
              f"mediaan={dists[len(dists)//2]:.4f}  p75={dists[3*len(dists)//4]:.4f}  "
              f"max={dists[-1]:.4f}")
        buckets = [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 9)]
        print(f"  Histogram (min-afstand tot dichtstbijzijnde legenda-kleur):")
        for lo, hi in buckets:
            cnt = sum(1 for d in dists if lo <= d < hi)
            print(f"    [{lo:.2f} – {hi:.2f}): {cnt:3d} segmenten")


# ============================================================
# VRAAG 2 — Segment-counts voor en na clustering (56 de Helling)
# ============================================================
def vraag2():
    print(f"\n{SEP}")
    print("VRAAG 2 — Segment-counts voor/na clustering (56 de Helling)")
    print(SEP)

    oud_path  = os.path.join(BASE, "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf")
    nieuw_path = os.path.join(BASE, "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf")

    oud_doc, oud_page, oud_ori = open_pagina(oud_path)
    nieuw_doc, nieuw_page, nieuw_ori = open_pagina(nieuw_path)
    legenda = vind_legenda_combined(nieuw_page, nieuw_ori, api_key=None)
    nieuw_segs = _extraheer_segmenten(nieuw_page, nieuw_ori)
    oud_segs   = _extraheer_segmenten(oud_page, oud_ori)
    oud_doc.close(); nieuw_doc.close()

    cfg = DiffConfig()
    tol = cfg.wand_kleur_tolerantie
    max_cost = cfg.wand_centroid_max_afstand
    cluster_afstand = cfg.wand_cluster_afstand

    print(f"\nSegmenten geëxtraheerd:")
    print(f"  Oud  tekening: {len(oud_segs)} segmenten")
    print(f"  Nieuw tekening: {len(nieuw_segs)} segmenten")

    matches, oud_ongemat, nieuw_ongemat = _hungarian_match(oud_segs, nieuw_segs, max_cost)
    print(f"\nNa Hungarian matching (max_cost={max_cost}):")
    print(f"  Gematchte paren  : {len(matches)}")
    print(f"  Oud ongemat      : {len(oud_ongemat)}  → verdwenen")
    print(f"  Nieuw ongemat    : {len(nieuw_ongemat)} → nieuw")

    # Bouw pre-cluster lijst (zelfde als in bereken_wand_diff)
    pre_cluster = []
    for idx in nieuw_ongemat:
        seg = nieuw_segs[idx]
        wt = _match_wandtype(seg, legenda, tol)
        pre_cluster.append({"type": "nieuw", "wandtype": wt,
                             "positie": list(seg["centroid"]), "bbox": seg["bbox"]})
    for idx in oud_ongemat:
        seg = oud_segs[idx]
        wt = _match_wandtype(seg, legenda, tol)
        pre_cluster.append({"type": "verdwenen", "wandtype": wt,
                             "positie": list(seg["centroid"]), "bbox": seg["bbox"]})

    print(f"\nVóór clustering: {len(pre_cluster)} items")

    clusters = _cluster_segmenten(pre_cluster, cluster_afstand)
    print(f"Na clustering (afstand={cluster_afstand}pt): {len(clusters)} clusters")

    nieuw_cl  = [c for c in clusters if c["type"] == "nieuw"]
    verd_cl   = [c for c in clusters if c["type"] == "verdwenen"]
    print(f"  Nieuw clusters   : {len(nieuw_cl)}")
    print(f"  Verdwenen clusters: {len(verd_cl)}")

    # Toon cluster-groottes (hoeveel segmenten per cluster)
    # We reconstrueren dit door te kijken hoeveel pre_cluster items
    # in elk cluster zouden vallen (via afstand)
    from collections import Counter
    cluster_sizes = []
    used = set()
    for c in clusters:
        leden = []
        for i, p in enumerate(pre_cluster):
            if i in used:
                continue
            if p["wandtype"] != c["wandtype"] or p["type"] != c["type"]:
                continue
            dist = math.hypot(p["positie"][0] - c["positie"][0],
                              p["positie"][1] - c["positie"][1])
            if dist < cluster_afstand * 1.5:  # ruime marge voor terugberekening
                leden.append(i)
        cluster_sizes.append(len(leden))

    cnt = Counter(cluster_sizes)
    print(f"\n  Cluster-grootte verdeling (# segmenten → # clusters):")
    for size in sorted(cnt):
        print(f"    {size:2d} segment(en) → {cnt[size]} cluster(s)")


# ============================================================
# VRAAG 3 — Muiden D2.1: segmenten voor matching + cost-verdeling
# ============================================================
def vraag3():
    print(f"\n{SEP}")
    print("VRAAG 3 — Muiden D2.1: segmenten + cost-distributie ongematchten")
    print(SEP)

    oud_path  = os.path.join(BASE, "WT-PLG-D2.1_20250324_B.pdf")
    nieuw_path = os.path.join(BASE, "WT-PLG-D2.1_20260202_E.pdf")

    oud_doc, oud_page, oud_ori = open_pagina(oud_path)
    nieuw_doc, nieuw_page, nieuw_ori = open_pagina(nieuw_path)
    legenda = vind_legenda_combined(nieuw_page, nieuw_ori, api_key=None)
    nieuw_segs = _extraheer_segmenten(nieuw_page, nieuw_ori)
    oud_segs   = _extraheer_segmenten(oud_page, oud_ori)
    oud_doc.close(); nieuw_doc.close()

    cfg = DiffConfig()
    max_cost = cfg.wand_centroid_max_afstand

    print(f"\nSegmenten geëxtraheerd:")
    print(f"  Oud  tekening: {len(oud_segs)} segmenten")
    print(f"  Nieuw tekening: {len(nieuw_segs)} segmenten")

    # Bouw cost-matrix en inspecteer VOOR Hungarian
    cost = _bouw_cost_matrix(oud_segs, nieuw_segs, max_cost)

    # Kleinste cost per nieuw-segment (beste mogelijke match in oud)
    min_costs_nieuw = []
    for j in range(len(nieuw_segs)):
        col_costs = [cost[i][j] for i in range(len(oud_segs))]
        min_costs_nieuw.append(min(col_costs) if col_costs else float("inf"))

    # Kleinste cost per oud-segment
    min_costs_oud = []
    for i in range(len(oud_segs)):
        row_costs = cost[i]
        min_costs_oud.append(min(row_costs) if row_costs else float("inf"))

    def hist(dists, label):
        dists = sorted(dists)
        print(f"\n  {label} (n={len(dists)}):")
        print(f"    min={dists[0]:.1f}  p25={dists[len(dists)//4]:.1f}  "
              f"mediaan={dists[len(dists)//2]:.1f}  p75={dists[3*len(dists)//4]:.1f}  "
              f"max={dists[-1]:.1f}")
        buckets = [(0, 30), (30, 80), (80, 150), (150, 300), (300, 600), (600, 99999)]
        print(f"    Histogram (kleinste matchkost):")
        for lo, hi in buckets:
            cnt = sum(1 for d in dists if lo <= d < hi)
            bar = "#" * min(cnt, 40)
            label2 = f"≥{lo}" if hi == 99999 else f"{lo}–{hi}"
            print(f"      [{label2:>7}pt): {cnt:4d}  {bar}")
        onder = sum(1 for d in dists if d <= max_cost)
        boven = len(dists) - onder
        print(f"    → Onder drempel {max_cost}pt: {onder} ({100*onder//len(dists)}%)")
        print(f"    → Boven drempel {max_cost}pt: {boven} ({100*boven//len(dists)}%) → ongemat")

    if min_costs_nieuw:
        hist(min_costs_nieuw, "Nieuw-segmenten: kleinste kost naar oud")
    if min_costs_oud:
        hist(min_costs_oud, "Oud-segmenten: kleinste kost naar nieuw")

    # Voer ook de echte matching uit
    matches, oud_ongemat, nieuw_ongemat = _hungarian_match(oud_segs, nieuw_segs, max_cost)
    print(f"\nNa Hungarian matching (max_cost={max_cost}):")
    print(f"  Gematchte paren  : {len(matches)}")
    print(f"  Oud ongemat      : {len(oud_ongemat)}  → verdwenen")
    print(f"  Nieuw ongemat    : {len(nieuw_ongemat)} → nieuw")

    # Centroid-posities van ongematchte nieuw-segmenten (eerste 10)
    print(f"\n  Eerste 10 ongematchte NIEUW segmenten (positie + kleur):")
    for idx in nieuw_ongemat[:10]:
        s = nieuw_segs[idx]
        print(f"    pos={[round(v) for v in s['centroid']]}  "
              f"kleur={s['kleur']}  lengte={s['lengte']:.0f}pt")

    # Centroid-posities van ongematchte oud-segmenten (eerste 10)
    if oud_ongemat:
        print(f"\n  Ongematchte OUD segmenten:")
        for idx in oud_ongemat[:10]:
            s = oud_segs[idx]
            print(f"    pos={[round(v) for v in s['centroid']]}  "
                  f"kleur={s['kleur']}  lengte={s['lengte']:.0f}pt")

    # Extra: vergelijk centroid-bereiken oud vs nieuw
    if oud_segs and nieuw_segs:
        oud_xs = [s["centroid"][0] for s in oud_segs]
        oud_ys = [s["centroid"][1] for s in oud_segs]
        nieuw_xs = [s["centroid"][0] for s in nieuw_segs]
        nieuw_ys = [s["centroid"][1] for s in nieuw_segs]
        print(f"\n  Centroid-bereik oud  tekening:  x=[{min(oud_xs):.0f}..{max(oud_xs):.0f}]  "
              f"y=[{min(oud_ys):.0f}..{max(oud_ys):.0f}]")
        print(f"  Centroid-bereik nieuw tekening: x=[{min(nieuw_xs):.0f}..{max(nieuw_xs):.0f}]  "
              f"y=[{min(nieuw_ys):.0f}..{max(nieuw_ys):.0f}]")


if __name__ == "__main__":
    vraag1()
    vraag2()
    vraag3()
    print(f"\n{'=' * 64}\nDIAGNOSTIEK KLAAR\n")

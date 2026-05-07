"""Analyse van cluster 0 uit de hybride pipeline."""
import sys
from pathlib import Path
from collections import defaultdict
import fitz

sys.path.insert(0, ".")
from app.diff_engine import _extract_lijnen, _vergelijk_lijnen, _vergelijk_fills, strip_annotations

MVP_DIR   = Path(__file__).parent.parent / "Karregat & Koning MVP"
PDF_OUD   = MVP_DIR / "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"
PDF_NIEUW = MVP_DIR / "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"
CLUSTER_AFSTAND = 80.0
MIN_ITEMS = 3
MIN_BBOX_DIM = 50.0

oud_stripped  = strip_annotations(str(PDF_OUD))
nieuw_stripped = strip_annotations(str(PDF_NIEUW))
doc_oud   = fitz.open(oud_stripped)
doc_nieuw = fitz.open(nieuw_stripped)

oud_items    = _extract_lijnen(doc_oud[0])
nieuw_items  = _extract_lijnen(doc_nieuw[0])
# Exact zelfde als hybride script: alle items doorgeven
_wg, _kg, lijn_toegevoegd, lijn_verdwenen = _vergelijk_lijnen(oud_items, nieuw_items)
_fg, fill_toegevoegd, fill_verdwenen       = _vergelijk_fills(oud_items, nieuw_items)
print(f"lt={len(lijn_toegevoegd)} lv={len(lijn_verdwenen)} ft={len(fill_toegevoegd)} fv={len(fill_verdwenen)}")

items = []
for l in lijn_toegevoegd:
    mx = (l["van"][0] + l["naar"][0]) / 2
    my = (l["van"][1] + l["naar"][1]) / 2
    items.append({"_tag": "lijn_toegevoegd", "_x": mx, "_y": my, **l})
for l in lijn_verdwenen:
    mx = (l["van"][0] + l["naar"][0]) / 2
    my = (l["van"][1] + l["naar"][1]) / 2
    items.append({"_tag": "lijn_verdwenen", "_x": mx, "_y": my, **l})
for f in fill_toegevoegd:
    items.append({"_tag": "fill_toegevoegd", "_x": f["pos"][0], "_y": f["pos"][1], **f})
for f in fill_verdwenen:
    items.append({"_tag": "fill_verdwenen", "_x": f["pos"][0], "_y": f["pos"][1], **f})

# Raster-BFS
posities = [(it["_x"], it["_y"]) for it in items]
cel = CLUSTER_AFSTAND
raster: dict = defaultdict(list)
for i, (x, y) in enumerate(posities):
    raster[(int(x // cel), int(y // cel))].append(i)

bezoekt = [False] * len(items)
clusters_raw = []
for start in range(len(items)):
    if bezoekt[start]:
        continue
    bezoekt[start] = True
    groep = [start]
    wachtrij = [start]
    while wachtrij:
        huidig = wachtrij.pop()
        x0, y0 = posities[huidig]
        cx, cy = int(x0 // cel), int(y0 // cel)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for buur in raster.get((cx + dx, cy + dy), []):
                    if not bezoekt[buur]:
                        bx, by = posities[buur]
                        if abs(bx - x0) <= cel and abs(by - y0) <= cel:
                            bezoekt[buur] = True
                            groep.append(buur)
                            wachtrij.append(buur)
    clusters_raw.append(groep)

clusters = []
for groep in clusters_raw:
    if len(groep) < MIN_ITEMS:
        continue
    xs = [posities[i][0] for i in groep]
    ys = [posities[i][1] for i in groep]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    if (bx1 - bx0) < MIN_BBOX_DIM and (by1 - by0) < MIN_BBOX_DIM:
        continue
    clusters.append({"indices": groep, "bbox": (bx0, by0, bx1, by1)})

clusters.sort(key=lambda c: -len(c["indices"]))
c0 = clusters[0]
print(f"Cluster 0: {len(c0['indices'])} items  bbox={tuple(round(v) for v in c0['bbox'])}")


def is_rood_hex(h: str) -> bool:
    if not h or not h.startswith("#") or len(h) < 7:
        return False
    r = int(h[1:3], 16)
    g = int(h[3:5], 16)
    b = int(h[5:7], 16)
    return r > 180 and g < 100 and b < 100


def kleur_cat(it: dict) -> str:
    tag = it["_tag"]
    if "fill" in tag:
        rgb = it.get("rgb", (0.0, 0.0, 0.0))
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        if r > 0.7 and g < 0.4 and b < 0.4:
            return "rood"
        if r > 0.9 and g > 0.9 and b > 0.9:
            return "wit"
        if r < 0.15 and g < 0.15 and b < 0.15:
            return "zwart"
        return f"fill #{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    else:
        h = it.get("kleur", "")
        if is_rood_hex(h):
            return "rood"
        if not h or h == "#000000":
            return "zwart"
        r = int(h[1:3], 16) if h.startswith("#") and len(h) >= 7 else 0
        g = int(h[3:5], 16) if h.startswith("#") and len(h) >= 7 else 0
        b = int(h[5:7], 16) if h.startswith("#") and len(h) >= 7 else 0
        if r == g == b:
            return "grijs"
        return f"overig {h}"


tag_counts: dict = defaultdict(int)
kat_counts: dict = defaultdict(int)

for idx in c0["indices"]:
    it = items[idx]
    tag_counts[it["_tag"]] += 1
    kat_counts[kleur_cat(it)] += 1

print("\n=== TAGS ===")
for tag, cnt in sorted(tag_counts.items()):
    print(f"  {tag:<22}: {cnt:>5}")

print("\n=== KLEUREN ===")
for kat, cnt in sorted(kat_counts.items(), key=lambda x: -x[1]):
    print(f"  {kat:<35}: {cnt:>5}")

# Paar voorbeeld-items per kleur-categorie
print("\n=== VOORBEELDEN per kleur ===")
voorbeelden: dict = defaultdict(list)
for idx in c0["indices"]:
    it = items[idx]
    kat = kleur_cat(it)
    if len(voorbeelden[kat]) < 3:
        voorbeelden[kat].append(it)

for kat, its in sorted(voorbeelden.items()):
    print(f"\n  [{kat}]")
    for it in its:
        tag = it["_tag"]
        if "fill" in tag:
            print(f"    {tag} rgb={it.get('rgb')} pos={it.get('pos')}")
        else:
            print(f"    {tag} kleur={it.get('kleur')} van={it.get('van')} naar={it.get('naar')}")

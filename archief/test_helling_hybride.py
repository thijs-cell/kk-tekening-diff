"""
Hybride pipeline test: pixel-detectie (diff_engine) + Vision-classificatie per cluster.
56 de Helling, pagina 1.

Stap 1: pixel-diff via diff_engine._extract_lijnen / _vergelijk_lijnen / _vergelijk_fills
Stap 2: raster-BFS clustering (80pt, min 3 items, min 50x50pt bbox)
Stap 3: crop per cluster op 200 DPI
Stap 4: Vision-classificatie per cluster (sequentieel)
Stap 5: rapport

Gebruik: python test_helling_hybride.py
"""

import asyncio
import base64
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import fitz

# App-module beschikbaar maken zonder productie-code te wijzigen
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from app.diff_engine import (
    _extract_lijnen,
    _vergelijk_lijnen,
    _vergelijk_fills,
    strip_annotations,
)
from app.config import DiffConfig

# ---------------------------------------------------------------------------
# Constanten
# ---------------------------------------------------------------------------

MVP_DIR = SCRIPT_DIR.parent / "Karregat & Koning MVP"
REF_DIR = SCRIPT_DIR / "references" / "helling"
TMP_DIR = Path("/tmp/helling_hybride")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OUD_PDF   = MVP_DIR / "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"
NIEUW_PDF = MVP_DIR / "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"

PAGINA             = 0
MODEL              = "claude-sonnet-4-5"
RENDER_DPI         = 200
CLUSTER_AFSTAND    = 80.0    # pt
MIN_ITEMS          = 3
MIN_BBOX_DIM       = 50.0    # pt — beide dimensies
CROP_PAD           = 100.0   # pt padding rondom cluster-bbox

# ---------------------------------------------------------------------------
# Helpers: geometrie
# ---------------------------------------------------------------------------

def _item_pos(item: dict) -> tuple[float, float]:
    if "van" in item:
        return (
            (item["van"][0] + item["naar"][0]) / 2,
            (item["van"][1] + item["naar"][1]) / 2,
        )
    p = item.get("pos", (0.0, 0.0))
    return (float(p[0]), float(p[1]))


def _item_bbox(item: dict) -> tuple[float, float, float, float]:
    if "van" in item and "naar" in item:
        x0 = min(item["van"][0], item["naar"][0])
        y0 = min(item["van"][1], item["naar"][1])
        x1 = max(item["van"][0], item["naar"][0])
        y1 = max(item["van"][1], item["naar"][1])
        if x0 == x1: x1 += 0.5
        if y0 == y1: y1 += 0.5
        return (x0, y0, x1, y1)
    if "bbox" in item:
        b = item["bbox"]
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    px, py = _item_pos(item)
    return (px - 1, py - 1, px + 1, py + 1)


def _bbox_unie(bboxen: list) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in bboxen),
        min(b[1] for b in bboxen),
        max(b[2] for b in bboxen),
        max(b[3] for b in bboxen),
    )


# ---------------------------------------------------------------------------
# Helpers: data
# ---------------------------------------------------------------------------

def _laad_wandtypes() -> list[dict]:
    items, namen = [], set()
    for pad in [REF_DIR / "vision_renvooi_oud.json", REF_DIR / "vision_renvooi_nieuw.json"]:
        if not pad.exists():
            continue
        data = json.loads(pad.read_text(encoding="utf-8"))
        for w in data.get("wandtypes", []):
            k = w["naam"].lower().strip()
            if k not in namen:
                namen.add(k)
                items.append(w)
    return items


def _wandtypes_tekst(wt: list[dict]) -> str:
    return "\n".join(
        f"- {w['naam']} ({w.get('categorie','?')}): {w.get('visuele_kenmerken','')}"
        for w in wt
    )


def _parse_json(tekst: str) -> dict:
    tekst = tekst.strip()
    if "```" in tekst:
        for deel in tekst.split("```"):
            deel = deel.lstrip("json").strip()
            if deel.startswith("{"):
                tekst = deel
                break
    try:
        return json.loads(tekst)
    except Exception:
        pass
    s, e = tekst.find("{"), tekst.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(tekst[s:e])
        except Exception:
            pass
    return {}


def _pix_b64(pix: fitz.Pixmap) -> str:
    return base64.standard_b64encode(pix.tobytes("jpeg", jpg_quality=90)).decode()


# ---------------------------------------------------------------------------
# Stap 1 — Pixel-diff
# ---------------------------------------------------------------------------

def stap1_pixel_diff() -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 1 -- Pixel-diff via diff_engine")
    print("=" * 64)

    cfg = DiffConfig()
    oud_c  = strip_annotations(str(OUD_PDF))
    nieuw_c = strip_annotations(str(NIEUW_PDF))

    oud_doc  = fitz.open(oud_c);  oud_page  = oud_doc[PAGINA]
    nieuw_doc = fitz.open(nieuw_c); nieuw_page = nieuw_doc[PAGINA]

    oud_l  = _extract_lijnen(oud_page)
    nieuw_l = _extract_lijnen(nieuw_page)

    _, _, lt, lv = _vergelijk_lijnen(oud_l, nieuw_l, drempel=cfg.lijn_match_drempel)
    _, ft, fv    = _vergelijk_fills(oud_l,  nieuw_l, drempel=cfg.fill_match_drempel)

    oud_doc.close(); nieuw_doc.close()

    print(f"  lijnen_toegevoegd : {len(lt)}")
    print(f"  lijnen_verdwenen  : {len(lv)}")
    print(f"  fills_toegevoegd  : {len(ft)}")
    print(f"  fills_verdwenen   : {len(fv)}")

    alle = []
    for items, tag in [(lt, "lijn_toegevoegd"), (lv, "lijn_verdwenen"),
                       (ft, "fill_toegevoegd"), (fv, "fill_verdwenen")]:
        for item in items:
            alle.append({**item, "_tag": tag})

    print(f"  Totaal            : {len(alle)}")
    return alle


# ---------------------------------------------------------------------------
# Stap 2 — Raster-BFS clustering
# ---------------------------------------------------------------------------

def stap2_cluster(alle: list[dict]) -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 2 -- Raster-BFS clustering")
    print("=" * 64)

    if not alle:
        print("  Geen items.")
        return []

    t0 = time.time()
    posities = [_item_pos(item) for item in alle]
    bboxen   = [_item_bbox(item) for item in alle]
    n        = len(alle)

    # Raster-index: cel = CLUSTER_AFSTAND × CLUSTER_AFSTAND pt
    cel = CLUSTER_AFSTAND
    raster: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(posities):
        raster[(int(x // cel), int(y // cel))].append(i)

    verwerkt = [False] * n
    clusters_raw = []

    for start in range(n):
        if verwerkt[start]:
            continue
        cluster = [start]
        verwerkt[start] = True
        frontier = [start]

        while frontier:
            huidig = frontier.pop()
            hx, hy = posities[huidig]
            gx, gy = int(hx // cel), int(hy // cel)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in raster.get((gx + dx, gy + dy), []):
                        if verwerkt[j]:
                            continue
                        jx, jy = posities[j]
                        if math.hypot(hx - jx, hy - jy) <= CLUSTER_AFSTAND:
                            verwerkt[j] = True
                            cluster.append(j)
                            frontier.append(j)

        cb = _bbox_unie([bboxen[i] for i in cluster])
        clusters_raw.append({
            "items":   [alle[i] for i in cluster],
            "bbox":    cb,
            "breedte": cb[2] - cb[0],
            "hoogte":  cb[3] - cb[1],
            "n":       len(cluster),
        })

    elapsed_cluster = time.time() - t0
    print(f"  Ruwe clusters: {len(clusters_raw)}  ({elapsed_cluster:.1f}s)")

    # Filter: min items + min bbox-dimensies
    gefilterd = [
        c for c in clusters_raw
        if c["n"] >= MIN_ITEMS
        and c["breedte"] >= MIN_BBOX_DIM
        and c["hoogte"]  >= MIN_BBOX_DIM
    ]
    weg = len(clusters_raw) - len(gefilterd)

    # Sorteer op item-count (groot → klein)
    gefilterd.sort(key=lambda c: c["n"], reverse=True)

    print(f"  Na filter (>={MIN_ITEMS} items, >={MIN_BBOX_DIM}x{MIN_BBOX_DIM}pt): "
          f"{len(gefilterd)} clusters, {weg} weggelaten")

    for i, c in enumerate(gefilterd):
        tags = defaultdict(int)
        for item in c["items"]:
            tags[item.get("_tag", "?")] += 1
        tag_str = "  ".join(f"{v}x{k.replace('_','-')}" for k, v in sorted(tags.items()))
        print(f"  {i:3d}: ({c['bbox'][0]:.0f},{c['bbox'][1]:.0f})"
              f"–({c['bbox'][2]:.0f},{c['bbox'][3]:.0f})"
              f"  {c['breedte']:.0f}x{c['hoogte']:.0f}pt  n={c['n']}  {tag_str}")

    return gefilterd


# ---------------------------------------------------------------------------
# Stap 3 — Crop per cluster
# ---------------------------------------------------------------------------

def stap3_crop(clusters: list[dict]) -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 3 -- Crops renderen op 200 DPI")
    print("=" * 64)

    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)

    oud_doc   = fitz.open(str(OUD_PDF));  oud_page  = oud_doc[PAGINA]
    nieuw_doc = fitz.open(str(NIEUW_PDF)); nieuw_page = nieuw_doc[PAGINA]
    pr = nieuw_page.rect

    resultaten = []
    for i, c in enumerate(clusters):
        x0, y0, x1, y1 = c["bbox"]
        cx0 = max(pr.x0, x0 - CROP_PAD)
        cy0 = max(pr.y0, y0 - CROP_PAD)
        cx1 = min(pr.x1, x1 + CROP_PAD)
        cy1 = min(pr.y1, y1 + CROP_PAD)
        clip = fitz.Rect(cx0, cy0, cx1, cy1)

        oud_pix   = oud_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        nieuw_pix = nieuw_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)

        oud_pix.save(str(TMP_DIR / f"cluster_{i:03d}_oud.png"))
        nieuw_pix.save(str(TMP_DIR / f"cluster_{i:03d}_nieuw.png"))

        print(f"  {i:3d}: clip=({cx0:.0f},{cy0:.0f},{cx1:.0f},{cy1:.0f})  "
              f"{oud_pix.width}x{oud_pix.height}px")

        resultaten.append({
            **c,
            "clip":     (cx0, cy0, cx1, cy1),
            "oud_b64":  _pix_b64(oud_pix),
            "nieuw_b64": _pix_b64(nieuw_pix),
        })

    oud_doc.close(); nieuw_doc.close()
    print(f"\n  Alle {len(resultaten)} crops opgeslagen in {TMP_DIR}/")
    return resultaten


# ---------------------------------------------------------------------------
# Stap 4 — Vision per cluster (sequentieel)
# ---------------------------------------------------------------------------

_PROMPT = """\
Twee crops van dezelfde locatie op een Nederlandse afbouwtekening (oud + nieuw).
De wandtypes op deze tekening volgens het renvooi:
{wt}

Op deze locatie is een pixel-verschil gedetecteerd. Beoordeel wat er is veranderd:

JSON:
{{
  "is_wand_wijziging": bool,
  "type": "toegevoegd" | "verdwenen" | "gewijzigd" | "geen_wand",
  "wandtype": string of "onbekend",
  "wandtype_oud": string of null,
  "wandtype_nieuw": string of null,
  "redenering": "korte uitleg"
}}

Mogelijkheden naast wand-wijziging:
- maatlijn-wijziging
- tekst-wijziging
- hulplijn-wijziging
- architect-aantekening (rode markering)
- symbool-wijziging (deur, raam)

Als geen wand-wijziging: "is_wand_wijziging": false en "type": "geen_wand".\
"""


async def _call_vision(
    client: anthropic.AsyncAnthropic,
    c: dict,
    wt_tekst: str,
    nr: int,
) -> dict:
    prompt = _PROMPT.format(wt=wt_tekst)
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",  "text": "Oud:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                  "data": c["oud_b64"]}},
                    {"type": "text",  "text": "Nieuw:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                  "data": c["nieuw_b64"]}},
                    {"type": "text",  "text": prompt},
                ],
            }],
        )
        tekst  = "".join(b.text for b in resp.content if hasattr(b, "text"))
        data   = _parse_json(tekst)
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        return {"verdict": data, "tokens": tokens, "fout": None}
    except Exception as e:
        return {"verdict": {}, "tokens": 0, "fout": str(e)}


async def stap4_vision(clusters: list[dict], wandtypes: list[dict]) -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 4 -- Vision per cluster (sequentieel)")
    print("=" * 64)

    client   = anthropic.AsyncAnthropic()
    wt_tekst = _wandtypes_tekst(wandtypes)
    resultaten = []
    totaal_tokens = 0

    for i, c in enumerate(clusters):
        res = await _call_vision(client, c, wt_tekst, i)
        totaal_tokens += res["tokens"]

        if res["fout"]:
            print(f"  {i:3d}: FOUT -- {res['fout']}")
        else:
            v = res["verdict"]
            is_wand = v.get("is_wand_wijziging", False)
            label   = "WAND    " if is_wand else "geen_wand"
            type_   = v.get("type", "?")
            wtype   = v.get("wandtype", "?")
            reden   = v.get("redenering", "")[:70]
            print(f"  {i:3d}: [{label}] {type_:12s} | {wtype:<22} | {reden}")

        resultaten.append({**c, "vision": res})

    print(f"\n  Totaal tokens stap 4: {totaal_tokens}")
    return resultaten


# ---------------------------------------------------------------------------
# Stap 5 — Eindrapport
# ---------------------------------------------------------------------------

def stap5_rapport(resultaten: list[dict], wandtypes: list[dict], t_start: float):
    elapsed = time.time() - t_start

    print("\n" + "=" * 64)
    print("STAP 5 -- EINDRAPPORT")
    print("=" * 64)

    wand_wijz  = []
    geen_wand  = []
    fouten     = []

    for r in resultaten:
        if r["vision"]["fout"]:
            fouten.append(r)
        elif r["vision"]["verdict"].get("is_wand_wijziging", False):
            wand_wijz.append(r)
        else:
            geen_wand.append(r)

    # Detailtabel
    print(f"\n{'Nr':>4}  {'Bbox':^38}  {'n':>5}  {'Verdict':^9}  {'Type':^12}  Wandtype")
    print("  " + "-" * 90)
    for r in resultaten:
        v    = r["vision"]["verdict"]
        fout = r["vision"]["fout"]
        bb   = r["bbox"]
        bb_s = f"({bb[0]:.0f},{bb[1]:.0f})–({bb[2]:.0f},{bb[3]:.0f})"
        if fout:
            print(f"  {resultaten.index(r):3d}  {bb_s:^38}  {r['n']:5d}  FOUT")
        else:
            is_w  = v.get("is_wand_wijziging", False)
            label = "WAND" if is_w else "geen_wand"
            type_ = v.get("type", "?")
            wtype = v.get("wandtype", "?")
            print(f"  {resultaten.index(r):3d}  {bb_s:^38}  {r['n']:5d}  {label:^9}  {type_:^12}  {wtype}")

    # Samenvatting
    print(f"\nSamenvatting:")
    print(f"  Totaal clusters onderzocht : {len(resultaten)}")
    print(f"  Echte wand-wijzigingen     : {len(wand_wijz)}")
    print(f"  Geen wand (false positives): {len(geen_wand)}")
    print(f"  Fouten                     : {len(fouten)}")

    # Per wandtype
    if wand_wijz:
        teller: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in wand_wijz:
            v = r["vision"]["verdict"]
            t  = v.get("type", "?")
            wt = v.get("wandtype", "onbekend")
            teller[wt][t] += 1
        print(f"\nPer wandtype:")
        for wt, types in sorted(teller.items()):
            details = ", ".join(f"{cnt}x {t}" for t, cnt in sorted(types.items()))
            print(f"  {wt}: {details}")

    # Kosten
    totaal_tokens = sum(r["vision"]["tokens"] for r in resultaten)
    kosten = totaal_tokens / 1_000_000 * 6.0  # ~$6 gemiddeld input+output mix
    print(f"\nKosten + tijd:")
    print(f"  Tokens totaal : {totaal_tokens}")
    print(f"  Kosten (schat): ~${kosten:.3f}")
    print(f"  Totale tijd   : {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    t0 = time.time()
    print("=" * 64)
    print("HYBRIDE PIPELINE TEST: 56 de Helling")
    print("=" * 64)

    wandtypes = _laad_wandtypes()
    if not wandtypes:
        print("FOUT: geen wandtypes in references/helling/. Draai eerst test_helling_endtoend.py.")
        return
    print(f"Wandtypes geladen: {len(wandtypes)}")

    alle   = stap1_pixel_diff()
    clust  = stap2_cluster(alle)
    if not clust:
        print("Geen clusters na filter. Stop.")
        return
    clust  = stap3_crop(clust)
    result = await stap4_vision(clust, wandtypes)
    stap5_rapport(result, wandtypes, t0)


if __name__ == "__main__":
    asyncio.run(main())

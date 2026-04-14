"""
Diff engine voor K&K demarcatietekeningen.

Vergelijkt twee PDF's op tekst, lijnen, vullingen en kleuren via pymupdf.
Retourneert een dict met alle wijzigingen — geen print statements.
"""

import logging
import os
import re
import tempfile

import fitz
from collections import defaultdict

from .config import DiffConfig
from .layout_detect import detect_layout, PageLayout
from .wall_detect import detecteer_wand_clusters, detecteer_verdwenen_wanden

logger = logging.getLogger(__name__)

# Bekende wandtypes die ALTIJD herkend moeten worden (uit CLAUDE.md)
BEKENDE_WANDTYPEN = [
    "kalkzandsteen 120mm",
    "kalkzandsteen 100mm",
    "gibo 100mm",
    "gibo zwaar 70mm",
    "gibo 70mm",
    "hsb-wand",
    "sandwichpaneel",
    "voorzetwand: isolatie + biobased plaat + gips",
    "isolatie+stuc",
    "pir+osb",
    "hardschuimisolatie",
    "prefabbeton",
    "beton",
    "rhombus gevelafwerking",
    "achterwand toilet",
]


# ---------------------------------------------------------------------------
# Annotation stripping
# ---------------------------------------------------------------------------

def strip_annotations(pdf_path: str) -> str:
    """
    Verwijder alle PDF annotations (comments, markups, stamps,
    text boxes, circles, arrows, highlights etc.) van alle pagina's.
    Retourneert pad naar schone PDF (temp file).
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        annots = list(page.annots()) if page.annots() else []
        for annot in annots:
            page.delete_annot(annot)
    clean_path = tempfile.mktemp(suffix=".pdf")
    doc.save(clean_path)
    doc.close()
    return clean_path


# ---------------------------------------------------------------------------
# Kleur helpers
# ---------------------------------------------------------------------------

def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _kleur_naam(r: float, g: float, b: float) -> str:
    if r < 0.05 and g < 0.05 and b < 0.05:
        return "zwart"
    if r > 0.95 and g > 0.95 and b > 0.95:
        return "wit"
    if abs(r - g) < 0.08 and abs(g - b) < 0.08 and 0.05 <= r <= 0.95:
        return f"grijs({int(r * 100)}%)"
    if r > 0.7 and g < 0.3 and b < 0.3:
        return "rood"
    if r < 0.3 and g > 0.5 and b < 0.3:
        return "groen"
    if r < 0.3 and g < 0.3 and b > 0.7:
        return "blauw"
    if r > 0.8 and g > 0.8 and b < 0.3:
        return "geel"
    if r > 0.8 and g > 0.4 and g < 0.8 and b < 0.2:
        return "oranje"
    if r > 0.4 and g < 0.3 and b > 0.4:
        return "paars"
    if r < 0.3 and g > 0.5 and b > 0.5:
        return "cyaan"
    return f"rgb({r:.2f},{g:.2f},{b:.2f})"


def _color_tuple_to_rgb(kleur) -> tuple[float, float, float] | None:
    if kleur is None:
        return None
    if isinstance(kleur, (int, float)):
        v = float(kleur)
        return (v, v, v)
    if len(kleur) == 1:
        v = float(kleur[0])
        return (v, v, v)
    if len(kleur) == 3:
        return (float(kleur[0]), float(kleur[1]), float(kleur[2]))
    if len(kleur) == 4:
        c, m, y, k = (float(x) for x in kleur)
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


def _span_color_to_rgb(color_int: int) -> tuple[float, float, float]:
    r = ((color_int >> 16) & 0xFF) / 255.0
    g = ((color_int >> 8) & 0xFF) / 255.0
    b = (color_int & 0xFF) / 255.0
    return (r, g, b)


def _is_zwart(r: float, g: float, b: float) -> bool:
    return r < 0.05 and g < 0.05 and b < 0.05


def _hex_naam(r: float, g: float, b: float) -> tuple[str, str]:
    return _rgb_to_hex(r, g, b), _kleur_naam(r, g, b)


# ---------------------------------------------------------------------------
# Extractie
# ---------------------------------------------------------------------------

def _splits_span_op_gaten(chars: list, rgb: tuple, span_tekst: str = "") -> list[dict]:
    """Splits een span in losse stukken op basis van ongebruikelijke gaten tussen tekens.

    In bouwtekeningen worden maatgetallen soms zonder spatie aaneengeregen
    in één span (bijv. "125345" voor "125" en "345"). Door de gap tussen
    opeenvolgende tekens te meten kunnen we ze splitsen.

    Drempel: gap > 0.8pt (karakter-breedte is typisch 3-5pt).
    """
    if not chars:
        return []

    GAP_DREMPEL = 0.8  # pt

    stukken = []
    huidig = [chars[0]]

    for prev, curr in zip(chars, chars[1:]):
        gap = curr["bbox"][0] - prev["bbox"][2]
        if gap > GAP_DREMPEL:
            stukken.append(huidig)
            huidig = [curr]
        else:
            huidig.append(curr)
    stukken.append(huidig)

    items = []
    for stuk in stukken:
        tekst = "".join(c["c"] for c in stuk).strip()
        if not tekst:
            continue
        x0 = stuk[0]["bbox"][0]
        y0 = min(c["bbox"][1] for c in stuk)
        x1 = stuk[-1]["bbox"][2]
        y1 = max(c["bbox"][3] for c in stuk)
        items.append({
            "tekst": tekst,
            "rgb": rgb,
            "pos": (round(x0, 1), round(y0, 1)),
            "bbox": (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
            "span_tekst": span_tekst,
        })
    return items


def _extract_tekst(page) -> list[dict]:
    items = []
    try:
        blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except Exception:
        return items
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                tekst = "".join(c["c"] for c in chars).strip()
                if not tekst:
                    continue
                r, g, b = _span_color_to_rgb(span.get("color", 0))
                rgb = (round(r, 3), round(g, 3), round(b, 3))

                # Splits op gaten tussen tekens (bijv. "125345" → "125" + "345")
                # span_tekst meegeven zodat context bewaard blijft (bijv. "Opp.: 2,70 m2")
                gesplitst = _splits_span_op_gaten(chars, rgb, span_tekst=tekst)
                if gesplitst:
                    items.extend(gesplitst)
                else:
                    # Fallback: gebruik bbox van de hele span
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    items.append({
                        "tekst": tekst,
                        "rgb": rgb,
                        "pos": (round(bbox[0], 1), round(bbox[1], 1)),
                        "bbox": tuple(round(v, 1) for v in bbox),
                    })
    return items


def _extract_lijnen(page) -> list[dict]:
    items = []
    try:
        paths = page.get_drawings()
    except Exception:
        return items
    for path in paths:
        stroke = _color_tuple_to_rgb(path.get("color"))
        fill = _color_tuple_to_rgb(path.get("fill"))
        width = round(path.get("width") or 0, 3)
        rect = path.get("rect", fitz.Rect())

        for item in path.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                items.append({
                    "van": (round(p1.x, 1), round(p1.y, 1)),
                    "naar": (round(p2.x, 1), round(p2.y, 1)),
                    "width": width,
                    "stroke": stroke,
                })

        if fill is not None and rect.width > 0 and rect.height > 0:
            items.append({
                "type": "fill",
                "rgb": (round(fill[0], 3), round(fill[1], 3), round(fill[2], 3)),
                "pos": (round(rect.x0, 1), round(rect.y0, 1)),
                "bbox": (round(rect.x0, 1), round(rect.y0, 1),
                         round(rect.x1, 1), round(rect.y1, 1)),
                "oppervlakte": round(rect.width * rect.height, 1),
            })

        if stroke is not None:
            items.append({
                "type": "stroke",
                "rgb": (round(stroke[0], 3), round(stroke[1], 3), round(stroke[2], 3)),
                "pos": (round(rect.x0, 1), round(rect.y0, 1)),
                "width": width,
            })

    return items


# ---------------------------------------------------------------------------
# Spatial grid helpers
# ---------------------------------------------------------------------------

def _grid_key(x: float, y: float, cell: float = 10.0) -> tuple[int, int]:
    return (int(x // cell), int(y // cell))


def _grid_neighbors(gx: int, gy: int):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (gx + dx, gy + dy)


def _afstand(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Tekst vergelijking
# ---------------------------------------------------------------------------

def _vergelijk_tekst(oud_items: list, nieuw_items: list, drempel: float = 15.0):
    gewijzigd = []
    kleur_gewijzigd = []
    toegevoegd = []
    verdwenen = []

    grid: dict[tuple, list] = defaultdict(list)
    for idx, nieuw in enumerate(nieuw_items):
        gk = _grid_key(nieuw["pos"][0], nieuw["pos"][1], drempel * 2)
        grid[gk].append(idx)

    matched_nieuw = set()

    for oud in oud_items:
        beste = None
        beste_d = float("inf")
        beste_idx = -1

        gk = _grid_key(oud["pos"][0], oud["pos"][1], drempel * 2)
        for cell in _grid_neighbors(*gk):
            for idx in grid.get(cell, ()):
                if idx in matched_nieuw:
                    continue
                nieuw = nieuw_items[idx]
                d = _afstand(oud["pos"], nieuw["pos"])
                if d < beste_d and d < drempel:
                    beste = nieuw
                    beste_d = d
                    beste_idx = idx

        if beste is not None:
            matched_nieuw.add(beste_idx)
            if oud["tekst"] != beste["tekst"]:
                gewijzigd.append({"oud": oud, "nieuw": beste})
            elif oud["rgb"] != beste["rgb"]:
                kleur_gewijzigd.append({
                    "tekst": oud["tekst"], "pos": oud["pos"],
                    "oud_rgb": oud["rgb"], "nieuw_rgb": beste["rgb"],
                })
        else:
            verdwenen.append(oud)

    for idx, nieuw in enumerate(nieuw_items):
        if idx not in matched_nieuw:
            toegevoegd.append(nieuw)

    return gewijzigd, toegevoegd, verdwenen, kleur_gewijzigd


# ---------------------------------------------------------------------------
# Lijn vergelijking
# ---------------------------------------------------------------------------

def _vergelijk_lijnen(oud_items: list, nieuw_items: list, drempel: float = 5.0):
    oud_lijnen = [i for i in oud_items if "van" in i]
    nieuw_lijnen = [i for i in nieuw_items if "van" in i]

    nieuw_exact: dict[tuple, list] = defaultdict(list)
    for idx, lijn in enumerate(nieuw_lijnen):
        nieuw_exact[(lijn["van"], lijn["naar"])].append(idx)

    grid: dict[tuple, list] = defaultdict(list)
    for idx, lijn in enumerate(nieuw_lijnen):
        gk = _grid_key(lijn["van"][0], lijn["van"][1], drempel * 2)
        grid[gk].append(idx)

    width_gewijzigd = []
    kleur_gewijzigd = []
    verdwenen = []
    matched_nieuw = set()

    for oud in oud_lijnen:
        key = (oud["van"], oud["naar"])
        match_idx = None

        if key in nieuw_exact:
            for idx in nieuw_exact[key]:
                if idx not in matched_nieuw:
                    match_idx = idx
                    break

        if match_idx is None:
            beste_d = float("inf")
            gk = _grid_key(oud["van"][0], oud["van"][1], drempel * 2)
            for cell in _grid_neighbors(*gk):
                for idx in grid.get(cell, ()):
                    if idx in matched_nieuw:
                        continue
                    nieuw = nieuw_lijnen[idx]
                    d_van = _afstand(oud["van"], nieuw["van"])
                    if d_van >= drempel:
                        continue
                    d_naar = _afstand(oud["naar"], nieuw["naar"])
                    if d_naar >= drempel:
                        continue
                    d = d_van + d_naar
                    if d < beste_d:
                        beste_d = d
                        match_idx = idx

        if match_idx is not None:
            matched_nieuw.add(match_idx)
            nieuw = nieuw_lijnen[match_idx]
            if abs(oud["width"] - nieuw["width"]) >= 0.5:  # TODO: use config in Phase 2
                width_gewijzigd.append({
                    "van": oud["van"], "naar": oud["naar"],
                    "oud_width": oud["width"], "nieuw_width": nieuw["width"],
                })
            oud_s = oud.get("stroke")
            nieuw_s = nieuw.get("stroke")
            if oud_s and nieuw_s and oud_s != nieuw_s:
                if not (_is_zwart(*oud_s) and _is_zwart(*nieuw_s)):
                    kleur_gewijzigd.append({
                        "van": oud["van"], "naar": oud["naar"],
                        "oud_rgb": oud_s, "nieuw_rgb": nieuw_s,
                    })
        else:
            verdwenen.append(oud)

    toegevoegd = [nieuw_lijnen[i] for i in range(len(nieuw_lijnen)) if i not in matched_nieuw]

    return width_gewijzigd, kleur_gewijzigd, toegevoegd, verdwenen


# ---------------------------------------------------------------------------
# Vulkleur vergelijking
# ---------------------------------------------------------------------------

def _vergelijk_fills(oud_items: list, nieuw_items: list, drempel: float = 10.0):
    oud_fills = [i for i in oud_items if i.get("type") == "fill"]
    nieuw_fills = [i for i in nieuw_items if i.get("type") == "fill"]

    grid: dict[tuple, list] = defaultdict(list)
    for idx, f in enumerate(nieuw_fills):
        gk = _grid_key(f["pos"][0], f["pos"][1], drempel * 2)
        grid[gk].append(idx)

    gewijzigd = []
    verdwenen = []
    matched_nieuw = set()

    for oud in oud_fills:
        beste_idx = None
        beste_d = float("inf")

        gk = _grid_key(oud["pos"][0], oud["pos"][1], drempel * 2)
        for cell in _grid_neighbors(*gk):
            for idx in grid.get(cell, ()):
                if idx in matched_nieuw:
                    continue
                nieuw = nieuw_fills[idx]
                if oud["oppervlakte"] > 0:
                    ratio = nieuw["oppervlakte"] / max(oud["oppervlakte"], 0.1)
                    if ratio < 0.5 or ratio > 2.0:
                        continue
                d = _afstand(oud["pos"], nieuw["pos"])
                if d < beste_d and d < drempel:
                    beste_d = d
                    beste_idx = idx

        if beste_idx is not None:
            matched_nieuw.add(beste_idx)
            nieuw = nieuw_fills[beste_idx]
            if oud["rgb"] != nieuw["rgb"]:
                gewijzigd.append({
                    "pos": oud["pos"],
                    "bbox": oud["bbox"],
                    "oppervlakte": oud["oppervlakte"],
                    "oud_rgb": oud["rgb"],
                    "nieuw_rgb": nieuw["rgb"],
                })
        else:
            verdwenen.append(oud)

    toegevoegd = [nieuw_fills[i] for i in range(len(nieuw_fills)) if i not in matched_nieuw]

    return gewijzigd, toegevoegd, verdwenen


# ---------------------------------------------------------------------------
# Kleurinventaris
# ---------------------------------------------------------------------------

def _kleur_inventaris_split(tekst_items: list, lijn_items: list) -> dict:
    tekst_teller: dict[str, int] = defaultdict(int)
    lijn_teller: dict[str, int] = defaultdict(int)
    vul_teller: dict[str, int] = defaultdict(int)

    for item in tekst_items:
        h, n = _hex_naam(*item["rgb"])
        tekst_teller[f"{h} ({n})"] += 1

    for item in lijn_items:
        if item.get("type") == "fill":
            h, n = _hex_naam(*item["rgb"])
            vul_teller[f"{h} ({n})"] += 1
        elif item.get("type") == "stroke":
            h, n = _hex_naam(*item["rgb"])
            lijn_teller[f"{h} ({n})"] += 1
        elif "van" in item and item.get("stroke"):
            h, n = _hex_naam(*item["stroke"])
            lijn_teller[f"{h} ({n})"] += 1

    return {
        "tekst": dict(sorted(tekst_teller.items(), key=lambda x: -x[1])),
        "lijnen": dict(sorted(lijn_teller.items(), key=lambda x: -x[1])),
        "vullingen": dict(sorted(vul_teller.items(), key=lambda x: -x[1])),
    }


# ---------------------------------------------------------------------------
# Auto-categorisatie
# ---------------------------------------------------------------------------

_RE_NUMERIEK = re.compile(r"^[\d.,]+$")
_RE_LETTER = re.compile(r"^[A-Z]$")


_RE_OPP_CONTEXT = re.compile(r"m2|m²|m\u00b2|opp\.?", re.IGNORECASE)


def _categoriseer_tekst_wijziging(
    oud_tekst: str, nieuw_tekst: str,
    span_tekst_oud: str = "", span_tekst_nieuw: str = "",
) -> str:
    # Oppervlakte eerst: ook als het getal gesplitst is van z'n eenheid
    # check de volledige span-context (bijv. "Opp.: 2,70 m2")
    for ctx in (span_tekst_oud, span_tekst_nieuw, oud_tekst, nieuw_tekst):
        if _RE_OPP_CONTEXT.search(ctx):
            return "oppervlakte"
    if _RE_NUMERIEK.match(oud_tekst) and _RE_NUMERIEK.match(nieuw_tekst):
        return "maat"
    if _RE_LETTER.match(oud_tekst) or _RE_LETTER.match(nieuw_tekst):
        return "revisieletter"
    combined = (oud_tekst + nieuw_tekst).lower()
    if "at." in combined or "type" in combined:
        return "ruimtelabel"
    if any(kw in combined for kw in ("koof", "plafond", "wand", "gibo", "isolatie")):
        return "bouwkundig"
    return "overig"


# ---------------------------------------------------------------------------
# Lijn sample helpers
# ---------------------------------------------------------------------------

def _lijn_sample(lijnen: list, max_items: int = 50) -> list[dict]:
    """Geef max 50 niet-zwarte lijnen als sample."""
    niet_zwart = []
    for l in lijnen:
        s = l.get("stroke")
        if s and not _is_zwart(*s):
            h, n = _hex_naam(*s)
            niet_zwart.append({
                "van": list(l["van"]),
                "naar": list(l["naar"]),
                "linewidth": l["width"],
                "kleur": h,
            })
            if len(niet_zwart) >= max_items:
                break
    return niet_zwart


# ---------------------------------------------------------------------------
# Legenda-parser
# ---------------------------------------------------------------------------

def _extract_legenda(page, legenda_x_ratio: float = 0.88) -> dict:
    """
    Lees de kleur-naar-wandtype mapping uit de legenda op de tekening.

    De legenda staat rechts op de tekening (x > 88% breedte) en bevat
    gekleurde blokjes naast tekst-labels zoals "Gibo 70mm", "kalkzandsteen 120mm".

    Returns: dict met afgeronde RGB tuples als key en wandtype als value.
             Bijv. {(0.16, 0.49, 0.35): "Gibo 70mm", ...}
    """
    pw = page.rect.width
    ph = page.rect.height
    legenda_x = pw * legenda_x_ratio
    max_y = ph * 0.6  # Dynamisch: legenda staat typisch in bovenste 60%

    # Verzamel tekst in legenda-zone
    tekst_items = _extract_tekst(page)
    legenda_tekst = [
        t for t in tekst_items
        if t["pos"][0] > legenda_x and t["pos"][1] < max_y
    ]

    # Verzamel gekleurde vlakken in legenda-zone (de gekleurde blokjes)
    lijn_items = _extract_lijnen(page)
    legenda_fills = [
        f for f in lijn_items
        if f.get("type") == "fill"
        and f["pos"][0] > legenda_x - 80  # blokjes staan net links van tekst
        and f["pos"][1] < max_y
        and 5 < f["oppervlakte"] < 900  # Min/max grootte voor legenda-kleurvlakken
    ]

    # Sorteer beide op y-positie
    legenda_tekst.sort(key=lambda t: t["pos"][1])
    legenda_fills.sort(key=lambda f: f["pos"][1])

    # Koppel: voor elk tekst-label, vind het gekleurde vlak dat er vlak boven/naast staat
    mapping = {}
    for tekst_item in legenda_tekst:
        label = tekst_item["tekst"].strip()
        tx = tekst_item["pos"][0]
        ty = tekst_item["pos"][1]

        # Skip niet-wandtype labels
        if len(label) < 3:
            continue
        # Skip revisie/datum teksten
        if any(kw in label.lower() for kw in ("uitgave", "datum", "getekend", "controle", "ndo", "definitief")):
            continue

        # Zoek dichtstbijzijnde fill op y-as (blokje staat op dezelfde hoogte)
        beste_fill = None
        beste_score = float("inf")
        for fill in legenda_fills:
            fx = fill["pos"][0]
            fy = fill["pos"][1]
            dy = abs(fy - ty)
            dx = tx - fx  # Fill moet links van tekst staan
            # Fill moet in de buurt zijn (binnen 20pt y, 80pt x) en links van tekst
            if dy < 20 and 0 < dx < 80:
                score = dy + dx * 0.1  # Prioriteer y-nabijheid
                if score < beste_score:
                    beste_score = score
                    beste_fill = fill

        if beste_fill is not None:
            rgb = beste_fill["rgb"]
            # Skip wit en zwart
            if rgb[0] > 0.95 and rgb[1] > 0.95 and rgb[2] > 0.95:
                continue
            if rgb[0] < 0.05 and rgb[1] < 0.05 and rgb[2] < 0.05:
                continue
            # Rond af op 2 decimalen voor matching
            key = (round(rgb[0], 2), round(rgb[1], 2), round(rgb[2], 2))
            mapping[key] = label

    return mapping


def _valideer_legenda(legenda: dict) -> None:
    """Log warnings voor onbekende wandtypes in de legenda-mapping."""
    for rgb, label in legenda.items():
        label_lower = label.strip().lower()
        # Normaliseer: verwijder spaties voor vergelijking
        label_norm = re.sub(r"\s+", "", label_lower)
        is_bekend = any(
            re.sub(r"\s+", "", bekende) in label_norm
            or label_norm in re.sub(r"\s+", "", bekende)
            for bekende in BEKENDE_WANDTYPEN
        )
        if not is_bekend:
            logger.warning(
                "WAARSCHUWING: Kleur (%.2f, %.2f, %.2f) in renvooi niet gekoppeld "
                "aan bekend wandtype. Tekst naast blokje: '%s'",
                rgb[0], rgb[1], rgb[2], label,
            )


def _lookup_wandtype(rgb: tuple, legenda: dict, max_afstand: float = 0.15) -> str | None:
    """Zoek wandtype via best-match: kleinste Euclidische kleurafstand wint."""
    if not legenda or not rgb:
        return None
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    # Wit en zwart zijn nooit wandtypes
    if min(r, g, b) > 0.95:
        return None
    if max(r, g, b) < 0.05:
        return None
    beste_label = None
    beste_d = max_afstand
    for (lr, lg, lb), label in legenda.items():
        d = ((r - lr) ** 2 + (g - lg) ** 2 + (b - lb) ** 2) ** 0.5
        if d < beste_d:
            beste_d = d
            beste_label = label
    return beste_label


def _lookup_wandtype_bij_bbox(
    bbox: tuple, alle_items: list, legenda: dict, margin: float = 5.0,
) -> str | None:
    """Zoek wandtype door fills te matchen die overlappen met een wand-bbox.

    Kiest de fill met de kleinste kleurafstand tot een legenda-entry,
    en negeert achtergrond-fills (wit/zwart) en fills die veel groter
    zijn dan de wand zelf.
    """
    if not legenda:
        return None
    bx0, by0, bx1, by1 = bbox
    wand_opp = (bx1 - bx0) * (by1 - by0)
    max_fill_opp = max(wand_opp * 20, 5000)  # Achtergrond-fills uitsluiten

    beste_label = None
    beste_d = 0.15
    for item in alle_items:
        if item.get("type") != "fill":
            continue
        fb = item.get("bbox")
        if fb is None:
            continue
        opp = item.get("oppervlakte", 0)
        if opp > max_fill_opp:
            continue
        if (fb[0] < bx1 + margin and fb[2] > bx0 - margin and
                fb[1] < by1 + margin and fb[3] > by0 - margin):
            rgb = item["rgb"]
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            for (lr, lg, lb), label in legenda.items():
                d = ((r - lr) ** 2 + (g - lg) ** 2 + (b - lb) ** 2) ** 0.5
                if d < beste_d:
                    beste_d = d
                    beste_label = label
    return beste_label


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def run_diff(
    oud_pdf_path: str,
    nieuw_pdf_path: str,
    pagina: int = 0,
    config: DiffConfig | None = None,
) -> dict:
    """
    Vergelijk twee PDF's en retourneer een dict met alle wijzigingen.

    Parameters:
        oud_pdf_path: pad naar de oude PDF
        nieuw_pdf_path: pad naar de nieuwe PDF
        pagina: 0-based pagina index
        config: optionele configuratie (defaults = huidige waarden)

    Returns:
        dict met alle diff-secties
    """
    if config is None:
        config = DiffConfig()

    # Strip annotations eerst
    oud_clean = strip_annotations(oud_pdf_path)
    nieuw_clean = strip_annotations(nieuw_pdf_path)

    oud_doc = fitz.open(oud_clean)
    nieuw_doc = fitz.open(nieuw_clean)

    try:
        if pagina >= len(oud_doc) or pagina >= len(nieuw_doc):
            return {
                "error": f"Pagina {pagina + 1} bestaat niet. "
                         f"OUD heeft {len(oud_doc)} pagina's, "
                         f"NIEUW heeft {len(nieuw_doc)} pagina's."
            }

        oud_page = oud_doc[pagina]
        nieuw_page = nieuw_doc[pagina]

        # Layout auto-detectie op nieuwe tekening
        layout = detect_layout(nieuw_page, config)

        # Extractie
        oud_tekst = _extract_tekst(oud_page)
        nieuw_tekst = _extract_tekst(nieuw_page)
        oud_lijnen = _extract_lijnen(oud_page)
        nieuw_lijnen = _extract_lijnen(nieuw_page)

        # Legenda: gebruik auto-detectie, fallback naar ratio-gebaseerd
        if layout.legenda_mapping:
            legenda = layout.legenda_mapping
        else:
            legenda = _extract_legenda(nieuw_page)

        # Valideer legenda tegen bekende wandtypes
        _valideer_legenda(legenda)

        # Vergelijkingen (drempels uit config)
        tekst_gewijzigd, tekst_toegevoegd, tekst_verdwenen, tekst_kleur = \
            _vergelijk_tekst(oud_tekst, nieuw_tekst,
                             drempel=config.tekst_match_drempel)

        lijn_width, lijn_kleur, lijnen_toegevoegd, lijnen_verdwenen = \
            _vergelijk_lijnen(oud_lijnen, nieuw_lijnen,
                              drempel=config.lijn_match_drempel)

        fill_gewijzigd, fills_toegevoegd, fills_verdwenen = \
            _vergelijk_fills(oud_lijnen, nieuw_lijnen,
                             drempel=config.fill_match_drempel)

        fills_toegevoegd_groot = [
            f for f in fills_toegevoegd
            if f["oppervlakte"] > config.min_fill_oppervlakte
        ]
        fills_verdwenen_groot = [
            f for f in fills_verdwenen
            if f["oppervlakte"] > config.min_fill_oppervlakte
        ]

        # Wanddetectie op toegevoegde/verwijderde lijnen (config drempels)
        # Tekst-items meegeven voor cross-referencing (korte wanden bevestigen)
        nieuwe_wanden = detecteer_wand_clusters(
            lijnen_toegevoegd,
            min_afstand=config.min_wand_afstand_pt,
            max_afstand=config.max_wand_afstand_pt,
            min_lengte=config.min_wand_lengte_pt,
            max_resultaten=config.max_wand_resultaten,
            tekst_items=nieuw_tekst,
        )
        verdwenen_wanden = detecteer_verdwenen_wanden(
            lijnen_verdwenen,
            min_afstand=config.min_wand_afstand_pt,
            max_afstand=config.max_wand_afstand_pt,
            min_lengte=config.min_wand_lengte_pt,
            max_resultaten=config.max_wand_resultaten,
            tekst_items=oud_tekst,
        )

        # Wandtype bepalen via fills + legenda
        for w in nieuwe_wanden:
            w["wandtype"] = _lookup_wandtype_bij_bbox(
                w["bbox"], nieuw_lijnen, legenda)
        for w in verdwenen_wanden:
            w["wandtype"] = _lookup_wandtype_bij_bbox(
                w["bbox"], oud_lijnen, legenda)

        # Sorteer helpers
        def _sort_pos(items, key="pos"):
            return sorted(items, key=lambda x: (x.get(key, (0, 0))[1], x.get(key, (0, 0))[0]))

        def _sort_van(items):
            return sorted(items, key=lambda x: (x["van"][1], x["van"][0]))

        # --- Build output dict ---

        # 1. Tekst gewijzigd
        out_tekst_gewijzigd = []
        for w in sorted(tekst_gewijzigd, key=lambda x: (x["oud"]["pos"][1], x["oud"]["pos"][0])):
            o, n = w["oud"], w["nieuw"]
            oh, on_h = _rgb_to_hex(*o["rgb"]), _rgb_to_hex(*n["rgb"])
            out_tekst_gewijzigd.append({
                "oud_tekst": o["tekst"],
                "nieuw_tekst": n["tekst"],
                "oud_pos": list(o["pos"]),
                "nieuw_pos": list(n["pos"]),
                "oud_bbox": list(o["bbox"]),
                "nieuw_bbox": list(n["bbox"]),
                "oud_kleur": oh,
                "nieuw_kleur": on_h,
                "categorie": _categoriseer_tekst_wijziging(
                    o["tekst"], n["tekst"],
                    span_tekst_oud=o.get("span_tekst", ""),
                    span_tekst_nieuw=n.get("span_tekst", ""),
                ),
            })

        # 2. Tekst toegevoegd
        out_tekst_toegevoegd = []
        for t in _sort_pos(tekst_toegevoegd):
            h, n = _hex_naam(*t["rgb"])
            out_tekst_toegevoegd.append({
                "tekst": t["tekst"],
                "pos": list(t["pos"]),
                "bbox": list(t["bbox"]),
                "kleur_hex": h,
                "kleur_naam": n,
            })

        # 3. Tekst verdwenen
        out_tekst_verdwenen = []
        for t in _sort_pos(tekst_verdwenen):
            h, n = _hex_naam(*t["rgb"])
            out_tekst_verdwenen.append({
                "tekst": t["tekst"],
                "pos": list(t["pos"]),
                "bbox": list(t["bbox"]),
                "kleur_hex": h,
                "kleur_naam": n,
            })

        # 4. Tekstkleur gewijzigd
        out_tekst_kleur = []
        for w in sorted(tekst_kleur, key=lambda x: (x["pos"][1], x["pos"][0])):
            oh, on_ = _hex_naam(*w["oud_rgb"])
            nh, nn_ = _hex_naam(*w["nieuw_rgb"])
            # bbox ophalen uit matched items
            oud_match = next((i for i in oud_tekst if i["pos"] == tuple(w["pos"])), None)
            bbox = list(oud_match["bbox"]) if oud_match else [w["pos"][0], w["pos"][1], w["pos"][0] + 30, w["pos"][1] + 10]
            out_tekst_kleur.append({
                "tekst": w["tekst"],
                "pos": list(w["pos"]),
                "bbox": bbox,
                "oud_kleur": oh,
                "oud_naam": on_,
                "nieuw_kleur": nh,
                "nieuw_naam": nn_,
            })

        # 6. Lijnkleur gewijzigd
        out_lijn_kleur = []
        for w in _sort_van(lijn_kleur):
            oh, on_ = _hex_naam(*w["oud_rgb"])
            nh, nn_ = _hex_naam(*w["nieuw_rgb"])
            out_lijn_kleur.append({
                "pos": list(w["van"]),
                "oud_kleur": oh,
                "oud_naam": on_,
                "nieuw_kleur": nh,
                "nieuw_naam": nn_,
            })

        # 7. Vulkleur gewijzigd (met wandtype uit legenda)
        out_fill_gewijzigd = []
        for w in _sort_pos(fill_gewijzigd):
            oh, on_ = _hex_naam(*w["oud_rgb"])
            nh, nn_ = _hex_naam(*w["nieuw_rgb"])
            oud_wandtype = _lookup_wandtype(w["oud_rgb"], legenda)
            nieuw_wandtype = _lookup_wandtype(w["nieuw_rgb"], legenda)
            out_fill_gewijzigd.append({
                "pos": list(w["pos"]),
                "bbox": list(w["bbox"]),
                "oppervlakte": w["oppervlakte"],
                "oud_kleur": oh,
                "oud_naam": on_,
                "nieuw_kleur": nh,
                "nieuw_naam": nn_,
                "oud_wandtype": oud_wandtype,
                "nieuw_wandtype": nieuw_wandtype,
            })

        # 8. Nieuwe gekleurde vlakken
        out_fills_nieuw = []
        for f in _sort_pos(fills_toegevoegd_groot):
            h, n = _hex_naam(*f["rgb"])
            out_fills_nieuw.append({
                "pos": list(f["pos"]),
                "kleur_hex": h,
                "kleur_naam": n,
                "oppervlakte": f["oppervlakte"],
            })

        # 9. Verdwenen gekleurde vlakken
        out_fills_weg = []
        for f in _sort_pos(fills_verdwenen_groot):
            h, n = _hex_naam(*f["rgb"])
            out_fills_weg.append({
                "pos": list(f["pos"]),
                "kleur_hex": h,
                "kleur_naam": n,
                "oppervlakte": f["oppervlakte"],
            })

        # Kleurinventaris
        inv_oud = _kleur_inventaris_split(oud_tekst, oud_lijnen)
        inv_nieuw = _kleur_inventaris_split(nieuw_tekst, nieuw_lijnen)

        # Tellingen
        oud_lijn_count = sum(1 for i in oud_lijnen if "van" in i)
        nieuw_lijn_count = sum(1 for i in nieuw_lijnen if "van" in i)

        # Layout serialiseren voor downstream modules
        layout_data = {
            "titelblok": list(layout.titelblok) if layout.titelblok else None,
            "legenda": list(layout.legenda) if layout.legenda else None,
            "koptekst": list(layout.koptekst) if layout.koptekst else None,
            "drawing_area": list(layout.drawing_area) if layout.drawing_area else None,
            "scale": layout.scale,
        }

        return {
            "meta": {
                "oud_bestand": oud_pdf_path,
                "nieuw_bestand": nieuw_pdf_path,
                "pagina": pagina + 1,
                "pagina_breedte": round(nieuw_page.rect.width, 1),
                "pagina_hoogte": round(nieuw_page.rect.height, 1),
                "oud_woorden": len(oud_tekst),
                "nieuw_woorden": len(nieuw_tekst),
                "oud_lijnen": oud_lijn_count,
                "nieuw_lijnen": nieuw_lijn_count,
            },
            "layout": layout_data,
            "_layout_obj": layout,  # intern gebruik door overlay/interpreter
            "nieuw_tekst_items": [
                {"tekst": t["tekst"], "pos": list(t["pos"])}
                for t in nieuw_tekst
            ],
            "legenda_mapping": {
                f"{r:.2f},{g:.2f},{b:.2f}": label
                for (r, g, b), label in legenda.items()
            },
            "tekst_gewijzigd": out_tekst_gewijzigd,
            "tekst_toegevoegd": out_tekst_toegevoegd,
            "tekst_verdwenen": out_tekst_verdwenen,
            "tekst_kleur_gewijzigd": out_tekst_kleur,
            "lijn_kleur_gewijzigd": out_lijn_kleur,
            "vul_kleur_gewijzigd": out_fill_gewijzigd,
            "nieuwe_gekleurde_vlakken": out_fills_nieuw,
            "verdwenen_gekleurde_vlakken": out_fills_weg,
            "lijnen_linewidth_gewijzigd": len(lijn_width),
            "lijnen_toegevoegd": len(lijnen_toegevoegd),
            "lijnen_toegevoegd_sample": _lijn_sample(lijnen_toegevoegd),
            "lijnen_verdwenen": len(lijnen_verdwenen),
            "lijnen_verdwenen_sample": _lijn_sample(lijnen_verdwenen),
            "nieuwe_wanden": [
                {"center": list(w["center"]), "bbox": list(w["bbox"]),
                 "dikte_pt": w["dikte_pt"], "wandtype": w.get("wandtype")}
                for w in nieuwe_wanden
            ],
            "verdwenen_wanden": [
                {"center": list(w["center"]), "bbox": list(w["bbox"]),
                 "dikte_pt": w["dikte_pt"], "wandtype": w.get("wandtype")}
                for w in verdwenen_wanden
            ],
            "kleur_inventaris": {
                "oud": inv_oud,
                "nieuw": inv_nieuw,
            },
            "totalen": {
                "tekst_gewijzigd": len(out_tekst_gewijzigd),
                "tekst_toegevoegd": len(out_tekst_toegevoegd),
                "tekst_verdwenen": len(out_tekst_verdwenen),
                "tekst_kleur_gewijzigd": len(out_tekst_kleur),
                "lijn_kleur_gewijzigd": len(out_lijn_kleur),
                "vul_kleur_gewijzigd": len(out_fill_gewijzigd),
                "nieuwe_gekleurde_vlakken": len(out_fills_nieuw),
                "verdwenen_gekleurde_vlakken": len(out_fills_weg),
                "lijnen_linewidth_gewijzigd": len(lijn_width),
                "lijnen_toegevoegd": len(lijnen_toegevoegd),
                "lijnen_verdwenen": len(lijnen_verdwenen),
                "nieuwe_wanden": len(nieuwe_wanden),
                "verdwenen_wanden": len(verdwenen_wanden),
            },
        }
    finally:
        oud_doc.close()
        nieuw_doc.close()
        for p in (oud_clean, nieuw_clean):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Wandvergelijking per wandtype (kleur-gebaseerd)
# ---------------------------------------------------------------------------

def compare_per_wandtype(
    oud_path: str,
    nieuw_path: str,
    pagina: int = 0,
    legenda_oud: dict | None = None,
    legenda_nieuw: dict | None = None,
    kleur_tolerantie: float = 0.10,
    verbose: bool = False,
) -> list[dict]:
    """
    Vergelijk wandwijzigingen per wandtype (kleur) tussen twee PDF's.

    Gebruikt de legenda om per kleur/wandtype te filteren — zo vergelijk je
    ~500 items per wandtype i.p.v. 65.000 lijnen tegelijk.

    Returns: lijst van wijzigingen met type, wandtype, positie, bbox, lengte.
    """
    oud_clean = strip_annotations(oud_path)
    nieuw_clean = strip_annotations(nieuw_path)

    oud_doc = fitz.open(oud_clean)
    nieuw_doc = fitz.open(nieuw_clean)

    try:
        if pagina >= len(oud_doc) or pagina >= len(nieuw_doc):
            logger.warning("Pagina %d bestaat niet in een van de PDF's", pagina)
            return []

        oud_page = oud_doc[pagina]
        nieuw_page = nieuw_doc[pagina]
        pw = nieuw_page.rect.width
        renvooi_x = pw * 0.88

        if legenda_oud is None:
            legenda_oud = _extract_legenda(oud_page)
        if legenda_nieuw is None:
            legenda_nieuw = _extract_legenda(nieuw_page)

        # Gecombineerde legenda: wandtype -> {kleur_oud, kleur_nieuw}
        gecombineerde_types: dict[str, dict] = {}
        for kleur, wandtype in legenda_oud.items():
            gecombineerde_types.setdefault(wandtype, {})["kleur_oud"] = kleur
        for kleur, wandtype in legenda_nieuw.items():
            gecombineerde_types.setdefault(wandtype, {})["kleur_nieuw"] = kleur

        # Filter: behoud alleen bekende wandtypes (skip "1200+", "peilmaat", etc.)
        def _is_bekend_wandtype(label: str) -> bool:
            label_norm = re.sub(r"\s+", "", label.strip().lower())
            return any(
                re.sub(r"\s+", "", b) in label_norm or label_norm in re.sub(r"\s+", "", b)
                for b in BEKENDE_WANDTYPEN
            )

        gecombineerde_types = {
            wt: kleuren for wt, kleuren in gecombineerde_types.items()
            if _is_bekend_wandtype(wt)
        }

        oud_items = _extract_lijnen(oud_page)
        nieuw_items = _extract_lijnen(nieuw_page)

        def _item_kleur(item):
            if "van" in item:
                return item.get("stroke")
            if item.get("type") == "fill":
                return item.get("rgb")
            return None

        def _item_pos(item):
            if "van" in item:
                return (
                    (item["van"][0] + item["naar"][0]) / 2,
                    (item["van"][1] + item["naar"][1]) / 2,
                )
            b = item.get("bbox")
            if b:
                return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            return item.get("pos", (0, 0))

        def _item_bbox(item):
            if "van" in item:
                x0 = min(item["van"][0], item["naar"][0])
                y0 = min(item["van"][1], item["naar"][1])
                x1 = max(item["van"][0], item["naar"][0])
                y1 = max(item["van"][1], item["naar"][1])
                if x0 == x1:
                    x1 += 1.0
                if y0 == y1:
                    y1 += 1.0
                return (x0, y0, x1, y1)
            b = item.get("bbox")
            if b:
                return tuple(b)
            return None

        def _in_renvooi(item):
            return _item_pos(item)[0] > renvooi_x

        def _kleur_dist(c1, c2):
            return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

        def _wijs_toe_exclusief(items, kleur_map: dict) -> dict:
            """Wijs elk item exclusief toe aan het dichtstbijzijnde wandtype.
            kleur_map: {wandtype: (r,g,b)}
            Geeft: {wandtype: [items]}  — elk item in maximaal één bucket.
            """
            from collections import defaultdict as _dd
            resultaat = _dd(list)
            for item in items:
                if _in_renvooi(item):
                    continue
                kleur = _item_kleur(item)
                if kleur is None:
                    continue
                beste_wt = None
                beste_d = kleur_tolerantie  # alleen toewijzen als binnen drempel
                for wt, zoek_kleur in kleur_map.items():
                    if zoek_kleur is None:
                        continue
                    d = _kleur_dist(kleur, zoek_kleur)
                    if d < beste_d:
                        beste_d = d
                        beste_wt = wt
                if beste_wt is not None:
                    resultaat[beste_wt].append(item)
            return resultaat

        def _match_items(oud_set, nieuw_set, drempel=10.0):
            matched_oud = set()
            matched_nieuw = set()
            for i, oud in enumerate(oud_set):
                pos_o = _item_pos(oud)
                beste_d = drempel
                beste_j = -1
                for j, nieuw in enumerate(nieuw_set):
                    if j in matched_nieuw:
                        continue
                    d = _afstand(pos_o, _item_pos(nieuw))
                    if d < beste_d:
                        beste_d = d
                        beste_j = j
                if beste_j >= 0:
                    matched_oud.add(i)
                    matched_nieuw.add(beste_j)
            verwijderd = [oud_set[i] for i in range(len(oud_set)) if i not in matched_oud]
            toegevoegd = [nieuw_set[j] for j in range(len(nieuw_set)) if j not in matched_nieuw]
            return verwijderd, toegevoegd

        def _cluster_items(items, max_afstand=50.0):
            if not items:
                return []
            clusters = []
            used = set()
            for i in range(len(items)):
                if i in used:
                    continue
                cluster = [i]
                used.add(i)
                pos_i = _item_pos(items[i])
                for j in range(i + 1, len(items)):
                    if j in used:
                        continue
                    if _afstand(pos_i, _item_pos(items[j])) <= max_afstand:
                        cluster.append(j)
                        used.add(j)
                clusters.append([items[k] for k in cluster])

            resultaat = []
            for cluster in clusters:
                bboxen = [_item_bbox(item) for item in cluster if _item_bbox(item)]
                if not bboxen:
                    continue
                x0 = min(b[0] for b in bboxen)
                y0 = min(b[1] for b in bboxen)
                x1 = max(b[2] for b in bboxen)
                y1 = max(b[3] for b in bboxen)
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                lengte = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                resultaat.append({
                    "bbox": (x0, y0, x1, y1),
                    "positie": (round(cx, 1), round(cy, 1)),
                    "lengte": round(lengte, 1),
                    "n_items": len(cluster),
                })
            return resultaat

        # Exclusieve kleur-maps: OUD items op kleur_oud, NIEUW items op kleur_nieuw
        oud_kleur_map = {
            wt: (kleuren.get("kleur_oud") or kleuren.get("kleur_nieuw"))
            for wt, kleuren in gecombineerde_types.items()
        }
        nieuw_kleur_map = {
            wt: (kleuren.get("kleur_nieuw") or kleuren.get("kleur_oud"))
            for wt, kleuren in gecombineerde_types.items()
        }

        oud_per_wt = _wijs_toe_exclusief(oud_items, oud_kleur_map)
        nieuw_per_wt = _wijs_toe_exclusief(nieuw_items, nieuw_kleur_map)

        # Analyse per wandtype
        tellingen: dict[str, dict] = {}
        alle_verwijderd: dict[str, dict] = {}
        alle_toegevoegd: dict[str, dict] = {}

        for wandtype, kleuren in gecombineerde_types.items():
            kleur_oud = kleuren.get("kleur_oud") or kleuren.get("kleur_nieuw")
            kleur_nieuw = kleuren.get("kleur_nieuw") or kleuren.get("kleur_oud")

            oud_gefilterd = list(oud_per_wt.get(wandtype, []))
            nieuw_gefilterd = list(nieuw_per_wt.get(wandtype, []))
            verwijderd_items, toegevoegd_items = _match_items(oud_gefilterd, nieuw_gefilterd)

            tellingen[wandtype] = {
                "kleur_oud": kleur_oud,
                "kleur_nieuw": kleur_nieuw,
                "oud": len(oud_gefilterd),
                "nieuw": len(nieuw_gefilterd),
                "verwijderd_items": len(verwijderd_items),
                "toegevoegd_items": len(toegevoegd_items),
            }
            alle_verwijderd[wandtype] = {
                "kleur": kleur_oud,
                "segmenten": _cluster_items(verwijderd_items),
            }
            alle_toegevoegd[wandtype] = {
                "kleur": kleur_nieuw,
                "segmenten": _cluster_items(toegevoegd_items),
            }

        if verbose:
            print(f"\nLegenda OUD ({len(legenda_oud)} entries):")
            for k, v in legenda_oud.items():
                print(f"  ({k[0]:.2f},{k[1]:.2f},{k[2]:.2f}) -> {v}")
            print(f"\nLegenda NIEUW ({len(legenda_nieuw)} entries):")
            for k, v in legenda_nieuw.items():
                print(f"  ({k[0]:.2f},{k[1]:.2f},{k[2]:.2f}) -> {v}")
            print(f"\nPer wandtype (tolerantie={kleur_tolerantie}):")
            for wt, t in sorted(tellingen.items()):
                seg_v = len(alle_verwijderd[wt]["segmenten"])
                seg_t = len(alle_toegevoegd[wt]["segmenten"])
                print(
                    f"  {wt:<45} | oud: {t['oud']:5d} | nieuw: {t['nieuw']:5d}"
                    f" | -items: {t['verwijderd_items']:4d} (+items: {t['toegevoegd_items']:4d})"
                    f" | -seg: {seg_v:3d} (+seg: {seg_t:3d})"
                )

        # GEWIJZIGD: cross-match verwijderd type A <-> toegevoegd type B
        gewijzigd_verwijderd: set[tuple] = set()
        gewijzigd_toegevoegd: set[tuple] = set()
        wijzigingen: list[dict] = []

        for wt_a, verwijderd_data in alle_verwijderd.items():
            kleur_a = verwijderd_data["kleur"]
            for seg_idx_a, seg_a in enumerate(verwijderd_data["segmenten"]):
                cx_a, cy_a = seg_a["positie"]
                beste_d = 50.0
                beste_match = None
                for wt_b, toegevoegd_data in alle_toegevoegd.items():
                    if wt_b == wt_a:
                        continue
                    kleur_b = toegevoegd_data["kleur"]
                    for seg_idx_b, seg_b in enumerate(toegevoegd_data["segmenten"]):
                        if (wt_b, seg_idx_b) in gewijzigd_toegevoegd:
                            continue
                        d = _afstand((cx_a, cy_a), seg_b["positie"])
                        if d < beste_d:
                            beste_d = d
                            beste_match = (wt_b, seg_idx_b, kleur_b, seg_b)

                if beste_match:
                    wt_b, seg_idx_b, kleur_b, seg_b = beste_match
                    gewijzigd_verwijderd.add((wt_a, seg_idx_a))
                    gewijzigd_toegevoegd.add((wt_b, seg_idx_b))
                    b_a, b_b = seg_a["bbox"], seg_b["bbox"]
                    bbox = (
                        min(b_a[0], b_b[0]), min(b_a[1], b_b[1]),
                        max(b_a[2], b_b[2]), max(b_a[3], b_b[3]),
                    )
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    wijzigingen.append({
                        "type": "gewijzigd",
                        "wandtype_oud": wt_a,
                        "wandtype_nieuw": wt_b,
                        "kleur_oud": list(kleur_a),
                        "kleur_nieuw": list(kleur_b),
                        "positie": [round(cx, 1), round(cy, 1)],
                        "bbox": [round(v, 1) for v in bbox],
                        "lengte": round(max(seg_a["lengte"], seg_b["lengte"]), 1),
                    })

        for wt, data in alle_verwijderd.items():
            for seg_idx, seg in enumerate(data["segmenten"]):
                if (wt, seg_idx) in gewijzigd_verwijderd:
                    continue
                wijzigingen.append({
                    "type": "verwijderd",
                    "wandtype_oud": wt,
                    "wandtype_nieuw": None,
                    "kleur_oud": list(data["kleur"]),
                    "kleur_nieuw": None,
                    "positie": [round(v, 1) for v in seg["positie"]],
                    "bbox": [round(v, 1) for v in seg["bbox"]],
                    "lengte": seg["lengte"],
                })

        for wt, data in alle_toegevoegd.items():
            for seg_idx, seg in enumerate(data["segmenten"]):
                if (wt, seg_idx) in gewijzigd_toegevoegd:
                    continue
                wijzigingen.append({
                    "type": "toegevoegd",
                    "wandtype_oud": None,
                    "wandtype_nieuw": wt,
                    "kleur_oud": None,
                    "kleur_nieuw": list(data["kleur"]),
                    "positie": [round(v, 1) for v in seg["positie"]],
                    "bbox": [round(v, 1) for v in seg["bbox"]],
                    "lengte": seg["lengte"],
                })

        wijzigingen.sort(key=lambda w: (w["positie"][1], w["positie"][0]))

        if verbose:
            print(f"\nWijzigingen ({len(wijzigingen)} totaal):")
            for w in wijzigingen:
                if w["type"] == "gewijzigd":
                    print(f"  [GEWIJZIGD]  {w['wandtype_oud']} -> {w['wandtype_nieuw']}"
                          f"  @ {w['positie']}  lengte={w['lengte']}pt")
                elif w["type"] == "verwijderd":
                    print(f"  [VERWIJDERD] {w['wandtype_oud']}"
                          f"  @ {w['positie']}  lengte={w['lengte']}pt")
                else:
                    print(f"  [TOEGEVOEGD] {w['wandtype_nieuw']}"
                          f"  @ {w['positie']}  lengte={w['lengte']}pt")

        return wijzigingen

    finally:
        oud_doc.close()
        nieuw_doc.close()
        for p in (oud_clean, nieuw_clean):
            try:
                os.unlink(p)
            except OSError:
                pass

"""
Diff engine voor K&K demarcatietekeningen.

Vergelijkt twee PDF's op tekst, lijnen, vullingen en kleuren via pymupdf.
Retourneert een dict met alle wijzigingen — geen print statements.
"""

import re
import fitz
from collections import defaultdict


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

def _extract_tekst(page) -> list[dict]:
    items = []
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except Exception:
        return items
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                tekst = span.get("text", "").strip()
                if not tekst:
                    continue
                r, g, b = _span_color_to_rgb(span.get("color", 0))
                bbox = span.get("bbox", (0, 0, 0, 0))
                items.append({
                    "tekst": tekst,
                    "rgb": (round(r, 3), round(g, 3), round(b, 3)),
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
            if abs(oud["width"] - nieuw["width"]) >= 0.5:
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


def _categoriseer_tekst_wijziging(oud_tekst: str, nieuw_tekst: str) -> str:
    if _RE_NUMERIEK.match(oud_tekst) and _RE_NUMERIEK.match(nieuw_tekst):
        return "maat"
    if _RE_LETTER.match(oud_tekst) or _RE_LETTER.match(nieuw_tekst):
        return "revisieletter"
    if "m\u00b2" in oud_tekst or "m\u00b2" in nieuw_tekst:
        return "oppervlakte"
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
# Hoofdfunctie
# ---------------------------------------------------------------------------

def run_diff(
    oud_pdf_path: str,
    nieuw_pdf_path: str,
    pagina: int = 0,
) -> dict:
    """
    Vergelijk twee PDF's en retourneer een dict met alle wijzigingen.

    Parameters:
        oud_pdf_path: pad naar de oude PDF
        nieuw_pdf_path: pad naar de nieuwe PDF
        pagina: 0-based pagina index

    Returns:
        dict met alle diff-secties
    """
    oud_doc = fitz.open(oud_pdf_path)
    nieuw_doc = fitz.open(nieuw_pdf_path)

    try:
        if pagina >= len(oud_doc) or pagina >= len(nieuw_doc):
            return {
                "error": f"Pagina {pagina + 1} bestaat niet. "
                         f"OUD heeft {len(oud_doc)} pagina's, "
                         f"NIEUW heeft {len(nieuw_doc)} pagina's."
            }

        oud_page = oud_doc[pagina]
        nieuw_page = nieuw_doc[pagina]

        # Extractie
        oud_tekst = _extract_tekst(oud_page)
        nieuw_tekst = _extract_tekst(nieuw_page)
        oud_lijnen = _extract_lijnen(oud_page)
        nieuw_lijnen = _extract_lijnen(nieuw_page)

        # Vergelijkingen
        tekst_gewijzigd, tekst_toegevoegd, tekst_verdwenen, tekst_kleur = \
            _vergelijk_tekst(oud_tekst, nieuw_tekst, drempel=15.0)

        lijn_width, lijn_kleur, lijnen_toegevoegd, lijnen_verdwenen = \
            _vergelijk_lijnen(oud_lijnen, nieuw_lijnen, drempel=5.0)

        fill_gewijzigd, fills_toegevoegd, fills_verdwenen = \
            _vergelijk_fills(oud_lijnen, nieuw_lijnen, drempel=10.0)

        fills_toegevoegd_groot = [f for f in fills_toegevoegd if f["oppervlakte"] > 100]
        fills_verdwenen_groot = [f for f in fills_verdwenen if f["oppervlakte"] > 100]

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
                "oud_kleur": oh,
                "nieuw_kleur": on_h,
                "categorie": _categoriseer_tekst_wijziging(o["tekst"], n["tekst"]),
            })

        # 2. Tekst toegevoegd
        out_tekst_toegevoegd = []
        for t in _sort_pos(tekst_toegevoegd):
            h, n = _hex_naam(*t["rgb"])
            out_tekst_toegevoegd.append({
                "tekst": t["tekst"],
                "pos": list(t["pos"]),
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
                "kleur_hex": h,
                "kleur_naam": n,
            })

        # 4. Tekstkleur gewijzigd
        out_tekst_kleur = []
        for w in sorted(tekst_kleur, key=lambda x: (x["pos"][1], x["pos"][0])):
            oh, on_ = _hex_naam(*w["oud_rgb"])
            nh, nn_ = _hex_naam(*w["nieuw_rgb"])
            out_tekst_kleur.append({
                "tekst": w["tekst"],
                "pos": list(w["pos"]),
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

        # 7. Vulkleur gewijzigd
        out_fill_gewijzigd = []
        for w in _sort_pos(fill_gewijzigd):
            oh, on_ = _hex_naam(*w["oud_rgb"])
            nh, nn_ = _hex_naam(*w["nieuw_rgb"])
            out_fill_gewijzigd.append({
                "pos": list(w["pos"]),
                "bbox": list(w["bbox"]),
                "oppervlakte": w["oppervlakte"],
                "oud_kleur": oh,
                "oud_naam": on_,
                "nieuw_kleur": nh,
                "nieuw_naam": nn_,
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

        return {
            "meta": {
                "oud_bestand": oud_pdf_path,
                "nieuw_bestand": nieuw_pdf_path,
                "pagina": pagina + 1,
                "oud_woorden": len(oud_tekst),
                "nieuw_woorden": len(nieuw_tekst),
                "oud_lijnen": oud_lijn_count,
                "nieuw_lijnen": nieuw_lijn_count,
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
            },
        }
    finally:
        oud_doc.close()
        nieuw_doc.close()

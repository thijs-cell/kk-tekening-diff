"""Text-based comparison of PDF floor plans using pdfplumber.

Extracts words from each page, categorizes them, and reports differences
between old and new versions. No AI calls, no image rendering.
"""

import io
import math
import re
from collections import defaultdict

import pdfplumber


# --- Character filtering ---

def _is_annotation_color(color) -> bool:
    """Return True if color is a bright annotation color (not black/dark)."""
    if color is None:
        return False
    if isinstance(color, (int, float)):
        return float(color) > 0.4
    if isinstance(color, (list, tuple)):
        c = tuple(float(v) for v in color)
        if len(c) == 1:
            return c[0] > 0.4
        if len(c) == 3:
            r, g, b = c
            if max(r, g, b) < 0.35:
                return False
            if r < 0.2 and g < 0.4 and b < 0.55:
                return False
            return True
        if len(c) == 4:
            _, _, _, k = c
            return k < 0.5
    return False


def _filter_chars(chars: list) -> list:
    """Filter out rotated chars and annotation colors."""
    return [
        ch for ch in chars
        if ch.get("upright") is not False
        and not _is_annotation_color(
            ch.get("stroking_color") or ch.get("non_stroking_color")
        )
    ]


# --- Word merging ---

def _merge_chars_to_words(
    chars: list, x_gap: float = 2.0, y_tol: float = 1.5,
) -> list[dict]:
    """Merge individual characters into words based on proximity."""
    if not chars:
        return []

    sorted_chars = sorted(
        chars, key=lambda c: (round(c["top"] / y_tol) * y_tol, c["x0"]),
    )

    words: list[dict] = []
    cur = {
        "text": sorted_chars[0].get("text", ""),
        "x0": sorted_chars[0]["x0"],
        "y0": sorted_chars[0]["top"],
        "x1": sorted_chars[0]["x1"],
        "y1": sorted_chars[0]["bottom"],
        "fontsize": sorted_chars[0].get("size", 0),
    }

    for ch in sorted_chars[1:]:
        same_line = abs(ch["top"] - cur["y0"]) < y_tol
        gap = ch["x0"] - cur["x1"]
        close_x = -1.0 <= gap < x_gap

        if same_line and close_x:
            cur["text"] += ch.get("text", "")
            cur["x1"] = ch["x1"]
            cur["y1"] = max(cur["y1"], ch["bottom"])
        else:
            if cur["text"].strip():
                words.append(cur)
            cur = {
                "text": ch.get("text", ""),
                "x0": ch["x0"],
                "y0": ch["top"],
                "x1": ch["x1"],
                "y1": ch["bottom"],
                "fontsize": ch.get("size", 0),
            }

    if cur["text"].strip():
        words.append(cur)
    return words


# --- Word categorization ---

_RUIMTE_KEYWORDS = [
    "kamer", "berging", "toilet", "badkamer", "badk", "tech",
    "entree", "woon", "keuken", "hal", "gang", "lift", "trap",
    "slaap", "loft", "galerij", "meterruimte", "hydrofoor",
    "buitenruimte", "buitenberging", "werkkast", "flatkast",
]

_WAND_KEYWORDS = [
    "gibo", "kalkzandsteen", "prefab", "beton", "hsb",
    "isolatie", "hardschuim", "pir", "osb", "rhombus", "mato",
    "sandwichpaneel", "voorzetwand", "biobased",
]

RELEVANT_CATEGORIES = {
    "ruimtenaam", "ruimtenummer", "oppervlakte", "getal",
    "maat_mm", "deurmaat", "merkcode", "wandcode",
    "geluideis", "brandeis", "peilmaat",
}

RENVOOI_X_MIN = 2100.0


def _categorize_word(text: str) -> str | None:
    t = text.strip()
    if not t:
        return None
    if t.endswith("m\u00b2") or t.endswith("m2"):
        return "oppervlakte"
    if re.match(r"^\d+mm$", t, re.IGNORECASE):
        return "maat_mm"
    if re.match(r"^\d{2,5}$", t):
        return "getal"
    if re.match(r"^dm=\d+$", t, re.IGNORECASE):
        return "deurmaat"
    t_lower = t.lower()
    for kw in _RUIMTE_KEYWORDS:
        if kw in t_lower:
            return "ruimtenaam"
    for kw in _WAND_KEYWORDS:
        if kw in t_lower:
            return "wandcode"
    if re.match(r"^Rw", t) or "dB" in t:
        return "geluideis"
    if "brandwerend" in t_lower or re.match(r"^\d+\s*min", t_lower):
        return "brandeis"
    if re.match(r"^merk\s", t, re.IGNORECASE):
        return "merkcode"
    if re.match(r"^[+-]\d+\s*[+]?P$", t):
        return "peilmaat"
    if re.match(r"^\d+\.\w+\.?\w*\d+", t):
        return "ruimtenummer"
    return None


# --- Page extraction ---

def _extract_page_words(page) -> list[dict]:
    """Extract and categorize words from a single pdfplumber page."""
    chars = page.chars or []
    filtered = _filter_chars(chars)
    words = _merge_chars_to_words(filtered)
    for w in words:
        w["category"] = _categorize_word(w["text"])
        w["is_renvooi"] = w["x0"] > RENVOOI_X_MIN
    return words


# --- Spatial matching ---

def _word_center(w: dict) -> tuple[float, float]:
    return ((w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2)


def _distance(w1: dict, w2: dict) -> float:
    cx1, cy1 = _word_center(w1)
    cx2, cy2 = _word_center(w2)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def _find_nearest(
    word: dict, candidates: list[dict], max_dist: float = 15.0,
) -> dict | None:
    best = None
    best_d = max_dist + 1
    for c in candidates:
        d = _distance(word, c)
        if d < best_d:
            best_d = d
            best = c
    return best if best_d <= max_dist else None


def _find_location_context(
    word: dict, all_words: list[dict], max_dist: float = 200.0,
) -> str:
    best_label = ""
    best_d = max_dist + 1
    for w in all_words:
        if w["category"] not in ("ruimtenaam", "ruimtenummer"):
            continue
        if w.get("is_renvooi"):
            continue
        d = _distance(word, w)
        if d < best_d:
            best_d = d
            best_label = w["text"].strip()
    return best_label


# --- Page comparison ---

def _compare_pages(
    old_words: list[dict],
    new_words: list[dict],
    page_num: int,
) -> list[dict]:
    changes: list[dict] = []

    old_relevant = [
        w for w in old_words
        if w["category"] in RELEVANT_CATEGORIES and not w.get("is_renvooi")
    ]
    new_relevant = [
        w for w in new_words
        if w["category"] in RELEVANT_CATEGORIES and not w.get("is_renvooi")
    ]

    old_matched: set[int] = set()
    new_matched: set[int] = set()

    # Pass 1: find changed text at same position
    for ni, nw in enumerate(new_relevant):
        nearest = _find_nearest(nw, old_relevant)
        if nearest is None:
            continue
        oi = old_relevant.index(nearest)
        if oi in old_matched:
            continue

        old_matched.add(oi)
        new_matched.add(ni)

        old_text = nearest["text"].strip()
        new_text = nw["text"].strip()

        if old_text != new_text:
            ctx = _find_location_context(nw, new_words)
            changes.append({
                "type": "gewijzigd",
                "pagina": page_num,
                "categorie": nw["category"],
                "oud": old_text,
                "nieuw": new_text,
                "x": round(nw["x0"], 1),
                "y": round(nw["y0"], 1),
                "locatie": ctx,
            })

    # Pass 2: new words without match = toegevoegd
    for ni, nw in enumerate(new_relevant):
        if ni in new_matched:
            continue
        if _find_nearest(nw, old_relevant) is not None:
            continue
        ctx = _find_location_context(nw, new_words)
        changes.append({
            "type": "toegevoegd",
            "pagina": page_num,
            "categorie": nw["category"],
            "tekst": nw["text"].strip(),
            "x": round(nw["x0"], 1),
            "y": round(nw["y0"], 1),
            "locatie": ctx,
        })

    # Pass 3: old words without match = verdwenen
    for oi, ow in enumerate(old_relevant):
        if oi in old_matched:
            continue
        if _find_nearest(ow, new_relevant) is not None:
            continue
        ctx = _find_location_context(ow, old_words)
        changes.append({
            "type": "verdwenen",
            "pagina": page_num,
            "categorie": ow["category"],
            "tekst": ow["text"].strip(),
            "x": round(ow["x0"], 1),
            "y": round(ow["y0"], 1),
            "locatie": ctx,
        })

    return changes


# --- Public API ---

def compare_pdfs_data(
    old_bytes: bytes,
    new_bytes: bytes,
    max_pages: int = 1,
) -> dict:
    """Compare two PDFs by extracting and diffing text data.

    Returns the same JSON structure as compare_results.json.
    """
    all_changes: list[dict] = []

    with pdfplumber.open(io.BytesIO(old_bytes)) as old_pdf, \
         pdfplumber.open(io.BytesIO(new_bytes)) as new_pdf:

        num_pages = min(max_pages, len(old_pdf.pages), len(new_pdf.pages))

        for p in range(num_pages):
            page_num = p + 1
            old_words = _extract_page_words(old_pdf.pages[p])
            new_words = _extract_page_words(new_pdf.pages[p])
            changes = _compare_pages(old_words, new_words, page_num)
            all_changes.extend(changes)

    by_type: dict[str, list] = defaultdict(list)
    for ch in all_changes:
        by_type[ch["type"]].append(ch)

    return {
        "samenvatting": {
            "gewijzigd": len(by_type.get("gewijzigd", [])),
            "toegevoegd": len(by_type.get("toegevoegd", [])),
            "verdwenen": len(by_type.get("verdwenen", [])),
            "totaal": len(all_changes),
        },
        "wijzigingen": all_changes,
    }

"""Text-based comparison of PDF floor plans using pdfplumber.

Extracts words from each page, categorizes them, and reports differences
between old and new versions. No AI calls, no image rendering.

Reports 5 change categories:
- ruimtenaam: room name changed
- maatvoering: dimension (mm) changed at interior wall
- wanddikte: wall thickness changed
- kozijnhoogte: lintel height changed (2000-2400mm range)
- indeling: walls or rooms added/removed
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

_WAND_PROXIMITY_KEYWORDS = ["gibo", "kalkzandsteen", "grw"]

_WANDDIKTE_VALUES = {50, 70, 100, 120}

_KOZIJNHOOGTE_MIN = 2000
_KOZIJNHOOGTE_MAX = 2400

RELEVANT_CATEGORIES = {
    "ruimtenaam", "getal", "wandcode",
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


# --- Linewidth analysis ---

def _extract_linewidths(page) -> dict[float, int]:
    """Extract line counts grouped by linewidth from a pdfplumber page."""
    counts: dict[float, int] = defaultdict(int)
    lines = page.lines or []
    for line in lines:
        lw = round(float(line.get("linewidth", 0) or 0), 2)
        if lw > 0:
            counts[lw] += 1
    return dict(counts)


def _compare_linewidths(
    old_counts: dict[float, int],
    new_counts: dict[float, int],
    page_num: int,
) -> list[dict]:
    """Compare linewidth distributions and report significant changes."""
    changes: list[dict] = []
    all_widths = set(old_counts.keys()) | set(new_counts.keys())

    for lw in sorted(all_widths):
        old_n = old_counts.get(lw, 0)
        new_n = new_counts.get(lw, 0)
        if old_n == new_n:
            continue
        total = max(old_n, new_n)
        diff_pct = abs(new_n - old_n) / total if total > 0 else 0
        if diff_pct > 0.10:
            changes.append({
                "type": "gewijzigd",
                "pagina": page_num,
                "categorie": "wanddikte",
                "prioriteit": "hoog",
                "betrouwbaarheid": "normaal",
                "bron": "linewidth",
                "linewidth": lw,
                "oud_aantal": old_n,
                "nieuw_aantal": new_n,
                "opmerking": (
                    f"Mogelijke wanddikte-wijziging: {old_n} lijnen met "
                    f"dikte {lw} in oud, {new_n} lijnen in nieuw"
                ),
            })

    return changes


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
    """Find nearest room name for location context."""
    best_label = ""
    best_d = max_dist + 1
    for w in all_words:
        if w["category"] != "ruimtenaam":
            continue
        if w.get("is_renvooi"):
            continue
        d = _distance(word, w)
        if d < best_d:
            best_d = d
            best_label = w["text"].strip()
    return best_label


def _has_nearby_wandcode(word: dict, all_words: list[dict], max_dist: float = 50.0) -> bool:
    """Check if a getal word has a wall code keyword nearby."""
    for w in all_words:
        if w is word:
            continue
        t_lower = w["text"].strip().lower()
        for kw in _WAND_PROXIMITY_KEYWORDS:
            if kw in t_lower:
                if _distance(word, w) <= max_dist:
                    return True
    return False


def _is_valid_getal(text: str) -> bool:
    """Check if a getal is valid for maatvoering (3-5 digits)."""
    t = text.strip()
    if not re.match(r"^\d+$", t):
        return False
    if len(t) <= 2:
        return False
    if len(t) > 5:
        return False
    return True


def _is_pure_number(text: str) -> bool:
    """Check if text is purely a number (no letters)."""
    return bool(re.match(r"^\d+$", text.strip()))


# --- Betrouwbaarheid & prioriteit helpers ---

def _calc_betrouwbaarheid(diff_mm: int) -> str:
    abs_diff = abs(diff_mm)
    if abs_diff > 2000:
        return "onzeker"
    if abs_diff >= 500:
        return "normaal"
    return "hoog"


def _calc_prioriteit_maat(diff_mm: int) -> str:
    if abs(diff_mm) > 30:
        return "normaal"
    return "laag"


def _is_ruimtenaam_type_change(old_text: str, new_text: str) -> bool:
    """Check if room name changed to a different type (e.g. Slaapkamer → Technische ruimte)."""
    old_lower = old_text.lower()
    new_lower = new_text.lower()
    old_type = None
    new_type = None
    for kw in _RUIMTE_KEYWORDS:
        if kw in old_lower:
            old_type = kw
            break
    for kw in _RUIMTE_KEYWORDS:
        if kw in new_lower:
            new_type = kw
            break
    if old_type and new_type and old_type != new_type:
        return True
    return False


# --- Shift detection ---

def _remove_shift_pairs(changes: list[dict]) -> list[dict]:
    """Remove toegevoegd+verdwenen pairs at nearly the same position (within 25pt)."""
    added = [c for c in changes if c["type"] == "toegevoegd"]
    removed = [c for c in changes if c["type"] == "verdwenen"]
    other = [c for c in changes if c["type"] not in ("toegevoegd", "verdwenen")]

    skip_added: set[int] = set()
    skip_removed: set[int] = set()

    for ai, a in enumerate(added):
        if ai in skip_added:
            continue
        for ri, r in enumerate(removed):
            if ri in skip_removed:
                continue
            if a.get("tekst") != r.get("tekst"):
                continue
            if a.get("categorie") != r.get("categorie"):
                continue
            dx = a["x"] - r["x"]
            dy = a["y"] - r["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= 25.0:
                skip_added.add(ai)
                skip_removed.add(ri)
                break

    result = list(other)
    result.extend(a for ai, a in enumerate(added) if ai not in skip_added)
    result.extend(r for ri, r in enumerate(removed) if ri not in skip_removed)
    return result


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
                "_word_new": nw,
                "_word_old": nearest,
                "_all_new": new_words,
                "_all_old": old_words,
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
            "_word": nw,
            "_all_words": new_words,
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
            "_word": ow,
            "_all_words": old_words,
        })

    return changes


# --- Post-processing: map raw changes to 5 output categories ---

def _classify_changes(raw_changes: list[dict]) -> list[dict]:
    """Map raw changes to output categories and filter noise."""
    result: list[dict] = []

    for ch in raw_changes:
        cat = ch["categorie"]
        typ = ch["type"]

        # 1. RUIMTENAAM — only gewijzigd room names
        if cat == "ruimtenaam" and typ == "gewijzigd":
            # FIX 3: skip if "oud" is a pure number (matching error)
            if _is_pure_number(ch.get("oud", "")):
                continue
            out = _clean(ch, "ruimtenaam")
            if _is_ruimtenaam_type_change(ch["oud"], ch["nieuw"]):
                out["prioriteit"] = "hoog"
            else:
                out["prioriteit"] = "normaal"
            out["betrouwbaarheid"] = "hoog"
            result.append(out)
            continue

        # 2/3/5. GETAL gewijzigd — maatvoering, wanddikte, or kozijnhoogte
        if cat == "getal" and typ == "gewijzigd":
            old_val = ch["oud"]
            new_val = ch["nieuw"]

            if not _is_valid_getal(old_val) or not _is_valid_getal(new_val):
                continue

            old_num = int(old_val)
            new_num = int(new_val)
            diff = new_num - old_num
            abs_diff = abs(diff)

            # FIX 1: filter samengeplakte getallen (>4 digits + >2000mm diff)
            if (len(old_val) > 4 or len(new_val) > 4) and abs_diff > 2000:
                continue

            # Check if this is a wall thickness change
            word_new = ch.get("_word_new")
            all_new = ch.get("_all_new", [])
            is_wand = (
                (old_num in _WANDDIKTE_VALUES or new_num in _WANDDIKTE_VALUES)
                and word_new is not None
                and _has_nearby_wandcode(word_new, all_new)
            )

            # Check if this is a kozijnhoogte change
            is_kozijn = (
                _KOZIJNHOOGTE_MIN <= old_num <= _KOZIJNHOOGTE_MAX
                or _KOZIJNHOOGTE_MIN <= new_num <= _KOZIJNHOOGTE_MAX
            )

            verschil = _format_diff(old_num, new_num)
            betrouwbaarheid = _calc_betrouwbaarheid(diff)

            if is_wand:
                out = _clean(ch, "wanddikte")
                out["verschil"] = verschil
                out["prioriteit"] = "hoog"
                out["betrouwbaarheid"] = betrouwbaarheid
                result.append(out)
            elif is_kozijn:
                out = _clean(ch, "kozijnhoogte")
                out["verschil"] = verschil
                out["prioriteit"] = "hoog"
                out["betrouwbaarheid"] = betrouwbaarheid
                result.append(out)
            else:
                out = _clean(ch, "maatvoering")
                out["verschil"] = verschil
                out["prioriteit"] = _calc_prioriteit_maat(diff)
                out["betrouwbaarheid"] = betrouwbaarheid
                result.append(out)
            continue

        # 3. WANDCODE gewijzigd — wall type change = wanddikte
        if cat == "wandcode" and typ == "gewijzigd":
            out = _clean(ch, "wanddikte")
            out["prioriteit"] = "hoog"
            out["betrouwbaarheid"] = "hoog"
            result.append(out)
            continue

        # 4. INDELING — wandcode or ruimtenaam toegevoegd/verdwenen
        if cat == "wandcode" and typ in ("toegevoegd", "verdwenen"):
            out = _clean(ch, "indeling")
            out["prioriteit"] = "hoog"
            out["betrouwbaarheid"] = "hoog"
            result.append(out)
            continue

        if cat == "ruimtenaam" and typ in ("toegevoegd", "verdwenen"):
            out = _clean(ch, "indeling")
            out["prioriteit"] = "hoog"
            out["betrouwbaarheid"] = "hoog"
            result.append(out)
            continue

    return result


def _clean(ch: dict, new_cat: str) -> dict:
    """Return a clean output dict without internal fields."""
    out = {k: v for k, v in ch.items() if not k.startswith("_")}
    out["categorie"] = new_cat
    return out


def _format_diff(old_val: int, new_val: int) -> str:
    diff = new_val - old_val
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff}mm"


# --- Public API ---

def compare_pdfs_data(
    old_bytes: bytes,
    new_bytes: bytes,
    max_pages: int = 1,
) -> dict:
    """Compare two PDFs by extracting and diffing text data.

    Returns JSON with 5 change categories:
    ruimtenaam, maatvoering, wanddikte, kozijnhoogte, indeling.
    """
    all_changes: list[dict] = []
    linewidth_changes: list[dict] = []

    with pdfplumber.open(io.BytesIO(old_bytes)) as old_pdf, \
         pdfplumber.open(io.BytesIO(new_bytes)) as new_pdf:

        num_pages = min(max_pages, len(old_pdf.pages), len(new_pdf.pages))

        for p in range(num_pages):
            page_num = p + 1
            old_page = old_pdf.pages[p]
            new_page = new_pdf.pages[p]

            old_words = _extract_page_words(old_page)
            new_words = _extract_page_words(new_page)
            changes = _compare_pages(old_words, new_words, page_num)
            all_changes.extend(changes)

            # Linewidth-based wanddikte detection
            old_lw = _extract_linewidths(old_page)
            new_lw = _extract_linewidths(new_page)
            lw_changes = _compare_linewidths(old_lw, new_lw, page_num)
            linewidth_changes.extend(lw_changes)

    # Remove shift pairs before classification
    all_changes = _remove_shift_pairs(all_changes)

    # Classify into output categories
    all_changes = _classify_changes(all_changes)

    # Append linewidth-based changes
    all_changes.extend(linewidth_changes)

    by_type: dict[str, list] = defaultdict(list)
    for ch in all_changes:
        by_type[ch["type"]].append(ch)

    by_cat: dict[str, list] = defaultdict(list)
    for ch in all_changes:
        by_cat[ch["categorie"]].append(ch)

    by_prio: dict[str, list] = defaultdict(list)
    for ch in all_changes:
        by_prio[ch.get("prioriteit", "normaal")].append(ch)

    return {
        "samenvatting": {
            "gewijzigd": len(by_type.get("gewijzigd", [])),
            "toegevoegd": len(by_type.get("toegevoegd", [])),
            "verdwenen": len(by_type.get("verdwenen", [])),
            "totaal": len(all_changes),
            "per_categorie": {
                "ruimtenaam": len(by_cat.get("ruimtenaam", [])),
                "maatvoering": len(by_cat.get("maatvoering", [])),
                "wanddikte": len(by_cat.get("wanddikte", [])),
                "kozijnhoogte": len(by_cat.get("kozijnhoogte", [])),
                "indeling": len(by_cat.get("indeling", [])),
            },
            "per_prioriteit": {
                "hoog": len(by_prio.get("hoog", [])),
                "normaal": len(by_prio.get("normaal", [])),
                "laag": len(by_prio.get("laag", [])),
            },
        },
        "wijzigingen": all_changes,
    }

"""
Tekening-profiel detectie: oriëntatie, schaal en legenda.

Elke tekening leert het systeem over zichzelf via deze drie functies
voordat er iets vergeleken wordt. Geen hardcoded aannames over positie,
kleur of schaal.
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def _is_bijna_wit(rgb: tuple) -> bool:
    return all(c > 0.90 for c in rgb[:3])


def _is_bijna_zwart(rgb: tuple) -> bool:
    return all(c < 0.10 for c in rgb[:3])


def _is_neutraal(rgb: tuple) -> bool:
    """Wit, zwart of egaal grijs — niet bruikbaar als wandtype-kleur."""
    if _is_bijna_wit(rgb) or _is_bijna_zwart(rgb):
        return True
    r, g, b = rgb[:3]
    # Egaal grijs: alle kanalen bijna gelijk EN in middenbereik
    return abs(r - g) < 0.04 and abs(g - b) < 0.04 and abs(r - b) < 0.04 and 0.3 < r < 0.8


def _rnd(rgb: tuple) -> tuple:
    return tuple(round(c, 3) for c in rgb[:3])


# ---------------------------------------------------------------------------
# Functie 1: oriëntatie
# ---------------------------------------------------------------------------

def detecteer_orientatie(page) -> dict:
    """
    Detecteert paginaoriëntatie en geeft een normalize-functie terug die
    raw PDF-coördinaten (zoals teruggegeven door get_text / get_drawings)
    omzet naar display-coördinaten (wat de gebruiker ziet).

    Returns:
        {
            "rotation": int,          # 0 / 90 / 180 / 270
            "display_width": float,
            "display_height": float,
            "normalize": Callable[[float, float], tuple[float, float]],
        }

    Ondersteunde gevallen:
    - rotation=0, willekeurige mediabox (incl. negatieve origin)
    - rotation=90  (Muiden D2.1: portret opgeslagen, landscape weergegeven)
    - rotation=180 / 270
    """
    rotation = page.rotation % 360
    mb = page.mediabox          # Rect(x0, y0, x1, y1) — kan negatieve origin hebben
    mb_w = float(mb.width)      # breedte van de ongeroteerde pagina
    mb_h = float(mb.height)     # hoogte van de ongeroteerde pagina

    # Display-afmetingen na rotatie
    if rotation in (90, 270):
        display_width = mb_h
        display_height = mb_w
    else:
        display_width = mb_w
        display_height = mb_h

    # PyMuPDF normaliseert de mediabox-offset al in de teruggegeven coördinaten
    # (negatieve origins worden weggewerkt door de interne transformatiematrix).
    # Voor rotation=0 zijn raw coördinaten == display coördinaten.
    # Voor rotation=90 geldt: display_x = display_width − raw_y, display_y = raw_x

    dw = display_width
    dh = display_height

    if rotation == 0:
        def normalize(x: float, y: float) -> tuple[float, float]:
            return (float(x), float(y))

    elif rotation == 90:
        def normalize(x: float, y: float) -> tuple[float, float]:
            return (dw - float(y), float(x))

    elif rotation == 180:
        def normalize(x: float, y: float) -> tuple[float, float]:
            return (dw - float(x), dh - float(y))

    elif rotation == 270:
        def normalize(x: float, y: float) -> tuple[float, float]:
            return (float(y), dh - float(x))

    else:
        def normalize(x: float, y: float) -> tuple[float, float]:
            return (float(x), float(y))

    return {
        "rotation": rotation,
        "display_width": display_width,
        "display_height": display_height,
        "normalize": normalize,
    }


# ---------------------------------------------------------------------------
# Functie 2: schaal
# ---------------------------------------------------------------------------

_RE_SCHAAL = re.compile(r"1\s*:\s*(\d+)")


def detecteer_schaal(page) -> float:
    """
    Detecteert de tekenischaal (bijv. 50.0 voor 1:50).

    Aanpak:
    1. Zoek spans met patroon "1:50" of "1 : 100" etc.
    2. Zoek losse spans: "SCHAAL" gevolgd door een "1:N"-span binnen 200pt.

    Fallback: 100.0

    Afgeleide waarde die de aanroeper zelf kan berekenen:
        mm_per_pt = schaal * 25.4 / 72
        - 1:50  → 17.64 mm/pt
        - 1:100 → 35.28 mm/pt
    """
    blokken = page.get_text("rawdict")["blocks"]
    spans: list[dict] = []

    for b in blokken:
        if b.get("type") != 0:
            continue
        for lijn in b.get("lines", []):
            for span in lijn.get("spans", []):
                txt = "".join(c["c"] for c in span.get("chars", [])).strip()
                if txt:
                    spans.append({"tekst": txt, "bbox": span["bbox"]})

    # Stap 1: schaal in één span (meest voorkomend)
    for s in spans:
        m = _RE_SCHAAL.search(s["tekst"])
        if m:
            return float(m.group(1))

    # Stap 2: losse spans — "SCHAAL" of "schaal" dicht bij "1:N"
    schaal_spans = [s for s in spans if re.search(r"\bschaal\b", s["tekst"], re.I)]
    for ss in schaal_spans:
        sx, sy = ss["bbox"][0], ss["bbox"][1]
        for s in spans:
            m = _RE_SCHAAL.search(s["tekst"])
            if m:
                dx = abs(s["bbox"][0] - sx)
                dy = abs(s["bbox"][1] - sy)
                if dx < 250 and dy < 60:
                    return float(m.group(1))

    return 100.0  # fallback


# ---------------------------------------------------------------------------
# Functie 3: legenda
# ---------------------------------------------------------------------------

_LEGENDA_TITELS = re.compile(r"\b(legenda|renvooi)\b", re.I)
_WANDTYPE_TERMEN = re.compile(
    r"\b(kalkzandsteen|gibo|isolatie|metselwerk|beton|prefab|hsb|sandwich|pir|"
    r"rhombus|hardschuim|achterwand|voorzetwand|mato|stuc|biobased|gyproc|cellenbeton|"
    r"ytong|poriso|siporex|damwand|glaswand|systeemwand|scheidingswand|binnenwand|"
    r"buitenwand|draagwand|spouwwand|brandwand|staalstud)\b",
    re.I,
)
_LEGENDA_RADIUS = 700       # pt — zoekzone rond de legenda-titel
_MAX_SWATCH_AREA = 8_000    # pt² — swatches zijn klein; wanden zijn groter
_X_ZOEKBEREIK = (-50, 320)  # tekst mag t/m 320pt rechts van swatch staan (display)
_Y_TOLERANTIE = 40          # pt — y-verschil swatch ↔ label
_MIN_LABEL_LENGTE = 4


def _vind_legenda_titels(page, normalize: Callable) -> list[tuple[float, float]]:
    """Retourneert display-posities van alle 'legenda'/'renvooi' titels op de pagina."""
    posities = []
    for b in page.get_text("rawdict")["blocks"]:
        if b.get("type") != 0:
            continue
        for lijn in b.get("lines", []):
            for span in lijn.get("spans", []):
                txt = "".join(c["c"] for c in span.get("chars", [])).strip()
                if _LEGENDA_TITELS.search(txt):
                    bx, by = span["bbox"][0], span["bbox"][1]
                    posities.append(normalize(bx, by))
    return posities


def vind_legenda(page, orientatie: dict) -> dict:
    """
    Zoekt de legenda/renvooi op de pagina via de 'legenda'/'renvooi' titel
    en retourneert een mapping van kleur (rgb-tuple) naar wandtype-naam.

    Werkt met:
    - Fill-gebaseerde swatches (gekleurde gevulde rechthoeken)
    - Stroke-gebaseerde swatches (witte fills + gekleurde arceringlijnen),
      zoals in Muiden D2.1 (rotation=90)

    Args:
        page: PyMuPDF page-object
        orientatie: resultaat van detecteer_orientatie(page)

    Returns:
        dict  bijv. {(0.25, 0.5, 0.5): "kalkzandsteenwand, CS12", ...}
        Lege dict als geen legenda gevonden.
    """
    normalize: Callable = orientatie["normalize"]

    # --- Stap 0: vind legenda-titels om zoekzone te bepalen -----------------
    titel_posities = _vind_legenda_titels(page, normalize)

    def in_legenda_zone(dx: float, dy: float) -> bool:
        if not titel_posities:
            return True  # geen titel gevonden: zoek hele pagina (fallback)
        return any(
            abs(dx - tx) < _LEGENDA_RADIUS and abs(dy - ty) < _LEGENDA_RADIUS
            for tx, ty in titel_posities
        )

    # --- Stap 1: verzamel gekleurde elementen binnen de legenda-zone --------
    gekleurde: list[dict] = []

    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue

        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if area > _MAX_SWATCH_AREA:
            continue

        rep_kleur: tuple | None = None

        fill = d.get("fill")
        if fill is not None and not _is_neutraal(fill):
            rep_kleur = _rnd(fill)

        if rep_kleur is None:
            kleur = d.get("color")
            breedte = d.get("width") or 0
            if kleur is not None and not _is_neutraal(kleur) and breedte > 0.15:
                rep_kleur = _rnd(kleur)

        if rep_kleur is None:
            continue

        cx = (rect[0] + rect[2]) / 2
        cy = (rect[1] + rect[3]) / 2
        dx, dy = normalize(cx, cy)

        if not in_legenda_zone(dx, dy):
            continue

        gekleurde.append({"rgb": rep_kleur, "dx": dx, "dy": dy})

    if len(gekleurde) < 2:
        return {}

    # --- Stap 2: verzamel tekst-spans binnen de legenda-zone ----------------
    _skip = re.compile(
        r"^[\d\s\.,:\-/°%]+$"              # puur cijfers/eenheden
        r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"  # datum
        r"|^[A-Z]{1,4}\d*$"                # kortecodes "A1", "BP00"
        r"|^merk\s"                         # "merk DC", "merk DD" etc.
    )

    teksten: list[dict] = []
    for b in page.get_text("rawdict")["blocks"]:
        if b.get("type") != 0:
            continue
        for lijn in b.get("lines", []):
            for span in lijn.get("spans", []):
                txt = "".join(c["c"] for c in span.get("chars", [])).strip()
                if len(txt) < _MIN_LABEL_LENGTE:
                    continue
                if _skip.match(txt):
                    continue
                bx0, by0 = span["bbox"][0], span["bbox"][1]
                dx, dy = normalize(bx0, by0)
                if not in_legenda_zone(dx, dy):
                    continue
                teksten.append({"tekst": txt, "dx": dx, "dy": dy})

    if not teksten:
        return {}

    # --- Stap 3: koppel elk gekleurd element aan het dichtstbijzijnde label -
    kandidaten: list[dict] = []

    for el in gekleurde:
        beste_label: str | None = None
        beste_score = float("inf")

        for t in teksten:
            x_off = t["dx"] - el["dx"]
            y_diff = abs(t["dy"] - el["dy"])

            if not (_X_ZOEKBEREIK[0] < x_off < _X_ZOEKBEREIK[1]):
                continue
            if y_diff > _Y_TOLERANTIE:
                continue

            score = x_off ** 2 + (y_diff * 4) ** 2
            if score < beste_score:
                beste_score = score
                beste_label = t["tekst"]

        if beste_label and _WANDTYPE_TERMEN.search(beste_label):
            kandidaten.append({
                "rgb": el["rgb"],
                "label": beste_label,
                "dx": el["dx"],
                "dy": el["dy"],
            })

    if len(kandidaten) < 2:
        return {}

    # --- Stap 4: bouw mapping kleur → label ---------------------------------
    # Per kleur: neem het eerste label (dichtstbijzijnde swatch, laagste score).
    # Kandidaten zijn al gefilterd op wandtype-termen.
    mapping: dict[tuple, str] = {}
    for item in kandidaten:
        if item["rgb"] not in mapping:
            mapping[item["rgb"]] = item["label"]

    return mapping

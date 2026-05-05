"""
Tekening-profiel detectie: oriëntatie, schaal en legenda.

Elke tekening leert het systeem over zichzelf via deze drie functies
voordat er iets vergeleken wordt. Geen hardcoded aannames over positie,
kleur of schaal.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Callable

import fitz


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
    r"buitenwand|draagwand|spouwwand|brandwand|staalstud|wand)\b",
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


# ---------------------------------------------------------------------------
# Functie 3b: Vision-gebaseerde legenda (fallback)
# ---------------------------------------------------------------------------

_KLEUR_VECTOREN = {
    "rood":       (0.85, 0.10, 0.10),
    "groen":      (0.10, 0.75, 0.10),
    "blauw":      (0.10, 0.10, 0.85),
    "geel":       (0.90, 0.90, 0.05),
    "oranje":     (0.90, 0.45, 0.05),
    "paars":      (0.55, 0.05, 0.75),
    "lila":       (0.65, 0.25, 0.80),
    "violet":     (0.55, 0.05, 0.75),
    "teal":       (0.05, 0.65, 0.65),
    "cyaan":      (0.05, 0.75, 0.80),
    "turquoise":  (0.05, 0.70, 0.65),
    "roze":       (0.90, 0.40, 0.65),
    "bruin":      (0.50, 0.25, 0.05),
    "oker":       (0.75, 0.55, 0.05),
    "magenta":    (0.80, 0.05, 0.60),
    "maroon":     (0.50, 0.15, 0.15),
    "pink":       (0.90, 0.40, 0.65),
    "purple":     (0.55, 0.05, 0.75),
    "green":      (0.10, 0.75, 0.10),
    "blue":       (0.10, 0.10, 0.85),
    "red":        (0.85, 0.10, 0.10),
    "orange":     (0.90, 0.45, 0.05),
    "yellow":     (0.90, 0.90, 0.05),
    "brown":      (0.50, 0.25, 0.05),
}

_VISION_PROMPT = (
    "Dit is de legenda/renvooi van een bouwtekening voor K&K Afbouw.\n"
    "K&K Afbouw plaatst gibowanden, voorzetwanden en plafonds.\n\n"
    "Geef ALLE wandtypes die je in deze legenda ziet.\n"
    "Per wandtype geef:\n"
    "- naam (exact zoals het in de legenda staat)\n"
    "- kleur beschrijving\n"
    "- arcering patroon (enkele diagonaal, kruisarcering, horizontaal, "
    "verticaal, zigzag, gevuld, geen)\n"
    "- relevant_voor_kk: true als het een wandtype is dat K&K plaatst "
    "of waar K&K rekening mee moet houden\n\n"
    "Relevante wandtypes: gibo, voorzetwand, isolatie, kalkzandsteen, "
    "sandwichpaneel, beton, prefab, hsb-wand, PIR+OSB, hardschuimisolatie, "
    "achterwand toilet, Rhombus/Mato gevelafwerking, binnenwand, scheidingswand\n"
    "NIET relevant: brandblussers, vluchtwegaanduidingen, peilmaten, "
    "deursymbolen, overstroomroosters\n\n"
    "Retourneer ALLEEN een JSON array, geen andere tekst:\n"
    '[{"naam": "gibo binnenwand 70mm", "kleur": "blauw-lila", '
    '"arcering": "verticaal", "relevant_voor_kk": true}]'
)


def _display_naar_raw_clip(
    dx0: float, dy0: float, dx1: float, dy1: float,
    rotation: int, display_width: float, display_height: float,
) -> fitz.Rect:
    """Converteer display-crop (display-coords) naar raw PDF Rect voor get_pixmap."""
    dw, dh = display_width, display_height
    if rotation == 0:
        return fitz.Rect(dx0, dy0, dx1, dy1)
    elif rotation == 90:
        # normalize: disp_x = dw - raw_y, disp_y = raw_x
        # inverse:   raw_x = disp_y,       raw_y = dw - disp_x
        return fitz.Rect(dy0, dw - dx1, dy1, dw - dx0)
    elif rotation == 180:
        # normalize: disp_x = dw - raw_x, disp_y = dh - raw_y
        # inverse:   raw_x = dw - disp_x, raw_y = dh - disp_y
        return fitz.Rect(dw - dx1, dh - dy1, dw - dx0, dh - dy0)
    elif rotation == 270:
        # normalize: disp_x = raw_y,       disp_y = dh - raw_x
        # inverse:   raw_x = dh - disp_y,  raw_y = disp_x
        return fitz.Rect(dh - dy1, dx0, dh - dy0, dx1)
    else:
        return fitz.Rect(dx0, dy0, dx1, dy1)


def vind_legenda_vision(page, ori: dict, api_key: str | None = None) -> dict:
    """
    Vision-gebaseerde legenda-lezer als fallback voor vind_legenda.

    Rendert een 600×800pt crop van de legenda-zone en vraagt Claude Vision
    om alle wandtypes te identificeren. Matcht kleur-beschrijvingen op de
    RGB-waarden uit de vector-data.

    Returns:
        dict: rgb_tuple -> {"naam": str, "relevant_voor_kk": bool, "arcering": str}
        Lege dict als Vision niet beschikbaar is of niets vindt.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {}

    normalize: Callable = ori["normalize"]
    rotation = ori["rotation"]
    dw = ori["display_width"]
    dh = ori["display_height"]

    # Stap 1: Vind legenda-titel posities (display coords)
    titel_posities = _vind_legenda_titels(page, normalize)
    if not titel_posities:
        return {}

    # Stap 2: Crop-zone — bepaal of de legenda links of rechts van de titel staat.
    #   Standaard: 600×800pt naar rechts/beneden (titel linksboven in legenda).
    #   Flip naar links als de titel dicht bij de rechter paginakant staat — de legenda
    #   strekt zich dan naar links uit (zoals Muiden D2.1: titel op x=3509, pagina 3827pt).
    #   Bekende beperking: bij tekeningen met meerdere afzonderlijke legendas op dezelfde
    #   pagina kan de gekozen crop een verkeerde legenda includeren. Niet opgelost vandaag.
    tx, ty = titel_posities[0]
    _STANDARD_W = 600
    _standard_x1 = min(dw, tx - 50 + _STANDARD_W)
    _clipped_w = _standard_x1 - max(0.0, tx - 50)

    if _clipped_w < _STANDARD_W * 0.7:
        # Titel staat dicht bij de rechter paginakant — legenda strekt zich naar links.
        # Gebruik 600pt hoogte om de wandtype-sectie te tonen zonder de deur/kozijn-tabellen
        # die lager op de pagina staan en Vision afleiden (Muiden D2.1 patroon).
        crop_x1 = min(dw, tx + 200)
        crop_x0 = max(0.0, crop_x1 - 1000)
        crop_y0 = max(0.0, min(t[1] for t in titel_posities) - 30)
        crop_y1 = min(dh, crop_y0 + 600)
    else:
        crop_x0 = max(0.0, tx - 50)
        crop_y0 = max(0.0, ty - 30)
        crop_x1 = _standard_x1
        crop_y1 = min(dh, crop_y0 + 800)

    # Stap 3: Clip aan paginagrens.
    #   get_pixmap(clip=...) verwacht display-coords (page.rect-ruimte) — NIET de raw
    #   pre-rotatie PDF-coords die _display_naar_raw_clip zou teruggeven.
    raw_clip = fitz.Rect(crop_x0, crop_y0, crop_x1, crop_y1) & page.rect

    if raw_clip.is_empty:
        return {}

    # Stap 4: Render als PNG (200 DPI)
    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat, clip=raw_clip, colorspace=fitz.csRGB)
    png_bytes = pix.tobytes("png")

    # Stap 5: Stuur naar Claude Vision API
    try:
        import anthropic
    except ImportError:
        print("[Vision] anthropic package niet geïnstalleerd — pip install anthropic")
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    b64_image = base64.b64encode(png_bytes).decode()

    try:
        _t0 = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
        )
        _t1 = time.time()
        usage = response.usage
        # Sonnet 4.6 prijzen mei 2026: $3/MTok input, $15/MTok output
        _cost = (usage.input_tokens * 3 + usage.output_tokens * 15) / 1_000_000
        print(
            f"[Vision] {usage.input_tokens} in + {usage.output_tokens} out tokens"
            f" = ${_cost:.4f}  ({_t1 - _t0:.1f}s)"
        )
        vision_tekst = response.content[0].text.strip()
    except Exception as e:
        print(f"[Vision] API fout: {e}")
        return {}

    # Stap 6: Parse JSON response
    try:
        start = vision_tekst.find("[")
        end = vision_tekst.rfind("]") + 1
        if start < 0 or end <= start:
            print(f"[Vision] Geen JSON array in response: {vision_tekst[:200]}")
            return {}
        vision_items: list[dict] = json.loads(vision_tekst[start:end])
    except json.JSONDecodeError as e:
        print(f"[Vision] JSON parse fout: {e} — response: {vision_tekst[:200]}")
        return {}

    # Stap 7: Verzamel niet-neutrale kleuren uit de legenda-zone (vector data)
    zone_kleuren: list[tuple] = []
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
        if crop_x0 <= dx <= crop_x1 and crop_y0 <= dy <= crop_y1:
            if rep_kleur not in zone_kleuren:
                zone_kleuren.append(rep_kleur)

    # Stap 8: Match Vision kleur-beschrijving → dichtstbijzijnde RGB
    def _match_kleur(desc: str) -> tuple | None:
        desc_l = desc.lower()
        doel = [0.0, 0.0, 0.0]
        gewicht = 0.0
        for naam, vec in _KLEUR_VECTOREN.items():
            if naam in desc_l:
                doel = [doel[i] + vec[i] for i in range(3)]
                gewicht += 1.0
        if gewicht == 0 or not zone_kleuren:
            return None
        doel_t = tuple(v / gewicht for v in doel)
        return min(zone_kleuren,
                   key=lambda kr: sum((a - b) ** 2 for a, b in zip(kr, doel_t)))

    # Stap 9: Bouw resultaat dict
    resultaat: dict = {}
    gebruikt: set = set()
    _fallback_idx = 0
    for item in vision_items:
        naam = item.get("naam", "").strip()
        if not naam:
            continue
        rgb = _match_kleur(item.get("kleur", ""))
        if rgb is None:
            # Kleur niet herkenbaar (bijv. "zwart/wit") → index-fallback als 3-tuple.
            # Veiligheid: afstand tot elke echte RGB >= sqrt(2) ~= 1.41, ver boven
            # huidige _lookup_wandtype drempel 0.15. Als die drempel ooit > 1.4 wordt,
            # breekt deze aanname.
            key = (-1, _fallback_idx, -1)
            _fallback_idx += 1
        elif rgb in gebruikt:
            continue
        else:
            key = rgb
        resultaat[key] = {
            "naam": naam,
            "relevant_voor_kk": bool(item.get("relevant_voor_kk", False)),
            "arcering": item.get("arcering", "geen"),
        }
        gebruikt.add(key)

    return resultaat


def vind_legenda_combined(page, ori: dict, api_key: str | None = None) -> dict:
    """
    Combineert vector-legenda (vind_legenda) met Vision-legenda als fallback.

    Selectielogica:
    - Geen api_key: altijd vector.
    - Vector < 2 wandtypes: Vision.
    - Vector >= 2 maar > 60% heeft hetzelfde label (uniform/weinig discriminerend): Vision.
    - Vector >= 2 en voldoende variatie: vector.

    Returns:
        dict: rgb_tuple -> str (wandtype-naam) — zelfde formaat als vind_legenda.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    vector = vind_legenda(page, ori)
    print(f"[Legenda] Vector: {len(vector)} wandtypes — {list(vector.values())}")

    if not api_key:
        if len(vector) >= 2:
            print(f"[Legenda] Gecombineerd: vector-primair ({len(vector)} wandtypes, geen API key)")
        return vector

    # Kwaliteitscheck: meer dan 60% van de labels identiek → vector is weinig discriminerend.
    _vector_uniform = False
    if len(vector) >= 2:
        _labels = list(vector.values())
        _meest = max(set(_labels), key=_labels.count)
        _aandeel = _labels.count(_meest) / len(_labels)
        if _aandeel > 0.60:
            _vector_uniform = True
            print(
                f"[Legenda] Vector uniform: '{_meest}' = "
                f"{_labels.count(_meest)}/{len(_labels)} ({_aandeel:.0%}) — Vision-fallback"
            )

    if len(vector) >= 2 and not _vector_uniform:
        print(f"[Legenda] Gecombineerd: vector-primair ({len(vector)} wandtypes, Vision overgeslagen)")
        return vector

    # Vision als fallback (vector < 2 of uniform)
    vision_raw = vind_legenda_vision(page, ori, api_key)
    vision = {rgb: info["naam"] for rgb, info in vision_raw.items()}
    print(f"[Legenda] Vision: {len(vision)} wandtypes — {list(vision.values())}")
    if _vector_uniform:
        print("[Legenda] Gecombineerd: Vision-primair (vector uniform — te weinig variatie)")
    else:
        print("[Legenda] Gecombineerd: Vision-primair (vector < 2 resultaten)")
    return vision


# ---------------------------------------------------------------------------
# Functie 4: wandvergelijking
# ---------------------------------------------------------------------------

_MATCH_TOLERANTIE_PT = 10   # display-pt — drempel voor "zelfde pad"
_MATCH_GRID = 5             # pt — rastercellgrootte voor snelle lookup


def vergelijk_wanden(oud_path: str, nieuw_path: str, pagina: int = 0) -> list[dict]:
    """
    Vergelijkt wandpaden tussen twee revisies van dezelfde tekening-pagina.

    1. Detecteert oriëntatie en legenda op beide pagina's.
    2. Filtert dikke gekleurde paden (width >= 0.5pt) per wandtype-kleur.
    3. Vergelijkt via grid-index (tolerantie _MATCH_TOLERANTIE_PT pt):
       - Paden in NIEUW zonder match in OUD  → "nieuw"
       - Paden in OUD zonder match in NIEUW  → "verdwenen"

    Returns:
        Lijst van dicts: type, wandtype, kleur, positie en bbox (display-coords).
    """

    def _open(pad, pnr):
        doc = fitz.open(pad)
        page = doc[pnr]
        ori = detecteer_orientatie(page)
        leg = vind_legenda(page, ori)
        return doc, page, ori, leg

    oud_doc, oud_page, oud_ori, oud_leg = _open(oud_path, pagina)
    nieuw_doc, nieuw_page, nieuw_ori, nieuw_leg = _open(nieuw_path, pagina)

    # Gecombineerde legenda: nieuw heeft prioriteit bij conflicten
    legenda = {**oud_leg, **nieuw_leg}

    if not legenda:
        oud_doc.close()
        nieuw_doc.close()
        return []

    kleuren = set(legenda.keys())

    def _haal_paden(page, ori) -> list[dict]:
        normalize = ori["normalize"]
        paden = []
        for d in page.get_drawings():
            # Fill-gebaseerde wanden (bijv. 56 de Helling): gekleurde gevulde vlakken
            fill = d.get("fill")
            rep_kr: tuple | None = None
            if fill is not None and not _is_neutraal(fill):
                kr_f = _rnd(fill)
                if kr_f in kleuren:
                    rep_kr = kr_f

            # Stroke-gebaseerde wanden (bijv. Muiden): gekleurde arceringlijnen
            if rep_kr is None:
                kleur = d.get("color")
                breedte = d.get("width") or 0
                if kleur is not None and breedte >= 0.5:
                    kr_s = _rnd(kleur)
                    if kr_s in kleuren:
                        rep_kr = kr_s

            if rep_kr is None:
                continue

            rect = d.get("rect")
            if rect is None:
                continue
            if max(rect[2] - rect[0], rect[3] - rect[1]) < 2:
                continue  # stip/punt
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            dx, dy = normalize(cx, cy)
            c1 = normalize(rect[0], rect[1])
            c2 = normalize(rect[2], rect[3])
            paden.append({
                "kr": rep_kr,
                "dx": dx,
                "dy": dy,
                "bbox": [min(c1[0], c2[0]), min(c1[1], c2[1]),
                         max(c1[0], c2[0]), max(c1[1], c2[1])],
            })
        return paden

    oud_paden = _haal_paden(oud_page, oud_ori)
    nieuw_paden = _haal_paden(nieuw_page, nieuw_ori)
    oud_doc.close()
    nieuw_doc.close()

    def _bouw_index(paden) -> set:
        g = _MATCH_GRID
        return {(p["kr"], int(p["dx"] // g), int(p["dy"] // g)) for p in paden}

    def _heeft_match(pad, idx) -> bool:
        kr = pad["kr"]
        gx = int(pad["dx"] // _MATCH_GRID)
        gy = int(pad["dy"] // _MATCH_GRID)
        cellen = _MATCH_TOLERANTIE_PT // _MATCH_GRID + 1
        return any(
            (kr, gx + ddx, gy + ddy) in idx
            for ddx in range(-cellen, cellen + 1)
            for ddy in range(-cellen, cellen + 1)
        )

    oud_idx = _bouw_index(oud_paden)
    nieuw_idx = _bouw_index(nieuw_paden)

    resultaten: list[dict] = []

    for pad in nieuw_paden:
        if not _heeft_match(pad, oud_idx):
            resultaten.append({
                "type": "nieuw",
                "wandtype": legenda[pad["kr"]],
                "kleur": pad["kr"],
                "positie": [round(pad["dx"], 1), round(pad["dy"], 1)],
                "bbox": [round(v, 1) for v in pad["bbox"]],
            })

    for pad in oud_paden:
        if not _heeft_match(pad, nieuw_idx):
            resultaten.append({
                "type": "verdwenen",
                "wandtype": legenda[pad["kr"]],
                "kleur": pad["kr"],
                "positie": [round(pad["dx"], 1), round(pad["dy"], 1)],
                "bbox": [round(v, 1) for v in pad["bbox"]],
            })

    return resultaten

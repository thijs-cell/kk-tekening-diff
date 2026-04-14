"""
Overlay generator voor K&K wijzigingsrapport.

Genereert een PDF met:
  - A4 samenvatting met 5 secties
  - Tekening met gekleurde vakjes/pijlen + nummers

Kleuren:
  Oranje = maatwijzigingen (vakje + nummer)
  Groen  = oppervlaktewijzigingen (vakje + nummer)
  Blauw  = ruimtenaamwijzigingen (vakje + nummer)
  Rood   = wandwijziging verwijderd (pijl + nummer)
  Paars  = wandwijziging toegevoegd/maat (pijl + nummer)
"""

import logging
import re
from datetime import datetime

import fitz

from .diff_engine import strip_annotations
from .interpreter import _zoek_context as _zoek_context_base
from .layout_detect import PageLayout

logger = logging.getLogger(__name__)


def _zoek_locatie(pos: list | tuple, alle_tekst: list[dict],
                  eigen_tekst: str = "") -> str:
    """Zoek dichtstbijzijnde ruimtenaam. Breder bereik dan interpreter."""
    # Eerst proberen met standaard bereik
    ctx = _zoek_context_base(pos, alle_tekst, max_afstand=120.0,
                             eigen_tekst=eigen_tekst)
    if ctx.get("ruimte"):
        return ctx["ruimte"]

    # Breder zoeken (250pt)
    ctx = _zoek_context_base(pos, alle_tekst, max_afstand=250.0,
                             eigen_tekst=eigen_tekst)
    if ctx.get("ruimte"):
        return ctx["ruimte"]

    # Fallback: dichtstbijzijnde tekst die langer is dan 3 chars
    beste_d = float("inf")
    beste = ""
    for t in alle_tekst:
        tekst = t.get("tekst", "")
        if len(tekst) < 3 or tekst == eigen_tekst:
            continue
        # Skip losse getallen
        if re.match(r"^\d[\d\s.,]*$", tekst):
            continue
        d = ((pos[0] - t["pos"][0]) ** 2 + (pos[1] - t["pos"][1]) ** 2) ** 0.5
        if d < beste_d and d < 400:
            beste_d = d
            beste = tekst
    return beste

# ---------------------------------------------------------------------------
# Stijl
# ---------------------------------------------------------------------------

ORANJE = {"color": (0.9, 0.5, 0.0), "fill": (1.0, 0.7, 0.0), "fill_opacity": 0.18, "width": 1.0}
ROOD = {"color": (0.8, 0.0, 0.0), "fill": (1.0, 0.0, 0.0), "fill_opacity": 0.18, "width": 1.0}
GROEN = {"color": (0.0, 0.6, 0.0), "fill": (0.0, 0.8, 0.0), "fill_opacity": 0.18, "width": 1.0}
PAARS = {"color": (0.5, 0.0, 0.7), "fill": (0.6, 0.0, 0.8), "fill_opacity": 0.18, "width": 1.0}
BLAUW = {"color": (0.1, 0.3, 0.8), "fill": (0.2, 0.4, 0.9), "fill_opacity": 0.18, "width": 1.0}


PADDING = 3
NUMMER_FONTSIZE = 6

# Revisie-ruis regex
_RE_REVISIE_RUIS = re.compile(
    r"^[A-Z]$"
    r"|^\d{2}-\d{2}-\d{4}$"
    r"|^\d{1,2}-\d{1,2}-'\d{2}$"
    r"|^NDO$"
    r"|^.{0,1}$"
    r"|^\d{1,2}e uitgave$",
    re.IGNORECASE,
)

# Ruimtenaam regex — alleen echte ruimtenamen, geen codes zoals "2.At.B05"
_RE_RUIMTE = re.compile(
    r"(keuken|badkamer|toilet|hal(?:letje)?|gang|slaapkamer|woonkamer|berging|"
    r"technische ruimte|techn\.?\s*ruimte|meterkast|meterruimte|bijkeuken|entree|overloop|"
    r"balkon|terras|tuin|garage|wasruimte|cv[- ]?ruimte|hydrofoor|"
    r"buitenberging|stookruimte|trappenhuis|werkruimte|lift|woonkamer|"
    r"kinderkamer|studeerkamer|kantoor|opslagruimte|fietsenstalling|"
    r"gemeenschappelijke ruimte|recreatieruimte|speelkamer)",
    re.IGNORECASE,
)

# Puur-maat regex: losse getallen die al door maatwijzigingen gedekt worden
_RE_PUUR_MAAT = re.compile(
    r"^\d[\d\s.,]*(?:\s*(?:mm|cm|m)\b)?(?:\s*\+\s*vl)?$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_revisie_ruis(tekst: str) -> bool:
    return _RE_REVISIE_RUIS.match(tekst.strip()) is not None


def _is_maat_of_ruimte(tekst: str) -> bool:
    """True als tekst puur een maatgetal of ruimtenaam is (al gedekt door andere secties)."""
    t = tekst.strip()
    if _RE_PUUR_MAAT.match(t):
        return True
    if _RE_RUIMTE.search(t) and len(t) < 30:
        return True
    return False


def _in_excluded_zone(pos: list | tuple, pw: float, ph: float,
                      layout: PageLayout | None = None) -> bool:
    """Check of een punt in een uitgesloten zone valt.

    Gebruikt layout auto-detectie als beschikbaar, anders fallback
    op hardcoded ratio's (0.88 breedte, 0.05 hoogte).
    """
    if layout is not None:
        return layout.is_excluded(pos)
    x, y = pos[0], pos[1]
    if x > pw * 0.88:
        return True
    if y < ph * 0.05:
        return True
    return False


def _bbox_to_rect(bbox: list, padding: float = PADDING) -> fitz.Rect:
    return fitz.Rect(
        bbox[0] - padding, bbox[1] - padding,
        bbox[2] + padding, bbox[3] + padding,
    )


def _rect_from_pos(x: float, y: float, w: float = 30, h: float = 10) -> fitz.Rect:
    return fitz.Rect(x - w, y - h, x + w, y + h)


def merge_overlapping_rects(rects: list[fitz.Rect], margin: float = 5) -> list[fitz.Rect]:
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged = [rects[0]]
    for rect in rects[1:]:
        last = merged[-1]
        if (rect.x0 <= last.x1 + margin and
                rect.y0 <= last.y1 + margin and
                rect.x1 >= last.x0 - margin and
                rect.y1 >= last.y0 - margin):
            merged[-1] = fitz.Rect(
                min(last.x0, rect.x0), min(last.y0, rect.y0),
                max(last.x1, rect.x1), max(last.y1, rect.y1),
            )
        else:
            merged.append(rect)
    return merged


def _kleur_verschil_significant(oud_naam: str, nieuw_naam: str) -> bool:
    if oud_naam == nieuw_naam:
        return False
    m_oud = re.match(r"grijs\((\d+)%\)", oud_naam)
    m_nieuw = re.match(r"grijs\((\d+)%\)", nieuw_naam)
    if m_oud and m_nieuw:
        if abs(int(m_oud.group(1)) - int(m_nieuw.group(1))) < 15:
            return False
    if "zwart" in oud_naam and "zwart" in nieuw_naam:
        return False
    oud_base = oud_naam.split("(")[0].strip()
    nieuw_base = nieuw_naam.split("(")[0].strip()
    if oud_base in {"zwart", "grijs"} and nieuw_base in {"zwart", "grijs"}:
        return False
    return True


# ---------------------------------------------------------------------------
# Collect functies — bouwen de 6 lijsten op
# ---------------------------------------------------------------------------

def _collect_maat(diff_result: dict, pw: float, ph: float, alle_tekst: list,
                  layout: PageLayout | None = None):
    """Maatwijzigingen: gewijzigde mm-waarden."""
    items = []
    renvooi = 0
    for item in diff_result.get("tekst_gewijzigd", []):
        if item.get("categorie") != "maat":
            continue
        pos = item.get("nieuw_pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        oud = item.get("oud_tekst", "")
        nieuw = item.get("nieuw_tekst", "")
        bbox = item.get("nieuw_bbox")
        rect = _bbox_to_rect(bbox) if bbox else _rect_from_pos(*pos)
        locatie = _zoek_locatie(pos, alle_tekst, eigen_tekst=nieuw)
        beschr = f"van {oud}mm naar {nieuw}mm"
        if locatie:
            beschr += f" (bij {locatie})"
        items.append({
            "rect": rect,
            "beschrijving": beschr,
        })
    return items, renvooi


_RE_MAAT_WAARDE = re.compile(r"^[\d.,]+$")
_VERSCHOVEN_STRAAL = 80.0  # pt — zelfde maat binnen deze straal = verschoven, niet nieuw


def _collect_nieuwe_maten(diff_result: dict, pw: float, ph: float, alle_tekst: list,
                          layout: PageLayout | None = None):
    """Nieuw toegevoegde maatgetallen — stonden niet in de oude tekening.

    Paars vakje: geeft aan dat er iets nieuws is bijgekomen op die plek.

    Filter 1: puur numeriek, waarde >= 50 (geen kamer/verdiepingsnummers).
    Filter 2: als dezelfde waarde in tekst_verdwenen zit binnen _VERSCHOVEN_STRAAL,
              is het dezelfde maat die iets verschoof — geen echte nieuwe maat.
    """
    import math as _math

    verdwenen_maten: dict[str, list] = {}
    for item in diff_result.get("tekst_verdwenen", []):
        t = item.get("tekst", "").strip()
        if _RE_MAAT_WAARDE.match(t):
            verdwenen_maten.setdefault(t, []).append(item.get("pos", [0, 0]))

    items = []
    renvooi = 0
    for item in diff_result.get("tekst_toegevoegd", []):
        tekst = item.get("tekst", "").strip()
        if not _RE_MAAT_WAARDE.match(tekst):
            continue
        try:
            waarde = float(tekst.replace(",", "."))
        except ValueError:
            continue
        if waarde < 50:
            continue
        pos = item.get("pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        verschoven = False
        for vpos in verdwenen_maten.get(tekst, []):
            d = _math.hypot(pos[0] - vpos[0], pos[1] - vpos[1])
            if d < _VERSCHOVEN_STRAAL:
                verschoven = True
                break
        if verschoven:
            continue
        bbox = item.get("bbox")
        rect = _bbox_to_rect(bbox) if bbox else _rect_from_pos(*pos)
        locatie = _zoek_locatie(pos, alle_tekst, eigen_tekst=tekst)
        beschr = f"Nieuw: {tekst}mm"
        if locatie:
            beschr += f" (bij {locatie})"
        items.append({"rect": rect, "beschrijving": beschr})
    return items, renvooi


def _collect_oppervlakte(diff_result: dict, pw: float, ph: float, alle_tekst: list,
                         layout: PageLayout | None = None):
    """Oppervlaktewijzigingen: gewijzigde m² waarden."""
    items = []
    renvooi = 0
    re_opp = re.compile(r"\d+[.,]\d+\s*m")
    for item in diff_result.get("tekst_gewijzigd", []):
        if item.get("categorie") != "oppervlakte":
            continue
        pos = item.get("nieuw_pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        oud = item.get("oud_tekst", "")
        nieuw = item.get("nieuw_tekst", "")
        if not (re_opp.search(oud) and re_opp.search(nieuw)):
            continue
        bbox = item.get("nieuw_bbox")
        rect = _bbox_to_rect(bbox) if bbox else _rect_from_pos(*pos)
        locatie = _zoek_locatie(pos, alle_tekst, eigen_tekst=nieuw)
        beschr = f"van {oud} naar {nieuw}"
        if locatie:
            beschr += f" (bij {locatie})"
        items.append({
            "rect": rect,
            "beschrijving": beschr,
        })
    return items, renvooi


def _collect_ruimtenaam(diff_result: dict, pw: float, ph: float,
                        layout: PageLayout | None = None):
    """Ruimtenaam wijzigingen: ruimtelabel veranderd."""
    items = []
    renvooi = 0

    # Directe wijzigingen: items waar minstens een van beide een ruimtenaam bevat
    gezien = set()
    for item in diff_result.get("tekst_gewijzigd", []):
        pos = item.get("nieuw_pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        oud = item.get("oud_tekst", "")
        nieuw = item.get("nieuw_tekst", "")
        # Minstens een van beide moet een ruimtenaam zijn
        if not (_RE_RUIMTE.search(oud) or _RE_RUIMTE.search(nieuw)):
            continue
        # Skip als het eigenlijk een oppervlakte of maat is
        if item.get("categorie") in ("maat", "oppervlakte"):
            continue
        # Dedup
        key = (oud.lower(), nieuw.lower())
        if key in gezien:
            continue
        gezien.add(key)
        bbox = item.get("nieuw_bbox")
        rect = _bbox_to_rect(bbox) if bbox else _rect_from_pos(*pos)
        items.append({
            "rect": rect,
            "beschrijving": f"{oud} \u2192 {nieuw}",
        })

    # Koppel verdwenen + toegevoegde ruimtenamen in de buurt
    verdwenen = [
        i for i in diff_result.get("tekst_verdwenen", [])
        if _RE_RUIMTE.search(i.get("tekst", ""))
    ]
    toegevoegd = [
        i for i in diff_result.get("tekst_toegevoegd", [])
        if _RE_RUIMTE.search(i.get("tekst", ""))
    ]
    matched_t = set()
    for vl in verdwenen:
        vpos = vl.get("pos", [0, 0])
        if _in_excluded_zone(vpos, pw, ph, layout):
            continue
        for ti, tl in enumerate(toegevoegd):
            if ti in matched_t:
                continue
            tpos = tl.get("pos", [0, 0])
            d = ((vpos[0] - tpos[0]) ** 2 + (vpos[1] - tpos[1]) ** 2) ** 0.5
            if d < 100:
                vtekst = vl.get("tekst", "")
                ttekst = tl.get("tekst", "")
                if vtekst.lower() != ttekst.lower():
                    bbox = tl.get("bbox")
                    rect = _bbox_to_rect(bbox) if bbox else _rect_from_pos(*tpos)
                    items.append({
                        "rect": rect,
                        "beschrijving": f"{vtekst} \u2192 {ttekst}",
                    })
                    matched_t.add(ti)
                    break

    return items, renvooi




def _collect_nieuwe_wanden(diff_result: dict, pw: float, ph: float, alle_tekst: list,
                           layout: PageLayout | None = None):
    """Nieuwe wanden — alleen als er geen verdwenen wand op dezelfde plek staat.

    Als er wel een verdwenen wand dichtbij is, wordt die combinatie afgehandeld
    door _collect_verdwenen_wanden als wandwijziging.
    """
    items = []
    renvooi = 0

    # Centra van verdwenen wanden voor cross-ref
    verdwenen_centra = [
        w.get("center", [0, 0]) for w in diff_result.get("verdwenen_wanden", [])
    ]

    for wand in diff_result.get("nieuwe_wanden", []):
        center = wand.get("center", [0, 0])
        if _in_excluded_zone(center, pw, ph, layout):
            renvooi += 1
            continue

        # Skip als er een verdwenen wand dichtbij staat (wordt als wijziging gemeld)
        heeft_verdwenen_match = False
        for vc in verdwenen_centra:
            d = ((center[0] - vc[0]) ** 2 + (center[1] - vc[1]) ** 2) ** 0.5
            if d < 30:
                heeft_verdwenen_match = True
                break
        if heeft_verdwenen_match:
            continue

        bbox = wand.get("bbox", [0, 0, 0, 0])
        breedte = bbox[2] - bbox[0]
        hoogte = bbox[3] - bbox[1]
        radius = min(max(breedte, hoogte) / 2 + 1, 8)
        radius = max(radius, 5)
        locatie = _zoek_locatie(center, alle_tekst)
        wandtype = wand.get("wandtype")
        beschr = f"Wand toegevoegd: {wandtype}" if wandtype else "Wand toegevoegd"
        if locatie:
            beschr += f" (bij {locatie})"
        items.append({
            "rect": fitz.Rect(center[0] - radius, center[1] - radius,
                              center[0] + radius, center[1] + radius),
            "center": center,
            "radius": radius,
            "beschrijving": beschr,
        })
    return items, renvooi


def _collect_verdwenen_wanden(diff_result: dict, pw: float, ph: float, alle_tekst: list,
                              layout: PageLayout | None = None):
    """Verdwenen wanden — alleen als er geen nieuwe wand op dezelfde plek staat.

    Als er wel een nieuwe wand dichtbij is, wordt het een wandwijziging
    (was X, is nu Y) in plaats van een verwijdering.
    """
    items = []
    wijziging_items = []  # wand op zelfde plek veranderd van type
    renvooi = 0

    # Bouw lijst van nieuwe-wand-centra voor cross-referencing
    nieuwe_centra = []
    for nw in diff_result.get("nieuwe_wanden", []):
        nieuwe_centra.append(nw)

    for wand in diff_result.get("verdwenen_wanden", []):
        center = wand.get("center", [0, 0])
        if _in_excluded_zone(center, pw, ph, layout):
            renvooi += 1
            continue
        bbox = wand.get("bbox", [0, 0, 0, 0])
        breedte = bbox[2] - bbox[0]
        hoogte = bbox[3] - bbox[1]
        radius = min(max(breedte, hoogte) / 2 + 1, 8)
        radius = max(radius, 5)
        locatie = _zoek_locatie(center, alle_tekst)
        oud_wt = wand.get("wandtype")

        # Check of er een nieuwe wand dichtbij staat (< 30pt)
        nieuw_match = None
        for nw in nieuwe_centra:
            nc = nw.get("center", [0, 0])
            d = ((center[0] - nc[0]) ** 2 + (center[1] - nc[1]) ** 2) ** 0.5
            if d < 30:
                nieuw_match = nw
                break

        rect = fitz.Rect(center[0] - radius, center[1] - radius,
                         center[0] + radius, center[1] + radius)

        if nieuw_match:
            nieuw_wt = nieuw_match.get("wandtype")
            if oud_wt and nieuw_wt and oud_wt != nieuw_wt:
                beschr = f"Was: {oud_wt} \u2192 Nu: {nieuw_wt}"
            elif oud_wt and nieuw_wt and oud_wt == nieuw_wt:
                continue  # zelfde type, geen echte wijziging
            else:
                beschr = f"Was: {oud_wt or 'onbekend'} \u2192 Nu: {nieuw_wt or 'onbekend'}"
            if locatie:
                beschr += f" (bij {locatie})"
            wijziging_items.append({
                "rect": rect, "center": center,
                "radius": radius, "beschrijving": beschr,
            })
        else:
            beschr = f"Wand verwijderd: {oud_wt}" if oud_wt else "Wand verwijderd"
            if locatie:
                beschr += f" (bij {locatie})"
            items.append({
                "rect": rect, "center": center,
                "radius": radius, "beschrijving": beschr,
            })

    return items, wijziging_items, renvooi


def _collect_wanden_per_type(
    oud_path: str, nieuw_path: str, pagina: int,
    pw: float, ph: float, layout: PageLayout | None = None,
):
    """Verzamel wandwijzigingen via compare_per_wandtype (kleur-gebaseerd).

    Returns: (gewijzigd_items, verwijderd_items, toegevoegd_items, renvooi_count)
    """
    from .diff_engine import compare_per_wandtype

    RADIUS = 8.0
    gewijzigd, verwijderd, toegevoegd = [], [], []
    renvooi = 0

    try:
        wijzigingen = compare_per_wandtype(oud_path, nieuw_path, pagina)
    except Exception as e:
        logger.warning("compare_per_wandtype mislukt op pagina %d: %s", pagina, e)
        return gewijzigd, verwijderd, toegevoegd, renvooi

    for w in wijzigingen:
        pos = w["positie"]
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        cx, cy = pos[0], pos[1]
        rect = fitz.Rect(cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS)

        if w["type"] == "gewijzigd":
            beschr = f"Was: {w['wandtype_oud']} \u2192 Nu: {w['wandtype_nieuw']}"
            gewijzigd.append({"rect": rect, "center": [cx, cy], "beschrijving": beschr})
        elif w["type"] == "verwijderd":
            beschr = f"Verwijderd: {w['wandtype_oud']}"
            verwijderd.append({"rect": rect, "center": [cx, cy], "beschrijving": beschr})
        elif w["type"] == "toegevoegd":
            beschr = f"Toegevoegd: {w['wandtype_nieuw']}"
            toegevoegd.append({"rect": rect, "center": [cx, cy], "beschrijving": beschr})

    return gewijzigd, verwijderd, toegevoegd, renvooi


def _hex_to_rgb(hex_str: str) -> tuple | None:
    """Convert '#aabbcc' naar (r, g, b) floats 0-1."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return None
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _collect_wandwijzigingen(diff_result: dict, pw: float, ph: float,
                             layout: PageLayout | None = None):
    """Alle wandwijzigingen uit drie bronnen:

    1. vul_kleur_gewijzigd  — fill veranderd van kleur (type A → type B)
    2. nieuwe_gekleurde_vlakken — fill helemaal nieuw (wand toegevoegd)
    3. verdwenen_gekleurde_vlakken — fill helemaal weg (wand verwijderd)

    Bij type-wijziging (A → B): zowel verwijderd (A) als toegevoegd (B).
    """
    from .diff_engine import _lookup_wandtype

    toegevoegd_items = []
    verwijderd_items = []
    renvooi = 0
    MIN_OPP = 100

    # Legenda voor hex-lookup
    legenda_raw = diff_result.get("legenda_mapping", {})
    legenda = {}
    for k, v in legenda_raw.items():
        parts = k.split(",")
        legenda[tuple(float(x) for x in parts)] = v

    # --- Bron 1: vulkleur gewijzigd (type A → type B) ---
    for item in diff_result.get("vul_kleur_gewijzigd", []):
        pos = item.get("pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        opp = item.get("oppervlakte", 0)
        if opp < MIN_OPP:
            continue
        oud_naam = item.get("oud_naam", "")
        nieuw_naam = item.get("nieuw_naam", "")
        if not _kleur_verschil_significant(oud_naam, nieuw_naam):
            continue

        oud_wt = item.get("oud_wandtype")
        nieuw_wt = item.get("nieuw_wandtype")
        bbox = item.get("bbox", [pos[0], pos[1], pos[0] + 30, pos[1] + 10])
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

        # Oud type weg → verwijderd
        if oud_wt:
            verwijderd_items.append({
                "rect": rect,
                "beschrijving": f"Wand verwijderd: {oud_wt}",
            })
        # Nieuw type erbij → toegevoegd
        if nieuw_wt:
            toegevoegd_items.append({
                "rect": rect,
                "beschrijving": f"Wand toegevoegd: {nieuw_wt}",
            })

    # --- Bron 2: helemaal nieuwe vlakken (wand toegevoegd) ---
    for item in diff_result.get("nieuwe_gekleurde_vlakken", []):
        pos = item.get("pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        opp = item.get("oppervlakte", 0)
        if opp < MIN_OPP:
            continue
        rgb = _hex_to_rgb(item.get("kleur_hex", ""))
        if not rgb:
            continue
        wt = _lookup_wandtype(rgb, legenda)
        if not wt:
            continue
        toegevoegd_items.append({
            "rect": _rect_from_pos(pos[0], pos[1], w=20, h=10),
            "beschrijving": f"Wand toegevoegd: {wt}",
        })

    # --- Bron 3: helemaal verdwenen vlakken (wand verwijderd) ---
    for item in diff_result.get("verdwenen_gekleurde_vlakken", []):
        pos = item.get("pos", [0, 0])
        if _in_excluded_zone(pos, pw, ph, layout):
            renvooi += 1
            continue
        opp = item.get("oppervlakte", 0)
        if opp < MIN_OPP:
            continue
        rgb = _hex_to_rgb(item.get("kleur_hex", ""))
        if not rgb:
            continue
        wt = _lookup_wandtype(rgb, legenda)
        if not wt:
            continue
        verwijderd_items.append({
            "rect": _rect_from_pos(pos[0], pos[1], w=20, h=10),
            "beschrijving": f"Wand verwijderd: {wt}",
        })

    return toegevoegd_items, verwijderd_items, renvooi


# ---------------------------------------------------------------------------
# Tekenfuncties
# ---------------------------------------------------------------------------

def _teken_rect(page, rect: fitz.Rect, stijl: dict):
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(
        color=stijl["color"], fill=stijl["fill"],
        fill_opacity=stijl["fill_opacity"], width=stijl["width"],
    )
    shape.commit()


def _teken_doorstreep(page, rect: fitz.Rect, stijl: dict):
    shape = page.new_shape()
    shape.draw_line(fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x1, rect.y1))
    shape.finish(color=stijl["color"], width=stijl["width"])
    shape.commit()


def _teken_nummer(page, rect: fitz.Rect, nummer: int, kleur: tuple):
    """Klein nummer met witte achtergrond linksboven de markering."""
    tekst = str(nummer)
    tw = len(tekst) * 4 + 3
    bg = fitz.Rect(rect.x0 - 1, rect.y0 - NUMMER_FONTSIZE - 3,
                   rect.x0 + tw, rect.y0 - 1)
    shape = page.new_shape()
    shape.draw_rect(bg)
    shape.finish(color=None, fill=(1, 1, 1), fill_opacity=0.85, width=0)
    shape.commit()
    try:
        page.insert_text(
            fitz.Point(rect.x0, rect.y0 - 3),
            tekst, fontsize=NUMMER_FONTSIZE, color=kleur,
        )
    except Exception:
        pass


def _teken_pijl_met_nummer(page, rect: fitz.Rect, nummer: int,
                            kleur: tuple = (0.5, 0.0, 0.7)):
    """Pijl naar het midden van de rect + nummer label. Alleen voor paars."""
    import math

    target_x = (rect.x0 + rect.x1) / 2
    target_y = (rect.y0 + rect.y1) / 2

    # Wissel richting om overlapping te beperken — grote offsets
    richting = nummer % 4
    offsets = [(150, -70), (-160, -70), (150, 70), (-160, 70)]
    dx, dy = offsets[richting]
    start_x = target_x + dx
    start_y = target_y + dy

    pw, ph = page.rect.width, page.rect.height
    start_x = max(30, min(pw - 80, start_x))
    start_y = max(30, min(ph - 30, start_y))

    # Pijllijn — dik en duidelijk
    shape = page.new_shape()
    shape.draw_line(fitz.Point(start_x, start_y), fitz.Point(target_x, target_y))
    shape.finish(color=kleur, width=3.5)
    shape.commit()

    # Gevulde driehoek pijlpunt
    line_dx = target_x - start_x
    line_dy = target_y - start_y
    line_len = math.hypot(line_dx, line_dy)
    if line_len > 0:
        ux, uy = line_dx / line_len, line_dy / line_len
        base_x = target_x - ux * 18
        base_y = target_y - uy * 18
        # Loodrechte richting — breder
        px, py = -uy * 9, ux * 9
        shape2 = page.new_shape()
        shape2.draw_line(fitz.Point(target_x, target_y),
                         fitz.Point(base_x + px, base_y + py))
        shape2.draw_line(fitz.Point(base_x + px, base_y + py),
                         fitz.Point(base_x - px, base_y - py))
        shape2.draw_line(fitz.Point(base_x - px, base_y - py),
                         fitz.Point(target_x, target_y))
        shape2.finish(color=kleur, fill=kleur, fill_opacity=0.9, width=0.5)
        shape2.commit()

    # Nummer label — groot en leesbaar
    label = str(nummer)
    pijl_fontsize = 12
    tekst_x = start_x + 4
    tekst_y = start_y - 3
    tw = len(label) * 5 + 6
    bg = fitz.Rect(tekst_x - 3, tekst_y - 9, tekst_x + tw, tekst_y + 3)
    shape3 = page.new_shape()
    shape3.draw_rect(bg)
    shape3.finish(color=kleur, fill=(1, 1, 1), fill_opacity=0.9, width=0.5)
    shape3.commit()
    try:
        page.insert_text(fitz.Point(tekst_x, tekst_y), label,
                         fontsize=pijl_fontsize, color=kleur)
    except Exception:
        pass


def _teken_cirkel(page, center: tuple, radius: float, stijl: dict):
    """Teken een klein subtiel cirkeltje als indicator."""
    shape = page.new_shape()
    shape.draw_circle(fitz.Point(center[0], center[1]), radius)
    shape.finish(
        color=stijl["color"], fill=None,
        width=0.8,
    )
    shape.commit()


def _teken_titel_balk(page, tekst: str):
    pw = page.rect.width
    bg = fitz.Rect(0, 0, pw, 20)
    shape = page.new_shape()
    shape.draw_rect(bg)
    shape.finish(color=(0.1, 0.1, 0.1), fill=(0.1, 0.3, 0.45),
                 fill_opacity=0.85, width=0)
    shape.commit()
    page.insert_text(fitz.Point(10, 14), tekst, fontsize=9, color=(1, 1, 1))


def _teken_legenda(page, tellingen: list[tuple]):
    """Legenda linksonder met titel, groter en duidelijker.

    tellingen: [(stijl, label, count, vorm), ...]
      vorm = "rect" (default) of "cirkel" of "pijl"
    """
    pw, ph = page.rect.width, page.rect.height
    visible = []
    for entry in tellingen:
        stijl, label, count = entry[0], entry[1], entry[2]
        vorm = entry[3] if len(entry) > 3 else "rect"
        if count > 0:
            visible.append((stijl, label, count, vorm))
    if not visible:
        return

    rij_h = 22
    box_w = 300
    box_h = 28 + len(visible) * rij_h
    x0 = 15
    y0 = ph - box_h - 15

    # Achtergrond
    bg = fitz.Rect(x0, y0, x0 + box_w, y0 + box_h)
    shape = page.new_shape()
    shape.draw_rect(bg)
    shape.finish(color=(0.1, 0.1, 0.1), fill=(1, 1, 1),
                 fill_opacity=0.95, width=2.0)
    shape.commit()

    # Titel
    page.insert_text(fitz.Point(x0 + 8, y0 + 18), "LEGENDA",
                     fontsize=11, color=(0.15, 0.15, 0.15))
    y = y0 + 18 + rij_h

    for stijl, label, count, vorm in visible:
        if vorm == "pijl":
            # Pijltje icoon
            s = page.new_shape()
            s.draw_line(fitz.Point(x0 + 8, y - 4), fitz.Point(x0 + 22, y - 4))
            s.finish(color=stijl["color"], width=2.0)
            s.commit()
            s2 = page.new_shape()
            s2.draw_line(fitz.Point(x0 + 22, y - 4), fitz.Point(x0 + 18, y - 8))
            s2.draw_line(fitz.Point(x0 + 22, y - 4), fitz.Point(x0 + 18, y))
            s2.finish(color=stijl["color"], width=1.5)
            s2.commit()
        else:
            # Vierkantje icoon
            blok = fitz.Rect(x0 + 8, y - 10, x0 + 22, y)
            s = page.new_shape()
            s.draw_rect(blok)
            s.finish(color=stijl["color"], fill=stijl["color"],
                     fill_opacity=0.35, width=1.0)
            s.commit()

        page.insert_text(fitz.Point(x0 + 30, y),
                         f"{label} ({count})", fontsize=10, color=(0, 0, 0))
        y += rij_h


def _teken_laag_vakjes(page, items: list, stijl: dict, nummer_start: int,
                       doorstreep: bool = False):
    """Teken vakjes + nummers voor een laag. Returns volgende nummer."""
    rects = [it["rect"] for it in items]
    merged = merge_overlapping_rects(rects)

    for rect in merged:
        _teken_rect(page, rect, stijl)
        if doorstreep:
            _teken_doorstreep(page, rect, stijl)

    nr = nummer_start
    for item in items:
        _teken_nummer(page, item["rect"], nr, stijl["color"])
        item["nr"] = nr
        nr += 1
    return nr


def _teken_kleine_pijl(page, rect: fitz.Rect, nummer: int, kleur: tuple):
    """Klein pijltje naar het midden van de rect — voor wand-indicatoren."""
    import math

    target_x = (rect.x0 + rect.x1) / 2
    target_y = (rect.y0 + rect.y1) / 2

    richting = nummer % 4
    offsets = [(50, -25), (-55, -25), (50, 25), (-55, 25)]
    dx, dy = offsets[richting]
    start_x = target_x + dx
    start_y = target_y + dy

    pw, ph = page.rect.width, page.rect.height
    start_x = max(20, min(pw - 40, start_x))
    start_y = max(20, min(ph - 20, start_y))

    # Pijllijn
    shape = page.new_shape()
    shape.draw_line(fitz.Point(start_x, start_y), fitz.Point(target_x, target_y))
    shape.finish(color=kleur, width=1.5)
    shape.commit()

    # Pijlpunt
    line_dx = target_x - start_x
    line_dy = target_y - start_y
    line_len = math.hypot(line_dx, line_dy)
    if line_len > 0:
        ux, uy = line_dx / line_len, line_dy / line_len
        base_x = target_x - ux * 8
        base_y = target_y - uy * 8
        px, py = -uy * 4, ux * 4
        s = page.new_shape()
        s.draw_line(fitz.Point(target_x, target_y),
                    fitz.Point(base_x + px, base_y + py))
        s.draw_line(fitz.Point(base_x + px, base_y + py),
                    fitz.Point(base_x - px, base_y - py))
        s.draw_line(fitz.Point(base_x - px, base_y - py),
                    fitz.Point(target_x, target_y))
        s.finish(color=kleur, fill=kleur, fill_opacity=0.8, width=0.3)
        s.commit()

    # Nummer
    label = str(nummer)
    tekst_x = start_x + 3
    tekst_y = start_y - 2
    tw = len(label) * 4 + 4
    bg = fitz.Rect(tekst_x - 2, tekst_y - 7, tekst_x + tw, tekst_y + 2)
    s2 = page.new_shape()
    s2.draw_rect(bg)
    s2.finish(color=kleur, fill=(1, 1, 1), fill_opacity=0.9, width=0.3)
    s2.commit()
    try:
        page.insert_text(fitz.Point(tekst_x, tekst_y), label,
                         fontsize=7, color=kleur)
    except Exception:
        pass


def _teken_laag_kleine_pijlen(page, items: list, stijl: dict, nummer_start: int):
    """Teken kleine pijltjes voor wand-indicatoren. Returns volgende nummer."""
    nr = nummer_start
    for item in items:
        _teken_kleine_pijl(page, item["rect"], nr, stijl["color"])
        item["nr"] = nr
        nr += 1
    return nr


def _teken_laag_pijlen(page, items: list, stijl: dict, nummer_start: int):
    """Teken pijlen + nummers voor wandwijzigingen. Returns volgende nummer."""
    nr = nummer_start
    for item in items:
        _teken_pijl_met_nummer(page, item["rect"], nr, stijl["color"])
        item["nr"] = nr
        nr += 1
    return nr


def _verschuif_cirkels(items: list, min_afstand: float = 20.0) -> list:
    """Verschuif cirkelposities iteratief zodat ze minimaal min_afstand uit elkaar staan."""
    import math as _math
    for _ in range(20):
        veranderd = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                r_i = items[i]["rect"]
                r_j = items[j]["rect"]
                cx_i = (r_i.x0 + r_i.x1) / 2
                cy_i = (r_i.y0 + r_i.y1) / 2
                cx_j = (r_j.x0 + r_j.x1) / 2
                cy_j = (r_j.y0 + r_j.y1) / 2
                d = _math.hypot(cx_j - cx_i, cy_j - cy_i)
                if d < min_afstand and d > 0.01:
                    dx = (cx_j - cx_i) / d
                    dy = (cy_j - cy_i) / d
                    shift = (min_afstand - d) / 2 + 1.0
                    hw = (r_j.x1 - r_j.x0) / 2
                    hh = (r_j.y1 - r_j.y0) / 2
                    ncx = cx_j + dx * shift
                    ncy = cy_j + dy * shift
                    items[j]["rect"] = fitz.Rect(ncx - hw, ncy - hh, ncx + hw, ncy + hh)
                    items[j]["center"] = [ncx, ncy]
                    veranderd = True
        if not veranderd:
            break
    return items


def _teken_laag_cirkels(page, items: list, stijl: dict, nummer_start: int,
                         doorstreep: bool = False) -> int:
    """Teken cirkels (r=8, opacity=0.3) + labels voor wandwijzigingen.

    Anti-overlap: cirkels worden uit elkaar geschoven, labels worden
    verticaal gestapeld als ze te dicht bij elkaar staan.
    Returns volgende nummer.
    """
    import math as _math
    RADIUS = 8.0
    LABEL_FONTSIZE = 7
    pw, ph = page.rect.width, page.rect.height

    if not items:
        return nummer_start

    # Maak een kopie zodat origineel niet muteert
    items = [dict(it) for it in items]

    # Anti-overlap cirkels
    items = _verschuif_cirkels(items, min_afstand=20.0)

    # Bereken label-posities
    label_pos = []
    for item in items:
        r = item["rect"]
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        beschr = item.get("beschrijving", "")
        label_w = len(beschr) * 4.2 + 6
        # Links van cirkel als dicht bij rechterrand (renvooi)
        if cx + RADIUS + label_w + 10 > pw * 0.84:
            lx = cx - RADIUS - label_w - 4
        else:
            lx = cx + RADIUS + 4
        ly = cy + 3
        label_pos.append([lx, ly])

    # Anti-overlap labels (verticaal stapelen)
    for i in range(len(label_pos)):
        for j in range(i):
            if abs(label_pos[i][0] - label_pos[j][0]) < 160 and abs(label_pos[i][1] - label_pos[j][1]) < 10:
                label_pos[i][1] = label_pos[j][1] + 12

    nr = nummer_start
    for i, item in enumerate(items):
        r = item["rect"]
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2

        # Cirkel met opacity 0.3
        shape = page.new_shape()
        shape.draw_circle(fitz.Point(cx, cy), RADIUS)
        shape.finish(color=stijl["color"], fill=stijl["fill"], fill_opacity=0.3, width=1.2)
        shape.commit()

        # X door cirkel bij verwijderd
        if doorstreep:
            off = RADIUS * 0.6
            for p1, p2 in [
                ((cx - off, cy - off), (cx + off, cy + off)),
                ((cx - off, cy + off), (cx + off, cy - off)),
            ]:
                s = page.new_shape()
                s.draw_line(fitz.Point(*p1), fitz.Point(*p2))
                s.finish(color=stijl["color"], width=1.5)
                s.commit()

        # Nummer in de cirkel
        try:
            label_nr = str(nr)
            nx = cx - 3 if len(label_nr) == 1 else cx - 5
            page.insert_text(fitz.Point(nx, cy + 3), label_nr, fontsize=6, color=stijl["color"])
        except Exception:
            pass

        # Label met witte achtergrond en verbindingslijn
        lx, ly = label_pos[i]
        beschr = item.get("beschrijving", "")
        if beschr:
            label_w = len(beschr) * 4.2 + 6
            bg = fitz.Rect(lx - 1, ly - LABEL_FONTSIZE - 1, lx + label_w, ly + 2)
            s = page.new_shape()
            s.draw_rect(bg)
            s.finish(color=None, fill=(1, 1, 1), fill_opacity=0.85, width=0)
            s.commit()
            # Verbindingslijn cirkel -> label
            connect_x = cx + RADIUS if lx > cx else cx - RADIUS
            s2 = page.new_shape()
            s2.draw_line(fitz.Point(connect_x, cy), fitz.Point(lx, ly - 2))
            s2.finish(color=stijl["color"], width=0.4)
            s2.commit()
            try:
                page.insert_text(fitz.Point(lx, ly), beschr,
                                  fontsize=LABEL_FONTSIZE, color=stijl["color"])
            except Exception:
                pass

        item["nr"] = nr
        nr += 1

    return nr


def _filter_wanden_bij_maat(wand_items: list, maat_items: list,
                            drempel: float = 60.0) -> list:
    """Verwijder nieuwe-wand-stippen die te dicht bij een maatwijziging liggen.

    Als er al een oranje maat-markering op die plek staat, ziet de lezer
    al dat er iets veranderd is — extra stip is dan overbodig.
    """
    import math as _math
    maat_centra = []
    for m in maat_items:
        r = m.get("rect")
        if r:
            maat_centra.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))

    resultaat = []
    for w in wand_items:
        r = w.get("rect")
        if r is None:
            resultaat.append(w)
            continue
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        te_dicht = any(
            _math.hypot(cx - mx, cy - my) < drempel
            for mx, my in maat_centra
        )
        if not te_dicht:
            resultaat.append(w)
    return resultaat


def _dedup_wand_items(items: list, drempel: float = 30.0) -> list:
    """Verwijder dubbele wandmarkeringen die <drempel pt van elkaar liggen.

    Behoudt het item met een wandtype-naam (voorkeur) boven een item zonder.
    """
    if len(items) <= 1:
        return items

    # Sorteer: items met wandtype-beschrijving eerst (hogere kwaliteit)
    def _heeft_wandtype(item):
        b = item.get("beschrijving", "")
        return "onbekend" not in b.lower() and (":" in b or "→" in b)

    items_sorted = sorted(items, key=lambda i: (not _heeft_wandtype(i),))
    result = []
    for item in items_sorted:
        rect = item.get("rect")
        if rect is None:
            result.append(item)
            continue
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        is_dup = False
        for bestaand in result:
            br = bestaand.get("rect")
            if br is None:
                continue
            bx = (br.x0 + br.x1) / 2
            by = (br.y0 + br.y1) / 2
            if ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 < drempel:
                is_dup = True
                break
        if not is_dup:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Tekening pagina bouwen
# ---------------------------------------------------------------------------

def _bouw_tekening_pagina(doc, clean_path: str, diff_result: dict,
                          pagina: int, naam: str,
                          oud_path: str | None = None,
                          nieuw_path: str | None = None):
    """Kopieer tekening + teken markeringen. Returns alle secties met nummers."""
    src = fitz.open(clean_path)
    doc.insert_pdf(src, from_page=pagina, to_page=pagina)
    page = doc[-1]
    pw, ph = page.rect.width, page.rect.height
    src.close()

    _teken_titel_balk(page, f"WIJZIGINGEN \u2014 {naam}")

    # Layout ophalen uit diff_result (auto-gedetecteerd)
    layout = diff_result.get("_layout_obj")

    # Alle tekst-items voor locatie-context
    alle_tekst = diff_result.get("nieuw_tekst_items", [])

    # Verzamel de 4 categorieën
    maat_items, maat_rv = _collect_maat(diff_result, pw, ph, alle_tekst, layout)
    nieuwe_maat_items, nieuwe_maat_rv = _collect_nieuwe_maten(diff_result, pw, ph, alle_tekst, layout)
    opp_items, opp_rv = _collect_oppervlakte(diff_result, pw, ph, alle_tekst, layout)
    ruimte_items, ruimte_rv = _collect_ruimtenaam(diff_result, pw, ph, layout)
    total_renvooi = maat_rv + nieuwe_maat_rv + opp_rv + ruimte_rv

    # Teken per laag met doorlopende nummering
    nr = 1
    nr = _teken_laag_vakjes(page, maat_items, ORANJE, nr)
    nr = _teken_laag_vakjes(page, nieuwe_maat_items, PAARS, nr)
    nr = _teken_laag_vakjes(page, opp_items, GROEN, nr)
    nr = _teken_laag_vakjes(page, ruimte_items, BLAUW, nr)

    _teken_legenda(page, [
        (ORANJE, "Maatwijziging", len(maat_items), "rect"),
        (PAARS, "Nieuwe maat toegevoegd", len(nieuwe_maat_items), "rect"),
        (GROEN, "Oppervlaktewijziging", len(opp_items), "rect"),
        (BLAUW, "Ruimtenaamwijziging", len(ruimte_items), "rect"),
    ])

    secties = {
        "maat": maat_items,
        "nieuwe_maat": nieuwe_maat_items,
        "oppervlakte": opp_items,
        "ruimtenaam": ruimte_items,
    }
    totaal = sum(len(v) for v in secties.values())

    return secties, totaal, total_renvooi


# ---------------------------------------------------------------------------
# Rapport (A4 samenvatting)
# ---------------------------------------------------------------------------

RAPPORT_SECTIES = [
    ("maat", "Maatwijzigingen", ORANJE),
    ("nieuwe_maat", "Nieuwe maat toegevoegd", PAARS),
    ("oppervlakte", "Oppervlaktewijzigingen", GROEN),
    ("ruimtenaam", "Ruimtenaamwijzigingen", BLAUW),
]


def _bouw_samenvatting(doc, diff_result: dict, oud_path: str, nieuw_path: str,
                       secties: dict, totaal: int, renvooi: int):
    """A4 samenvatting met 6 secties."""
    meta = diff_result.get("meta", {})
    kleur_zwart = (0, 0, 0)
    kleur_grijs = (0.4, 0.4, 0.4)

    page = doc.new_page(width=595, height=842)
    y = 50

    def _nieuwe_pagina():
        nonlocal page, y
        page = doc.new_page(width=595, height=842)
        y = 50
        page.insert_text(fitz.Point(50, y), "Wijzigingsrapport (vervolg)",
                         fontsize=11, color=(0.1, 0.3, 0.45))
        y += 20

    def _check_ruimte(nodig: int = 30):
        nonlocal y
        if y > 842 - nodig:
            _nieuwe_pagina()

    # Header
    page.insert_text(fitz.Point(50, y),
                     "Wijzigingsrapport K&K Demarcatietekeningen",
                     fontsize=14, color=(0.1, 0.3, 0.45))
    y += 25

    for regel in [
        f"Oud: {meta.get('oud_bestand', oud_path)}",
        f"Nieuw: {meta.get('nieuw_bestand', nieuw_path)}",
        f"Pagina: {meta.get('pagina', 1)}",
        f"Datum: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
    ]:
        page.insert_text(fitz.Point(50, y), regel, fontsize=8, color=kleur_grijs)
        y += 14
    y += 10

    # Totalen
    page.insert_text(fitz.Point(50, y), f"Totaal: {totaal} wijzigingen",
                     fontsize=11, color=kleur_zwart)
    y += 18

    if renvooi > 0:
        page.insert_text(fitz.Point(50, y),
                         f"({renvooi} items in renvooi/legenda-zone niet gemarkeerd)",
                         fontsize=8, color=kleur_grijs)
        y += 14
    y += 8

    # 6 secties
    for sectie_key, sectie_label, sectie_stijl in RAPPORT_SECTIES:
        items = secties.get(sectie_key, [])
        if not items:
            continue

        _check_ruimte(40)

        # Scheidingslijn
        shape = page.new_shape()
        shape.draw_line(fitz.Point(50, y), fitz.Point(545, y))
        shape.finish(color=(0.8, 0.8, 0.8), width=0.3)
        shape.commit()
        y += 10

        # Sectie header
        page.insert_text(fitz.Point(50, y), sectie_label,
                         fontsize=10, color=sectie_stijl["color"])
        page.insert_text(fitz.Point(50 + len(sectie_label) * 5.5 + 5, y),
                         f"({len(items)})", fontsize=8, color=kleur_grijs)
        y += 14

        # Items
        for item in items:
            _check_ruimte(14)
            nr = item.get("nr", "?")
            beschr = item["beschrijving"]
            if len(beschr) > 80:
                beschr = beschr[:77] + "..."

            # Waarschuwing bij ruimtenaam
            waarschuwing = ""
            if sectie_key == "ruimtenaam":
                waarschuwing = "  \u26a0 Andere eisen mogelijk!"

            page.insert_text(fitz.Point(55, y), str(nr),
                             fontsize=8, color=sectie_stijl["color"])
            page.insert_text(fitz.Point(75, y), beschr,
                             fontsize=8, color=kleur_zwart)
            y += 11

            if waarschuwing:
                _check_ruimte(12)
                page.insert_text(fitz.Point(75, y), waarschuwing,
                                 fontsize=7, color=(0.8, 0.0, 0.0))
                y += 11

        y += 6


# ---------------------------------------------------------------------------
# Hoofdfuncties
# ---------------------------------------------------------------------------

def generate_overlay_pdf(
    oud_pdf_path: str,
    nieuw_pdf_path: str,
    diff_result: dict,
    pagina: int = 0,
) -> bytes:
    """Genereer PDF: samenvatting + gemarkeerde tekening."""
    import os

    nieuw_clean = strip_annotations(nieuw_pdf_path)

    try:
        doc = fitz.open()
        nieuw_naam = diff_result.get("meta", {}).get("nieuw_bestand", "nieuw.pdf")

        secties, totaal, renvooi = _bouw_tekening_pagina(
            doc, nieuw_clean, diff_result, pagina, nieuw_naam,
            oud_path=oud_pdf_path, nieuw_path=nieuw_pdf_path)

        _bouw_samenvatting(doc, diff_result, oud_pdf_path, nieuw_pdf_path,
                           secties, totaal, renvooi)

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    finally:
        try:
            os.unlink(nieuw_clean)
        except OSError:
            pass


def generate_multi_page_overlay(
    oud_pdf_path: str,
    nieuw_pdf_path: str,
    aantal_paginas: int,
    oud_naam: str = "oud.pdf",
    nieuw_naam: str = "nieuw.pdf",
    config: "DiffConfig | None" = None,
) -> bytes:
    """Genereer compleet rapport voor ALLE pagina's."""
    import os
    from .config import DiffConfig as _DiffConfig
    from .diff_engine import run_diff

    if config is None:
        config = _DiffConfig()

    nieuw_clean = strip_annotations(nieuw_pdf_path)

    try:
        doc = fitz.open()

        for pag_idx in range(aantal_paginas):
            diff_result = run_diff(oud_pdf_path, nieuw_pdf_path, pagina=pag_idx,
                                   config=config)

            if "error" in diff_result:
                continue

            if "meta" in diff_result:
                diff_result["meta"]["oud_bestand"] = oud_naam
                diff_result["meta"]["nieuw_bestand"] = nieuw_naam

            totalen = diff_result.get("totalen", {})
            if sum(totalen.values()) == 0:
                continue

            pag_label = f"{nieuw_naam} - pagina {pag_idx + 1}"

            secties, totaal, renvooi = _bouw_tekening_pagina(
                doc, nieuw_clean, diff_result, pag_idx, pag_label,
                oud_path=oud_pdf_path, nieuw_path=nieuw_pdf_path)

            _bouw_samenvatting(
                doc, diff_result, oud_pdf_path, nieuw_pdf_path,
                secties, totaal, renvooi)

        if len(doc) == 0:
            page = doc.new_page(width=595, height=842)
            page.insert_text(
                fitz.Point(50, 100),
                "Geen wijzigingen gevonden op de vergelijkbare pagina's.",
                fontsize=14, color=(0.1, 0.3, 0.45),
            )

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    finally:
        try:
            os.unlink(nieuw_clean)
        except OSError:
            pass


def generate_split_rapport(
    oud_pdf_path: str,
    nieuw_pdf_path: str,
    aantal_paginas: int,
    oud_naam: str = "oud.pdf",
    nieuw_naam: str = "nieuw.pdf",
    config: "DiffConfig | None" = None,
) -> tuple[bytes, bytes]:
    """
    Genereer TWEE losse PDF's:
      1. rapport_pdf: alle samenvattingspagina's (A4)
      2. tekeningen_pdf: alle gemarkeerde tekeningen

    Richard kan ze naast elkaar openen.
    """
    import os
    from .config import DiffConfig as _DiffConfig
    from .diff_engine import run_diff

    if config is None:
        config = _DiffConfig()

    nieuw_clean = strip_annotations(nieuw_pdf_path)

    try:
        doc_rapport = fitz.open()
        doc_tekeningen = fitz.open()

        for pag_idx in range(aantal_paginas):
            diff_result = run_diff(oud_pdf_path, nieuw_pdf_path, pagina=pag_idx,
                                   config=config)

            if "error" in diff_result:
                continue

            if "meta" in diff_result:
                diff_result["meta"]["oud_bestand"] = oud_naam
                diff_result["meta"]["nieuw_bestand"] = nieuw_naam

            totalen = diff_result.get("totalen", {})
            if sum(totalen.values()) == 0:
                continue

            pag_label = f"{nieuw_naam} - pagina {pag_idx + 1}"

            # Bouw tekening in temp doc, dan verplaats
            doc_temp = fitz.open()
            secties, totaal, renvooi = _bouw_tekening_pagina(
                doc_temp, nieuw_clean, diff_result, pag_idx, pag_label,
                oud_path=oud_pdf_path, nieuw_path=nieuw_pdf_path)
            doc_tekeningen.insert_pdf(doc_temp)
            doc_temp.close()

            # Bouw samenvatting in rapport doc
            _bouw_samenvatting(
                doc_rapport, diff_result, oud_pdf_path, nieuw_pdf_path,
                secties, totaal, renvooi)

        if len(doc_rapport) == 0:
            page = doc_rapport.new_page(width=595, height=842)
            page.insert_text(
                fitz.Point(50, 100),
                "Geen wijzigingen gevonden.",
                fontsize=14, color=(0.1, 0.3, 0.45),
            )

        if len(doc_tekeningen) == 0:
            page = doc_tekeningen.new_page(width=595, height=842)
            page.insert_text(
                fitz.Point(50, 100),
                "Geen tekeningen met wijzigingen.",
                fontsize=14, color=(0.1, 0.3, 0.45),
            )

        rapport_bytes = doc_rapport.tobytes()
        tekeningen_bytes = doc_tekeningen.tobytes()
        doc_rapport.close()
        doc_tekeningen.close()
        return rapport_bytes, tekeningen_bytes
    finally:
        try:
            os.unlink(nieuw_clean)
        except OSError:
            pass

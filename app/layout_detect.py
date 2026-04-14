"""
Auto-detectie van tekening-layout: titelblok, legenda, schaal.

Vervangt hardcoded positie-ratio's door automatische detectie.
Fallback naar config-defaults als detectie niet lukt.
"""

import re
from dataclasses import dataclass, field

import fitz

from .config import DiffConfig


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PageLayout:
    """Gedetecteerde layout van een tekeningpagina."""

    page_rect: fitz.Rect
    titelblok: fitz.Rect | None = None
    legenda: fitz.Rect | None = None
    koptekst: fitz.Rect | None = None
    drawing_area: fitz.Rect | None = None
    scale: float | None = None
    legenda_mapping: dict = field(default_factory=dict)

    # Interne velden voor fallback
    _fallback_renvooi_x: float = 0.0
    _fallback_koptekst_y: float = 0.0

    def is_excluded(self, pos: list | tuple) -> bool:
        """Check of een punt in een uitgesloten zone valt.

        Sluit alleen de legenda (renvooi) en koptekst uit —
        NIET het titelblok, omdat detailtekeningen soms in
        datzelfde rechterdeel van de pagina staan.
        """
        pt = fitz.Point(pos[0], pos[1])

        if self.legenda and self.legenda.contains(pt):
            return True
        if self.koptekst and self.koptekst.contains(pt):
            return True

        # Fallback: als geen zones gedetecteerd, gebruik ratio's
        if not self.legenda:
            if pos[0] > self._fallback_renvooi_x:
                return True
            if pos[1] < self._fallback_koptekst_y:
                return True

        return False

    def is_in_legenda(self, pos: list | tuple) -> bool:
        """Check of een punt in de legenda-zone valt."""
        if self.legenda:
            return self.legenda.contains(fitz.Point(pos[0], pos[1]))
        # Fallback
        if self._fallback_renvooi_x > 0:
            return pos[0] > self._fallback_renvooi_x
        return False


# ---------------------------------------------------------------------------
# Tekst extractie (lichtgewicht, alleen voor layout-detectie)
# ---------------------------------------------------------------------------

def _extract_tekst_items(page) -> list[dict]:
    """Extraheer tekst-items met positie (voor layout-detectie)."""
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
                bbox = span.get("bbox", (0, 0, 0, 0))
                items.append({
                    "tekst": tekst,
                    "pos": (bbox[0], bbox[1]),
                    "bbox": bbox,
                })
    return items


def _extract_fills(page) -> list[dict]:
    """Extraheer gekleurde vlakken (voor legenda-detectie)."""
    items = []
    try:
        paths = page.get_drawings()
    except Exception:
        return items
    for path in paths:
        fill = path.get("fill")
        if fill is None:
            continue
        # Normaliseer kleur naar RGB
        if isinstance(fill, (int, float)):
            r = g = b = float(fill)
        elif len(fill) == 1:
            r = g = b = float(fill[0])
        elif len(fill) == 3:
            r, g, b = float(fill[0]), float(fill[1]), float(fill[2])
        elif len(fill) == 4:
            c, m, y, k = (float(x) for x in fill)
            r, g, b = (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)
        else:
            continue

        rect = path.get("rect", fitz.Rect())
        if rect.width <= 0 or rect.height <= 0:
            continue

        opp = rect.width * rect.height
        items.append({
            "rgb": (round(r, 3), round(g, 3), round(b, 3)),
            "pos": (round(rect.x0, 1), round(rect.y0, 1)),
            "bbox": (round(rect.x0, 1), round(rect.y0, 1),
                     round(rect.x1, 1), round(rect.y1, 1)),
            "oppervlakte": round(opp, 1),
        })
    return items


# ---------------------------------------------------------------------------
# Titelblok detectie
# ---------------------------------------------------------------------------

_RE_METADATA = re.compile(
    r"(\d{2}-\d{2}-\d{4})"           # datum DD-MM-YYYY
    r"|(\d{1,2}-\d{1,2}-'\d{2})"     # datum korte notatie
    r"|(^[A-Z]$)"                     # revisieletter
    r"|(NDO|definitief)"              # status
    r"|(\d{1,2}e\s+uitgave)"          # uitgave
    r"|(schaal|scale|getekend|drawn|controle|checked)"
    r"|(datum|date|project)"
    r"|(blad|sheet|rev\.?)"
    r"|(1:\d{2,3})",                  # schaal notatie
    re.IGNORECASE,
)


def _detect_titelblok(
    tekst_items: list[dict],
    page_rect: fitz.Rect,
    config: DiffConfig,
) -> fitz.Rect | None:
    """Detecteer het titelblok op basis van metadata-tekst dichtheid.

    Bouwt een grid over de pagina en zoekt het gebied met de hoogste
    concentratie metadata-tekst (datums, revisies, schaal, etc.).
    """
    pw, ph = page_rect.width, page_rect.height
    if not tekst_items:
        return None

    # Grid: 10x10 cellen
    cols, rows = 10, 10
    cell_w, cell_h = pw / cols, ph / rows

    # Tel metadata-hits per cel
    grid = [[0] * cols for _ in range(rows)]
    for item in tekst_items:
        if _RE_METADATA.search(item["tekst"]):
            cx = min(int(item["pos"][0] / cell_w), cols - 1)
            cy = min(int(item["pos"][1] / cell_h), rows - 1)
            grid[cy][cx] += 1

    # Zoek het 3x3 blok met de hoogste dichtheid (titelblok is typisch een cluster)
    beste_score = 0
    beste_cx, beste_cy = cols - 1, rows - 1  # default: rechtsonder

    for cy in range(rows - 2):
        for cx in range(cols - 2):
            score = sum(
                grid[cy + dy][cx + dx]
                for dy in range(3)
                for dx in range(3)
            )
            if score > beste_score:
                beste_score = score
                beste_cx = cx
                beste_cy = cy

    # Minimale confidence: minstens 3 metadata-hits in het cluster
    if beste_score < 3:
        return None

    # Expandeer het gevonden cluster naar het volledige titelblok
    # door alle metadata-tekst in de buurt mee te nemen
    cluster_x0 = beste_cx * cell_w
    cluster_y0 = beste_cy * cell_h
    cluster_x1 = (beste_cx + 3) * cell_w
    cluster_y1 = (beste_cy + 3) * cell_h

    # Verfijn: neem alle metadata-items die in of dichtbij het cluster liggen
    margin = max(cell_w, cell_h) * 0.5
    relevant = [
        item for item in tekst_items
        if _RE_METADATA.search(item["tekst"])
        and item["pos"][0] >= cluster_x0 - margin
        and item["pos"][1] >= cluster_y0 - margin
        and item["pos"][0] <= cluster_x1 + margin
        and item["pos"][1] <= cluster_y1 + margin
    ]

    if len(relevant) < 3:
        return None

    # Bouw bounding box van het titelblok
    # Expandeer naar pagina-randen als het cluster aan de rand zit
    x_coords = [it["bbox"][0] for it in relevant] + [it["bbox"][2] for it in relevant]
    y_coords = [it["bbox"][1] for it in relevant] + [it["bbox"][3] for it in relevant]

    tb_x0 = min(x_coords) - 10
    tb_y0 = min(y_coords) - 10
    tb_x1 = max(x_coords) + 10
    tb_y1 = max(y_coords) + 10

    # Als het titelblok tegen de rechterrand zit, expandeer naar pw
    if tb_x1 > pw * 0.85:
        tb_x1 = pw
    # Als het tegen de onderrand zit, expandeer naar ph
    if tb_y1 > ph * 0.85:
        tb_y1 = ph

    return fitz.Rect(tb_x0, tb_y0, tb_x1, tb_y1)


# ---------------------------------------------------------------------------
# Legenda detectie
# ---------------------------------------------------------------------------

def _detect_legenda(
    tekst_items: list[dict],
    fills: list[dict],
    page_rect: fitz.Rect,
    config: DiffConfig,
) -> tuple[fitz.Rect | None, dict]:
    """Detecteer de legenda door gekleurde blokjes + tekst-labels te clusteren.

    Zoekt verticale kolommen van gekleurde fills met tekst ernaast.
    Returns: (legenda_rect, {(r,g,b): label, ...})
    """
    pw, ph = page_rect.width, page_rect.height

    # Filter: kleine gekleurde fills (legenda-blokjes)
    kandidaat_fills = [
        f for f in fills
        if config.fill_min_area < f["oppervlakte"] < config.fill_max_area
        and not _is_wit_of_zwart(f["rgb"])
    ]

    if len(kandidaat_fills) < 3:
        return None, {}

    # Groepeer fills in verticale kolommen (zelfde x-band ±15pt)
    kolommen = _groepeer_verticaal(kandidaat_fills, x_tolerantie=15.0)

    # Score elke kolom: hoeveel fills hebben een tekst-label ernaast?
    beste_kolom = None
    beste_mapping = {}
    beste_score = 0

    for kolom in kolommen:
        if len(kolom) < 2:
            continue
        mapping = _match_fills_met_tekst(kolom, tekst_items)
        score = len(mapping)
        if score > beste_score:
            beste_score = score
            beste_kolom = kolom
            beste_mapping = mapping

    if beste_score < 2 or not beste_kolom:
        return None, {}

    # Bouw bounding rect van legenda
    all_x = []
    all_y = []
    for f in beste_kolom:
        all_x.extend([f["bbox"][0], f["bbox"][2]])
        all_y.extend([f["bbox"][1], f["bbox"][3]])
    # Inclusief tekst-labels (staan rechts van de fills)
    for f in beste_kolom:
        for t in tekst_items:
            dx = t["pos"][0] - f["pos"][0]
            dy = abs(t["pos"][1] - f["pos"][1])
            if dy < 20 and 0 < dx < 100:
                all_x.append(t["bbox"][2])
                all_y.extend([t["bbox"][1], t["bbox"][3]])

    legenda_rect = fitz.Rect(
        min(all_x) - 10, min(all_y) - 15,
        max(all_x) + 10, max(all_y) + 15,
    )

    return legenda_rect, beste_mapping


def _is_wit_of_zwart(rgb: tuple) -> bool:
    r, g, b = rgb
    if r > 0.95 and g > 0.95 and b > 0.95:
        return True
    if r < 0.05 and g < 0.05 and b < 0.05:
        return True
    return False


def _groepeer_verticaal(
    fills: list[dict], x_tolerantie: float = 15.0,
) -> list[list[dict]]:
    """Groepeer fills in verticale kolommen op basis van x-positie."""
    fills_sorted = sorted(fills, key=lambda f: f["pos"][0])
    kolommen: list[list[dict]] = []

    for fill in fills_sorted:
        geplaatst = False
        for kolom in kolommen:
            # Check of de x-positie past bij deze kolom
            kolom_x = sum(f["pos"][0] for f in kolom) / len(kolom)
            if abs(fill["pos"][0] - kolom_x) < x_tolerantie:
                kolom.append(fill)
                geplaatst = True
                break
        if not geplaatst:
            kolommen.append([fill])

    # Sorteer elke kolom op y-positie
    for kolom in kolommen:
        kolom.sort(key=lambda f: f["pos"][1])

    return kolommen


def _match_fills_met_tekst(
    fills: list[dict], tekst_items: list[dict],
) -> dict:
    """Koppel gekleurde fills aan tekst-labels.

    Returns: {(r, g, b): label_tekst, ...}
    """
    mapping = {}

    for fill in fills:
        fx, fy = fill["pos"]
        rgb = fill["rgb"]

        # Zoek dichtstbijzijnde tekst-label rechts van de fill
        beste_tekst = None
        beste_score = float("inf")

        for t in tekst_items:
            tx, ty = t["pos"]
            label = t["tekst"].strip()

            # Tekst moet rechts van fill staan, op dezelfde hoogte
            dx = tx - fx
            dy = abs(ty - fy)

            if dy > 20 or dx < 0 or dx > 100:
                continue
            if len(label) < 3:
                continue
            # Skip metadata
            if _RE_METADATA.search(label):
                continue

            score = dy + dx * 0.1
            if score < beste_score:
                beste_score = score
                beste_tekst = label

        if beste_tekst is not None:
            key = (round(rgb[0], 2), round(rgb[1], 2), round(rgb[2], 2))
            mapping[key] = beste_tekst

    return mapping


# ---------------------------------------------------------------------------
# Schaal detectie
# ---------------------------------------------------------------------------

_RE_SCHAAL = re.compile(r"(?:schaal|scale)?\s*1\s*:\s*(\d{2,3})", re.IGNORECASE)


def _detect_schaal(
    tekst_items: list[dict],
    titelblok: fitz.Rect | None,
) -> float | None:
    """Detecteer de tekenschaal uit tekst-items (bijv. '1:50' → 50.0).

    Zoekt bij voorkeur binnen het titelblok.
    """
    kandidaten = []

    for item in tekst_items:
        m = _RE_SCHAAL.search(item["tekst"])
        if m:
            schaal = float(m.group(1))
            # Sanity check: gangbare schalen
            if 10 <= schaal <= 500:
                in_titelblok = (
                    titelblok is not None
                    and titelblok.contains(fitz.Point(item["pos"][0], item["pos"][1]))
                )
                kandidaten.append((schaal, in_titelblok))

    if not kandidaten:
        return None

    # Prefer schaal die in titelblok staat
    in_tb = [k for k in kandidaten if k[1]]
    if in_tb:
        return in_tb[0][0]
    return kandidaten[0][0]


# ---------------------------------------------------------------------------
# Koptekst detectie
# ---------------------------------------------------------------------------

def _detect_koptekst(
    tekst_items: list[dict],
    page_rect: fitz.Rect,
    config: DiffConfig,
) -> fitz.Rect | None:
    """Detecteer een dunne koptekst-strip bovenaan de pagina."""
    ph = page_rect.height
    pw = page_rect.width
    grens = ph * config.koptekst_y_ratio

    # Tel tekst-items in de bovenste strip
    items_in_kop = [t for t in tekst_items if t["pos"][1] < grens]

    if items_in_kop:
        return fitz.Rect(0, 0, pw, grens)

    return None


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def detect_layout(page, config: DiffConfig | None = None) -> PageLayout:
    """Detecteer de volledige layout van een tekeningpagina.

    Combineert titelblok, legenda, schaal en koptekst detectie.
    Fallback naar config-defaults als detectie faalt.
    """
    if config is None:
        config = DiffConfig()

    page_rect = page.rect
    pw, ph = page_rect.width, page_rect.height

    # Extractie
    tekst_items = _extract_tekst_items(page)
    fills = _extract_fills(page)

    # Detectie
    titelblok = _detect_titelblok(tekst_items, page_rect, config)
    legenda_rect, legenda_mapping = _detect_legenda(
        tekst_items, fills, page_rect, config)
    scale = _detect_schaal(tekst_items, titelblok)
    koptekst = _detect_koptekst(tekst_items, page_rect, config)

    # Bereken drawing area (pagina minus uitgesloten zones)
    da_x0, da_y0 = 0.0, 0.0
    da_x1, da_y1 = pw, ph

    if koptekst:
        da_y0 = max(da_y0, koptekst.y1)
    if titelblok:
        # Titelblok kan rechts of onderaan zitten
        if titelblok.x0 > pw * 0.5:
            da_x1 = min(da_x1, titelblok.x0)
        if titelblok.y0 > ph * 0.5:
            da_y1 = min(da_y1, titelblok.y0)

    drawing_area = fitz.Rect(da_x0, da_y0, da_x1, da_y1)

    layout = PageLayout(
        page_rect=page_rect,
        titelblok=titelblok,
        legenda=legenda_rect,
        koptekst=koptekst,
        drawing_area=drawing_area,
        scale=scale,
        legenda_mapping=legenda_mapping,
        _fallback_renvooi_x=pw * config.renvooi_x_ratio,
        _fallback_koptekst_y=ph * config.koptekst_y_ratio,
    )

    return layout

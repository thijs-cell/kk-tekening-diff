"""
Wandcluster-detectie voor K&K tekening-diff.

Detecteert nieuwe en verwijderde wanden door parallelle lijnparen te vinden
met een afstand die overeenkomt met typische wanddiktes (70-100mm op schaal).

Conservatief: liever een echte wand missen dan de tekening vullen met
false-positives. Max 5 resultaten per pagina.
"""

import math
from collections import defaultdict


# Wanddikte in PDF-punten bij verschillende schalen:
# Schaal 1:100 → 70mm = 2.0pt,  100mm = 2.8pt
# Schaal 1:50  → 70mm = 4.0pt,  100mm = 5.7pt
# Strakker bereik: alleen echte wanddiktes doorlaten
MIN_WAND_AFSTAND_PT = 1.8
MAX_WAND_AFSTAND_PT = 12.0  # Verruimd voor draagmuren (was 6.0)
MIN_WAND_LENGTE_PT = 30.0   # Verlaagd voor korte muurtjes (was 50.0)
MAX_WAND_LENGTE_PT = 800.0  # Langere lijnen zijn sectie/vloerlijnen, geen wanden
MAX_RESULTATEN = 50  # Verruimd (was 15)

# Wanddiktes in mm — voor cross-referencing met tekst
WANDDIKTE_MM = {50, 70, 100, 120, 150, 200}
# Korte wanden (<100pt) moeten bevestigd worden door nabije wanddikte-tekst
KORTE_WAND_DREMPEL = 100.0
TEKST_BEVESTIG_AFSTAND = 80.0  # Max afstand in pt tot bevestigende tekst

# Hoektolerantie voor parallelle lijnen (radialen)
HOEK_TOLERANTIE = 0.08


def _lijn_lengte(lijn: dict) -> float:
    """Bereken lengte van een lijnsegment."""
    vx = lijn["naar"][0] - lijn["van"][0]
    vy = lijn["naar"][1] - lijn["van"][1]
    return math.hypot(vx, vy)


def _lijn_hoek(lijn: dict) -> float:
    """Bereken hoek van een lijnsegment (0 tot pi)."""
    vx = lijn["naar"][0] - lijn["van"][0]
    vy = lijn["naar"][1] - lijn["van"][1]
    hoek = math.atan2(vy, vx)
    # Normaliseer naar 0..pi (richting maakt niet uit)
    if hoek < 0:
        hoek += math.pi
    return hoek


def _zijn_parallel(h1: float, h2: float) -> bool:
    """Controleer of twee hoeken parallel zijn."""
    verschil = abs(h1 - h2)
    if verschil > math.pi / 2:
        verschil = math.pi - verschil
    return verschil < HOEK_TOLERANTIE


def _loodrechte_afstand(l1: dict, l2: dict) -> float:
    """Bereken de loodrechte afstand tussen twee parallelle lijnen."""
    # Gebruik het middelpunt van l2 en de lijn door l1
    mx = (l2["van"][0] + l2["naar"][0]) / 2
    my = (l2["van"][1] + l2["naar"][1]) / 2

    x1, y1 = l1["van"]
    x2, y2 = l1["naar"]

    dx = x2 - x1
    dy = y2 - y1
    lengte = math.hypot(dx, dy)
    if lengte < 0.001:
        return float("inf")

    # Afstand punt tot lijn
    return abs(dy * mx - dx * my + x2 * y1 - y2 * x1) / lengte


def _overlap_langs_lijn(l1: dict, l2: dict) -> float:
    """Bereken hoeveel twee parallelle lijnen overlappen langs hun richting."""
    # Project beide lijnen op hun gezamenlijke richting
    dx = l1["naar"][0] - l1["van"][0]
    dy = l1["naar"][1] - l1["van"][1]
    lengte = math.hypot(dx, dy)
    if lengte < 0.001:
        return 0.0
    nx, ny = dx / lengte, dy / lengte

    # Projecties van l1
    p1a = l1["van"][0] * nx + l1["van"][1] * ny
    p1b = l1["naar"][0] * nx + l1["naar"][1] * ny
    # Projecties van l2
    p2a = l2["van"][0] * nx + l2["van"][1] * ny
    p2b = l2["naar"][0] * nx + l2["naar"][1] * ny

    start1, end1 = min(p1a, p1b), max(p1a, p1b)
    start2, end2 = min(p2a, p2b), max(p2a, p2b)

    overlap = min(end1, end2) - max(start1, start2)
    return max(0.0, overlap)


def _heeft_wanddikte_tekst_nabij(
    center: tuple[float, float],
    tekst_items: list[dict],
    max_afstand: float = TEKST_BEVESTIG_AFSTAND,
) -> bool:
    """Check of er een wanddikte-tekst (70, 100, etc.) in de buurt staat."""
    for t in tekst_items:
        tekst = t.get("tekst", "").strip()
        try:
            waarde = int(tekst)
        except ValueError:
            continue
        if waarde not in WANDDIKTE_MM:
            continue
        pos = t.get("pos", (0, 0))
        d = math.hypot(center[0] - pos[0], center[1] - pos[1])
        if d < max_afstand:
            return True
    return False


def detecteer_wand_clusters(
    lijnen: list[dict],
    min_afstand: float = MIN_WAND_AFSTAND_PT,
    max_afstand: float = MAX_WAND_AFSTAND_PT,
    min_lengte: float = MIN_WAND_LENGTE_PT,
    max_resultaten: int = MAX_RESULTATEN,
    tekst_items: list[dict] | None = None,
) -> list[dict]:
    """
    Detecteer wandclusters in een lijst lijnsegmenten.

    Zoekt paren van parallelle lijnen met een afstand die past bij
    een wanddikte op tekenschaal.

    Bij korte wanden (<100pt) wordt cross-referencing met tekst gedaan:
    alleen als er een wanddikte-tekst (70/100/etc.) in de buurt staat
    wordt het als wand geaccepteerd.

    Returns:
        lijst van {"center": (x, y), "bbox": (x0, y0, x1, y1), "dikte_pt": float}
    """
    if tekst_items is None:
        tekst_items = []

    # Filter: alleen lijnen met passende lengte (niet te kort, niet te lang)
    kandidaten = [
        l for l in lijnen
        if "van" in l
        and min_lengte <= _lijn_lengte(l) <= MAX_WAND_LENGTE_PT
    ]
    if not kandidaten:
        return []

    # Bereken hoeken
    hoeken = [_lijn_hoek(l) for l in kandidaten]

    # Groepeer op hoek met grid (0.1 rad buckets)
    hoek_grid: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(hoeken):
        bucket = int(h / 0.1)
        hoek_grid[bucket].append(i)

    gevonden: list[dict] = []
    gebruikt: set[int] = set()

    for bucket, indices in hoek_grid.items():
        # Check ook aangrenzende buckets
        alle_indices = list(indices)
        for nb in (bucket - 1, bucket + 1):
            alle_indices.extend(hoek_grid.get(nb, []))
        alle_indices = list(set(alle_indices))

        for i in range(len(alle_indices)):
            idx_a = alle_indices[i]
            if idx_a in gebruikt:
                continue
            la = kandidaten[idx_a]
            ha = hoeken[idx_a]

            for j in range(i + 1, len(alle_indices)):
                idx_b = alle_indices[j]
                if idx_b in gebruikt:
                    continue
                lb = kandidaten[idx_b]
                hb = hoeken[idx_b]

                if not _zijn_parallel(ha, hb):
                    continue

                afstand = _loodrechte_afstand(la, lb)
                if afstand < min_afstand or afstand > max_afstand:
                    continue

                # Check overlap langs de lijn (minstens 70% van de kortste)
                overlap = _overlap_langs_lijn(la, lb)
                min_len = min(_lijn_lengte(la), _lijn_lengte(lb))
                if overlap < min_len * 0.7:
                    continue

                # Gevonden: dit is een wandpaar
                alle_x = [la["van"][0], la["naar"][0], lb["van"][0], lb["naar"][0]]
                alle_y = [la["van"][1], la["naar"][1], lb["van"][1], lb["naar"][1]]
                bbox = (min(alle_x), min(alle_y), max(alle_x), max(alle_y))
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                gem_lengte = (_lijn_lengte(la) + _lijn_lengte(lb)) / 2

                # Korte wanden: eis bevestiging door wanddikte-tekst in de buurt
                if gem_lengte < KORTE_WAND_DREMPEL and tekst_items:
                    if not _heeft_wanddikte_tekst_nabij(center, tekst_items):
                        continue  # Waarschijnlijk maatlijn of arcering

                # Confidence: langere lijnen = beter, tekst-bevestigd = bonus
                confidence = gem_lengte / 100.0
                if gem_lengte < KORTE_WAND_DREMPEL and tekst_items:
                    if _heeft_wanddikte_tekst_nabij(center, tekst_items):
                        confidence += 0.5  # Bonus voor tekst-bevestiging

                gevonden.append({
                    "center": center,
                    "bbox": bbox,
                    "dikte_pt": round(afstand, 1),
                    "confidence": round(confidence, 2),
                })
                gebruikt.add(idx_a)
                gebruikt.add(idx_b)
                break  # Ga naar volgende lijn

    merged = _merge_nabije_wanden(gevonden)
    # Sorteer op confidence, beperk tot max_resultaten
    merged.sort(key=lambda w: w.get("confidence", 0), reverse=True)
    return merged[:max_resultaten]


def _merge_nabije_wanden(wanden: list[dict], afstand: float = 150.0) -> list[dict]:
    """Voeg wanden samen die dicht bij elkaar liggen."""
    if not wanden:
        return []
    merged = []
    gebruikt = set()
    for i, w1 in enumerate(wanden):
        if i in gebruikt:
            continue
        cluster_bbox = list(w1["bbox"])
        for j, w2 in enumerate(wanden):
            if j <= i or j in gebruikt:
                continue
            cx1, cy1 = w1["center"]
            cx2, cy2 = w2["center"]
            d = math.hypot(cx1 - cx2, cy1 - cy2)
            if d < afstand:
                cluster_bbox[0] = min(cluster_bbox[0], w2["bbox"][0])
                cluster_bbox[1] = min(cluster_bbox[1], w2["bbox"][1])
                cluster_bbox[2] = max(cluster_bbox[2], w2["bbox"][2])
                cluster_bbox[3] = max(cluster_bbox[3], w2["bbox"][3])
                gebruikt.add(j)
        center = ((cluster_bbox[0] + cluster_bbox[2]) / 2,
                  (cluster_bbox[1] + cluster_bbox[3]) / 2)
        merged.append({
            "center": center,
            "bbox": tuple(cluster_bbox),
            "dikte_pt": w1["dikte_pt"],
            "confidence": w1.get("confidence", 0),
        })
    return merged


def detecteer_verdwenen_wanden(
    lijnen: list[dict],
    min_afstand: float = MIN_WAND_AFSTAND_PT,
    max_afstand: float = MAX_WAND_AFSTAND_PT,
    min_lengte: float = MIN_WAND_LENGTE_PT,
    max_resultaten: int = MAX_RESULTATEN,
    tekst_items: list[dict] | None = None,
) -> list[dict]:
    """Detecteer verdwenen wanden. Zelfde algoritme als nieuwe wanden."""
    return detecteer_wand_clusters(
        lijnen, min_afstand, max_afstand, min_lengte, max_resultaten,
        tekst_items=tekst_items,
    )

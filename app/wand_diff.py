"""
Wand-vergelijkingspipeline: detecteert bijgekomen en verdwenen wanden tussen twee
PDF-revisies via fingerprint-extractie + Hungarian matching.

Publieke API: bereken_wand_diff()
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import fitz

if TYPE_CHECKING:
    from .config import DiffConfig

# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def _is_neutraal(rgb: tuple) -> bool:
    if not rgb or len(rgb) < 3:
        return True
    r, g, b = rgb[:3]
    if all(c > 0.90 for c in (r, g, b)):
        return True
    if all(c < 0.10 for c in (r, g, b)):
        return True
    return abs(r - g) < 0.04 and abs(g - b) < 0.04 and abs(r - b) < 0.04 and 0.3 < r < 0.8


def _rnd(rgb: tuple) -> tuple:
    return tuple(round(c, 3) for c in rgb[:3])


def _kleur_afstand(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])))


# ---------------------------------------------------------------------------
# Inline Hungarian-matching (Jonker-Volgenant pad-augmentatie, pure Python)
# ---------------------------------------------------------------------------

def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """
    Minimale-cost matching. Geeft lijst van (rij, kolom) paren.
    Gebruikt Jonker-Volgenant pad-augmentatie.
    """
    INF = float("inf")
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return []

    transposed = n > m
    if transposed:
        cost = [[cost[r][c] for r in range(n)] for c in range(m)]
        n, m = m, n

    # Potentiaalvectoren
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)   # p[j] = rij gematcht aan kolom j (1-based; 0 = onbezet)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minval = [INF] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1

            for j in range(1, m + 1):
                if not used[j]:
                    c = cost[i0 - 1][j - 1] if j - 1 < len(cost[i0 - 1]) else INF
                    val = c - u[i0] - v[j]
                    if val < minval[j]:
                        minval[j] = val
                        way[j] = j0
                    if minval[j] < delta:
                        delta = minval[j]
                        j1 = j

            if j1 < 0 or delta == INF:
                break

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minval[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Resultaat
    pairs = []
    for j in range(1, m + 1):
        if p[j] > 0:
            row_idx = p[j] - 1
            col_idx = j - 1
            if transposed:
                pairs.append((col_idx, row_idx))
            else:
                pairs.append((row_idx, col_idx))

    return pairs


# ---------------------------------------------------------------------------
# Segmentextractie
# ---------------------------------------------------------------------------

def _extraheer_segmenten(page: fitz.Page, ori: dict) -> list[dict]:
    """
    Haalt gekleurde tekensegmenten op uit page.get_drawings().
    Filtert neutrale kleuren (wit, zwart, grijs) en zeer kleine of
    zeer grote elementen. Geeft voor elk segment terug:
      color, centroid, bbox, area
    in display-coördinaten (via ori["normalize"]).
    """
    normalize = ori.get("normalize", lambda x, y: (x, y))
    segs = []

    for d in page.get_drawings():
        # Gebruik fill-kleur als primaire kleur, anders stroke
        rgb = d.get("fill") or d.get("color")
        if not rgb or len(rgb) < 3:
            continue
        if _is_neutraal(rgb):
            continue

        rect = d.get("rect")
        if not rect:
            continue

        x0, y0, x1, y1 = rect
        nx0, ny0 = normalize(x0, y0)
        nx1, ny1 = normalize(x1, y1)

        # Normaliseer richting (negatieve breedte/hoogte mogelijk na rotatie)
        rx0, rx1 = min(nx0, nx1), max(nx0, nx1)
        ry0, ry1 = min(ny0, ny1), max(ny0, ny1)

        w = rx1 - rx0
        h = ry1 - ry0
        area = w * h

        # Filter te kleine deeltjes (artefacten) en te grote vlakken (achtergrond)
        if area < 4.0 or area > 500_000.0:
            continue

        cx = (rx0 + rx1) / 2
        cy = (ry0 + ry1) / 2

        segs.append({
            "color":    _rnd(rgb),
            "centroid": (cx, cy),
            "bbox":     [rx0, ry0, rx1, ry1],
            "area":     area,
        })

    return segs


# ---------------------------------------------------------------------------
# Pre-clustering (reduce arcering-segmenten naar wand-regio's)
# ---------------------------------------------------------------------------

def _pre_cluster_segs(segs: list[dict], max_afstand: float) -> list[dict]:
    """
    Groepeert nabijgelegen segmenten van dezelfde kleur via raster-BFS.
    Reduceert arceringspatronen (~1000 kleine segmenten) naar wand-regio's (~50).
    """
    if not segs:
        return []

    # Indexeer per kleur
    per_kleur: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(segs):
        per_kleur[s["color"]].append(i)

    resultaat = []
    verwerkt = [False] * len(segs)

    for kleur, indices in per_kleur.items():
        # BFS
        idx_set = set(indices)
        for start in indices:
            if verwerkt[start]:
                continue
            cluster = [start]
            verwerkt[start] = True
            frontier = [start]

            while frontier:
                huidig = frontier.pop()
                cx, cy = segs[huidig]["centroid"]
                for j in idx_set:
                    if verwerkt[j]:
                        continue
                    jx, jy = segs[j]["centroid"]
                    if abs(jx - cx) <= max_afstand and abs(jy - cy) <= max_afstand:
                        verwerkt[j] = True
                        cluster.append(j)
                        frontier.append(j)

            # Samenvoegen tot één segment
            all_bboxes = [segs[i]["bbox"] for i in cluster]
            bx0 = min(b[0] for b in all_bboxes)
            by0 = min(b[1] for b in all_bboxes)
            bx1 = max(b[2] for b in all_bboxes)
            by1 = max(b[3] for b in all_bboxes)

            resultaat.append({
                "color":    kleur,
                "centroid": ((bx0 + bx1) / 2, (by0 + by1) / 2),
                "bbox":     [bx0, by0, bx1, by1],
                "area":     (bx1 - bx0) * (by1 - by0),
            })

    return resultaat


# ---------------------------------------------------------------------------
# Cost matrix + Hungarian matching
# ---------------------------------------------------------------------------

def _bouw_cost_matrix(
    oud: list[dict],
    nieuw: list[dict],
    kleur_penalty_factor: float = 3.0,
) -> list[list[float]]:
    """
    Kostenmatrix: afstand tussen centroids.
    Kleur-mismatch → cost × penalty_factor.
    """
    matrix = []
    for o in oud:
        rij = []
        ox, oy = o["centroid"]
        for n in nieuw:
            nx, ny = n["centroid"]
            dist = math.sqrt((ox - nx) ** 2 + (oy - ny) ** 2)
            if o["color"] != n["color"]:
                dist *= kleur_penalty_factor
            rij.append(dist)
        matrix.append(rij)
    return matrix


def _hungarian_match(
    oud: list[dict],
    nieuw: list[dict],
    max_cost: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Past Hungarian matching toe en filtert paren boven max_cost.
    Geeft terug: (gematcht, oud_ongemat, nieuw_ongemat).
    """
    if not oud or not nieuw:
        return [], list(range(len(oud))), list(range(len(nieuw)))

    matrix = _bouw_cost_matrix(oud, nieuw)
    alle_paren = _hungarian(matrix)

    gematcht = []
    oud_gemat = set()
    nieuw_gemat = set()

    for r, c in alle_paren:
        if r < len(oud) and c < len(nieuw):
            cost = matrix[r][c]
            if cost <= max_cost:
                gematcht.append((r, c))
                oud_gemat.add(r)
                nieuw_gemat.add(c)

    oud_ongemat  = [i for i in range(len(oud))  if i not in oud_gemat]
    nieuw_ongemat = [i for i in range(len(nieuw)) if i not in nieuw_gemat]

    return gematcht, oud_ongemat, nieuw_ongemat


# ---------------------------------------------------------------------------
# Wandtype-lookup
# ---------------------------------------------------------------------------

def _match_wandtype(seg: dict, legenda: dict, kleur_tol: float) -> str:
    """
    Zoekt dichtstbijzijnde kleur in legenda (RGB-dict → wandtype-naam).
    Geeft lege string als geen match binnen tolerantie.
    """
    if not legenda:
        return ""
    kleur = seg.get("color")
    if not kleur:
        return ""
    beste_naam = ""
    beste_afstand = float("inf")
    for leg_kleur, naam in legenda.items():
        try:
            afstand = _kleur_afstand(kleur, leg_kleur)
        except Exception:
            continue
        if afstand < beste_afstand:
            beste_afstand = afstand
            beste_naam = naam
    if beste_afstand <= kleur_tol:
        return beste_naam
    return ""


# ---------------------------------------------------------------------------
# Cluster eindresultaten
# ---------------------------------------------------------------------------

def _cluster_segmenten(
    pre_cluster: list[dict],
    cluster_afstand: float,
) -> list[dict]:
    """
    Groepeert nabijgelegen wijzigingen van hetzelfde type tot één resultaat.
    Positie = centroid van de cluster.
    """
    if not pre_cluster:
        return []

    # Groepeer per type
    per_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(pre_cluster):
        per_type[r["type"]].append(i)

    resultaat = []
    verwerkt = [False] * len(pre_cluster)

    for type_naam, indices in per_type.items():
        for start in indices:
            if verwerkt[start]:
                continue
            cluster = [start]
            verwerkt[start] = True
            frontier = [start]

            while frontier:
                huidig = frontier.pop()
                hx, hy = pre_cluster[huidig]["positie"]
                for j in indices:
                    if verwerkt[j]:
                        continue
                    jx, jy = pre_cluster[j]["positie"]
                    if math.sqrt((jx - hx) ** 2 + (jy - hy) ** 2) <= cluster_afstand:
                        verwerkt[j] = True
                        cluster.append(j)
                        frontier.append(j)

            # Representatieve positie = centroid
            xs = [pre_cluster[i]["positie"][0] for i in cluster]
            ys = [pre_cluster[i]["positie"][1] for i in cluster]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # Bbox = omsluitend rechthoek van alle segmenten in cluster
            all_bboxes = [pre_cluster[i]["bbox"] for i in cluster]
            bx0 = min(b[0] for b in all_bboxes)
            by0 = min(b[1] for b in all_bboxes)
            bx1 = max(b[2] for b in all_bboxes)
            by1 = max(b[3] for b in all_bboxes)

            # Wandtype = meest voorkomende in cluster
            types_wand = [pre_cluster[i].get("wandtype", "") for i in cluster]
            wandtype = max(set(types_wand), key=types_wand.count) if types_wand else ""

            resultaat.append({
                "type":     type_naam,
                "wandtype": wandtype,
                "positie":  [cx, cy],
                "bbox":     [bx0, by0, bx1, by1],
            })

    return resultaat


# ---------------------------------------------------------------------------
# Publieke API
# ---------------------------------------------------------------------------

def bereken_wand_diff(
    oud_page:   fitz.Page,
    nieuw_page: fitz.Page,
    oud_ori:    dict,
    nieuw_ori:  dict,
    legenda:    dict,
    api_key:    str | None = None,
    cfg:        "DiffConfig | None" = None,
) -> list[dict]:
    """
    Vergelijkt wandsegmenten tussen twee PDF-paginarevisies.

    Parameters
    ----------
    oud_page / nieuw_page : geopende fitz.Page objecten
    oud_ori / nieuw_ori   : oriëntatie-dicts van detecteer_orientatie()
    legenda               : RGB-tuple -> wandtype-naam (van vind_legenda_combined)
    api_key               : gereserveerd (niet gebruikt in deze versie)
    cfg                   : DiffConfig instance (defaults worden gebruikt bij None)

    Returns
    -------
    Lijst van dicts met: type, wandtype, kleur, positie, bbox (display-coördinaten)
    """
    max_cost        = cfg.wand_centroid_max_afstand  if cfg else 150.0
    kleur_tol       = cfg.wand_kleur_tolerantie      if cfg else 0.15
    cluster_afstand = cfg.wand_cluster_afstand       if cfg else 80.0
    pre_afs         = cfg.wand_pre_cluster_afstand   if cfg else 40.0

    oud_segs  = _extraheer_segmenten(oud_page,  oud_ori)
    nieuw_segs = _extraheer_segmenten(nieuw_page, nieuw_ori)

    # Pre-clustering reduceert arcering-segmenten tot wand-regio's
    oud_segs  = _pre_cluster_segs(oud_segs,  pre_afs)
    nieuw_segs = _pre_cluster_segs(nieuw_segs, pre_afs)

    _, oud_ongemat, nieuw_ongemat = _hungarian_match(oud_segs, nieuw_segs, max_cost)

    pre_cluster: list[dict] = []

    for idx in nieuw_ongemat:
        seg = nieuw_segs[idx]
        wandtype = _match_wandtype(seg, legenda, kleur_tol)
        pre_cluster.append({
            "type":     "nieuw",
            "wandtype": wandtype,
            "positie":  list(seg["centroid"]),
            "bbox":     seg["bbox"],
        })

    for idx in oud_ongemat:
        seg = oud_segs[idx]
        wandtype = _match_wandtype(seg, legenda, kleur_tol)
        pre_cluster.append({
            "type":     "verdwenen",
            "wandtype": wandtype,
            "positie":  list(seg["centroid"]),
            "bbox":     seg["bbox"],
        })

    if not pre_cluster:
        return []

    resultaten = _cluster_segmenten(pre_cluster, cluster_afstand)

    for r in resultaten:
        r["kleur"] = None

    return resultaten

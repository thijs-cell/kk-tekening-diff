"""
Configuratie voor K&K tekening-diff.

Centrale plek voor alle drempelwaarden en parameters.
Defaults komen overeen met het huidige (werkende) gedrag.
"""

from dataclasses import dataclass


@dataclass
class DiffConfig:
    """Alle configureerbare parameters voor de diff pipeline."""

    # --- Fallback zone ratios (alleen als auto-detect faalt) ---
    renvooi_x_ratio: float = 0.88
    koptekst_y_ratio: float = 0.05

    # --- Matching drempels ---
    tekst_match_drempel: float = 20.0
    lijn_match_drempel: float = 5.0
    fill_match_drempel: float = 10.0

    # --- Wall detect ---
    min_wand_afstand_pt: float = 1.8
    max_wand_afstand_pt: float = 12.0
    min_wand_lengte_pt: float = 30.0
    max_wand_lengte_pt: float = 800.0
    max_wand_resultaten: int = 50

    # --- Legenda parser ---
    kleur_tolerantie: float = 0.10
    fill_min_area: float = 5.0
    fill_max_area: float = 900.0
    min_fill_oppervlakte: float = 100.0

    # --- Lijn vergelijking ---
    lijn_width_verschil: float = 0.5

    # --- wand_diff pipeline ---
    use_new_wand_diff: bool = True            # False → oud vergelijk_wanden() pad
    use_vision_pipeline: bool = False         # True → wand_diff_vision.py (experimental)
    vision_per_segment_actief: bool = False   # True → Claude Vision per onbekend segment (v2)
    wand_min_segment_lengte: float = 5.0      # pt — kleinste te detecteren segment
    wand_centroid_max_afstand: float = 150.0  # pt — kostdrempel Hungarian verwerping
    wand_kleur_tolerantie: float = 0.15       # RGB Euclidisch voor kleurmatch
    wand_arcering_hoek_tol: float = 5.0       # graden — arceringhoek tolerantie (v2)
    wand_arcering_spatie_tol: float = 2.0     # pt — arceringspatie tolerantie (v2)
    wand_vision_confidence_min: float = 0.65  # min Vision confidence (v2)
    wand_cluster_afstand: float = 80.0        # pt — clustering drempel
    wand_oval_min_dim: float = 8.0            # pt — minimale ellipse-dimensie
    wand_pre_cluster_afstand: float = 40.0   # pt — pre-match clustering (reduceert arcering-segmenten)

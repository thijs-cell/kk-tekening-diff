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
    tekst_match_drempel: float = 15.0
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

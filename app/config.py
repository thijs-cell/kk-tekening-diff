"""Configuratie via environment variables met standaardwaarden."""

import os


def _bool_env(key: str, default: bool) -> bool:
    """Lees een boolean environment variable met fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _int_env(key: str, default: int) -> int:
    """Lees een integer environment variable met fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    """Lees een float environment variable met fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# PDF-naar-afbeelding resolutie
DPI: int = _int_env("DPI", 200)

# Drempelwaarde voor verschildetectie (0-255)
SENSITIVITY: int = _int_env("SENSITIVITY", 15)

# Percentage van de rechterkant dat gemaskeerd wordt (titelblok)
TITLE_BLOCK_MASK_PERCENT: float = _float_env("TITLE_BLOCK_MASK_PERCENT", 12.0)

# Maximale bestandsgrootte in MB
MAX_FILE_SIZE_MB: int = _int_env("MAX_FILE_SIZE_MB", 15)

# ORB feature detection parameters
ORB_FEATURES: int = _int_env("ORB_FEATURES", 5000)
RANSAC_THRESHOLD: float = _float_env("RANSAC_THRESHOLD", 5.0)
MIN_GOOD_MATCHES: int = _int_env("MIN_GOOD_MATCHES", 10)

# Overlay parameters
OVERLAY_ALPHA: float = _float_env("OVERLAY_ALPHA", 0.3)
BBOX_LINE_THICKNESS: int = _int_env("BBOX_LINE_THICKNESS", 3)
BBOX_PADDING: int = _int_env("BBOX_PADDING", 5)
MIN_CONTOUR_AREA: int = _int_env("MIN_CONTOUR_AREA", 100)

# Morphologische kernel
MORPH_KERNEL_SIZE: int = _int_env("MORPH_KERNEL_SIZE", 3)
MORPH_ITERATIONS: int = _int_env("MORPH_ITERATIONS", 2)

# AI interpretatie (Claude Vision)
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ENABLE_AI_INTERPRETATION: bool = _bool_env("ENABLE_AI_INTERPRETATION", True)

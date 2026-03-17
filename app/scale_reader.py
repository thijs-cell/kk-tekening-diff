"""Lees de schaal uit het titelblok van een bouwtekening via Claude Vision."""

import base64
import io
import logging
import time

import anthropic
from PIL import Image

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

SCALE_PROMPT = (
    "Lees de schaal van deze bouwtekening. Het staat in het titelblok als "
    "SCHAAL gevolgd door een getal zoals 1:50 of 1:100. "
    "Antwoord alleen met het getal na de dubbele punt, bijvoorbeeld 50. "
    "Alleen het getal, niks anders."
)

SCALE_DPI = 300
TITLEBLOCK_RIGHT_FRACTION = 0.15
TITLEBLOCK_BOTTOM_FRACTION = 0.25


def _crop_title_block(pil_img: Image.Image) -> Image.Image:
    """Crop het titelblok (rechter 15% + onderste 25%) uit de afbeelding."""
    w, h = pil_img.size
    left = int(w * (1.0 - TITLEBLOCK_RIGHT_FRACTION))
    top = int(h * (1.0 - TITLEBLOCK_BOTTOM_FRACTION))
    return pil_img.crop((left, top, w, h))


def _pil_to_base64_png(pil_img: Image.Image) -> str:
    """Encodeer PIL Image naar base64 PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_scale(pil_img: Image.Image) -> int:
    """
    Lees de schaal uit het titelblok van een bouwtekening.

    Parameters:
        pil_img: PIL Image van een pagina (gerenderd op SCALE_DPI).

    Returns:
        Schaal als integer (bijv. 50 voor 1:50). Fallback: 50.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("Geen ANTHROPIC_API_KEY, fallback schaal 50")
        return 50

    title_crop = _crop_title_block(pil_img)
    crop_b64 = _pil_to_base64_png(title_crop)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    start = time.time()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=32,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": crop_b64,
                            },
                        },
                        {"type": "text", "text": SCALE_PROMPT},
                    ],
                }
            ],
            timeout=20.0,
        )

        elapsed = time.time() - start
        text = response.content[0].text.strip()
        logger.info("Schaal gelezen in %.1fs: '%s'", elapsed, text)

        # Parse het getal
        digits = "".join(c for c in text if c.isdigit())
        if digits:
            scale = int(digits)
            if scale in (10, 20, 25, 50, 100, 200, 500):
                return scale
            logger.warning("Onverwachte schaal %d, fallback 50", scale)
            return 50

        logger.warning("Kan schaal niet parsen uit '%s', fallback 50", text)
        return 50

    except Exception as e:
        elapsed = time.time() - start
        logger.error("Schaal lezen mislukt na %.1fs: %s", elapsed, str(e))
        return 50


def calculate_pixels_per_mm(dpi: int, scale: int) -> float:
    """
    Bereken hoeveel pixels er in 1 mm werkelijkheid zitten.

    Bij DPI=200 en schaal 1:50: 200 / (25.4 * 50) = 0.157 px/mm.
    """
    return dpi / (25.4 * scale)

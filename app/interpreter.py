"""AI-interpretatie via strook-gebaseerde vergelijking met Claude Vision API."""

import base64
import io
import json
import logging
import time
from typing import Any

import anthropic
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    MAX_STRIP_IMAGE_BYTES,
    MIN_CONTOUR_AREA,
    NUM_STRIPS,
)

logger = logging.getLogger(__name__)

CHANGE_TYPES = (
    "WANDDIKTE", "MAATVOERING", "KOZIJNHOOGTE",
    "WANDVERSCHUIVING", "INDELING", "SPARING", "MATERIAALCODE",
)


def _build_vision_prompt(scale: int, verschuivingen_tekst: str) -> str:
    """Bouw de Claude Vision prompt voor een strook-vergelijking."""
    return (
        "Je analyseert twee stroken van een demarcatietekening voor gibowanden. "
        "Boven staat de OUDE versie, onder de NIEUWE versie. "
        f"De tekening heeft schaal 1:{scale}.\n\n"
        "CONTEXT OVER DEZE TEKENINGEN:\n"
        "- Wanden zijn dubbele parallelle lijnen met arcering ertussen. "
        "De afstand tussen de lijnen geeft de wanddikte.\n"
        "- Wandreferenties staan naast wanden als 'grw=400' of 'grw=480' "
        "(grw = gibowand, getal = dikte in mm).\n"
        "- Maatvoering staat langs maatlijnen in millimeters "
        "(bijv. 2375, 1034, 530).\n"
        "- Ruimtes hebben een code, naam en oppervlakte: "
        "bijv. '0.B09 Buitenberging 5.1 m\u00b2'.\n"
        "- Kozijnen hebben hoogte-aanduidingen, kritiek bereik is 2060-2364mm.\n"
        "- Materiaalcodes: GNL70, GZL70, GHL70, GHL50, GHL100.\n"
        "- Rode driehoekige pijlen zijn revisie-markeringen van de tekenaar.\n\n"
        "OpenCV heeft de volgende verschuivingen gemeten in dit gebied:\n"
        f"{verschuivingen_tekst}\n\n"
        "Zoek ALLEEN naar deze 7 typen wijzigingen, NEGEER al het andere:\n"
        "1. WANDDIKTE: is een 'grw=' waarde veranderd? Of zien de dubbele "
        "wandlijnen er dikker/dunner uit?\n"
        "2. MAATVOERING: is een getal op een maatlijn veranderd?\n"
        "3. KOZIJNHOOGTE: is een getal bij een kozijn veranderd "
        "(bereik 2060-2364mm)?\n"
        "4. WANDVERSCHUIVING: is een wand van positie veranderd? "
        "OpenCV heeft de verschuiving al gemeten, bevestig of ontken dit visueel.\n"
        "5. INDELING: is een kamer/ruimte toegevoegd, verwijderd of anders "
        "ingedeeld?\n"
        "6. SPARING: is een sparing (opening in wand) toegevoegd, verwijderd "
        "of verplaatst?\n"
        "7. MATERIAALCODE: is een code (GNL, GZL, GHL + getal) veranderd?\n\n"
        "Als je GEEN van deze 7 wijzigingen ziet in dit gebied, antwoord dan "
        "met een leeg changes array.\n\n"
        "Antwoord in JSON:\n"
        "{\n"
        '  "changes": [\n'
        "    {\n"
        '      "type": "WANDDIKTE" of "MAATVOERING" of "KOZIJNHOOGTE" of '
        '"WANDVERSCHUIVING" of "INDELING" of "SPARING" of "MATERIAALCODE",\n'
        '      "description": "beschrijving in het Nederlands, max 2 zinnen. '
        'Bij verschuiving: gebruik de OpenCV-gemeten waarde in mm.",\n'
        '      "location": "waar op de strook, bijv. \'linkerdeel bij ruimte '
        "0.B09' of 'midden bij kozijn H14'\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Alleen JSON, geen andere tekst."
    )


def _strip_has_changes(diff_mask: np.ndarray, y_start: int, y_end: int) -> bool:
    """Check of een strook wijzigingen bevat in het diff masker."""
    strip = diff_mask[y_start:y_end, :]
    return int(np.count_nonzero(strip)) > 0


def _get_displacements_in_strip(
    displacements: list[dict[str, Any]],
    y_start: int,
    y_end: int,
) -> list[dict[str, Any]]:
    """Filter verschuivingen die binnen een strook vallen."""
    result = []
    for d in displacements:
        cy = d["centroid"][1]
        if y_start <= cy < y_end:
            result.append(d)
    return result


def _format_displacements(strip_displacements: list[dict[str, Any]]) -> str:
    """Formatteer verschuivingsdata als leesbare tekst voor de prompt."""
    if not strip_displacements:
        return "Geen verschuivingen gemeten in dit gebied."

    lines = []
    for i, d in enumerate(strip_displacements):
        x, y = d["centroid"]
        if d["verschuiving_mm"] is not None:
            lines.append(
                f"- Positie ({x}, {y}): verschuiving van {d['verschuiving_mm']}mm "
                f"({d['verschuiving_px']}px)"
            )
        else:
            lines.append(
                f"- Positie ({x}, {y}): nieuwe toevoeging of verwijdering "
                f"(geen overeenkomstig element gevonden)"
            )
    return "\n".join(lines)


def _create_strip_comparison(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    y_start: int,
    y_end: int,
) -> bytes:
    """
    Maak een vergelijkingsafbeelding: oud boven, nieuw onder met labels.

    Returns:
        PNG bytes van de gecombineerde strook.
    """
    old_strip = old_bgr[y_start:y_end, :]
    new_strip = new_bgr[y_start:y_end, :]

    strip_h, strip_w = old_strip.shape[:2]
    label_h = 40
    separator_h = 4

    total_h = label_h + strip_h + separator_h + label_h + strip_h
    total_w = strip_w

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Label "OUD"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "OUD", (10, 28), font, 0.9, (0, 0, 200), 2)
    cv2.line(canvas, (0, label_h - 2), (total_w, label_h - 2), (180, 180, 180), 1)

    # Oude strook
    canvas[label_h:label_h + strip_h, 0:strip_w] = old_strip

    # Scheidingslijn
    sep_y = label_h + strip_h
    canvas[sep_y:sep_y + separator_h, :] = (100, 100, 100)

    # Label "NIEUW"
    new_label_y = sep_y + separator_h
    cv2.putText(canvas, "NIEUW", (10, new_label_y + 28), font, 0.9, (0, 150, 0), 2)
    cv2.line(
        canvas,
        (0, new_label_y + label_h - 2),
        (total_w, new_label_y + label_h - 2),
        (180, 180, 180),
        1,
    )

    # Nieuwe strook
    new_strip_y = new_label_y + label_h
    canvas[new_strip_y:new_strip_y + strip_h, 0:strip_w] = new_strip

    # Encodeer als PNG
    success, buffer = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("PNG encoding mislukt voor strook")
    return buffer.tobytes()


def _downscale_to_fit(png_bytes: bytes, max_bytes: int) -> bytes:
    """Verklein een PNG afbeelding tot het binnen max_bytes past."""
    if len(png_bytes) <= max_bytes:
        return png_bytes

    img = Image.open(io.BytesIO(png_bytes))
    quality_steps = [85, 70, 55, 40]

    for quality in quality_steps:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            return buf.getvalue()

    # Als JPEG niet klein genoeg is, verklein de afbeelding
    scale = 0.75
    while scale > 0.2:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=60)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
        scale -= 0.1

    # Laatste poging: heel klein
    resized = img.resize((img.width // 4, img.height // 4), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=50)
    return buf.getvalue()


def _call_claude_vision(
    client: anthropic.Anthropic,
    image_bytes: bytes,
    prompt: str,
    page_num: int,
    strip_idx: int,
) -> list[dict[str, str]]:
    """
    Stuur een strook-vergelijking naar Claude Vision API en parse de response.

    Returns:
        Lijst van {"type": ..., "description": ..., "location": ...} dicts.
    """
    # Bepaal media type
    if image_bytes[:3] == b"\xff\xd8\xff":
        media_type = "image/jpeg"
    else:
        media_type = "image/png"

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    start = time.time()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            timeout=60.0,
        )

        elapsed = time.time() - start
        usage = response.usage
        logger.info(
            "AI call pagina %d strook %d: %.1fs, %d in / %d out tokens",
            page_num, strip_idx, elapsed,
            usage.input_tokens, usage.output_tokens,
        )

        # Parse JSON uit response
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        changes = data.get("changes", [])

        # Valideer structuur
        valid: list[dict[str, str]] = []
        for c in changes:
            change_type = c.get("type", "").upper()
            if change_type not in CHANGE_TYPES:
                continue
            if "description" not in c:
                continue
            valid.append({
                "type": change_type,
                "description": c["description"],
                "location": c.get("location", ""),
            })
        return valid

    except json.JSONDecodeError as e:
        logger.warning(
            "AI response pagina %d strook %d: ongeldige JSON: %s",
            page_num, strip_idx, str(e),
        )
        return []
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "AI call mislukt pagina %d strook %d na %.1fs: %s",
            page_num, strip_idx, elapsed, str(e),
        )
        return []


def interpret_page(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    diff_mask: np.ndarray,
    page_num: int,
    scale: int = 50,
    pixels_per_mm: float = 0.157,
    displacements: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Interpreteer wijzigingen per strook via Claude Vision.

    Splits de pagina in NUM_STRIPS horizontale stroken. Alleen stroken
    met wijzigingen worden naar Claude gestuurd.

    Parameters:
        old_bgr: Oude afbeelding in BGR formaat (aligned).
        new_bgr: Nieuwe afbeelding in BGR formaat.
        diff_mask: Binair masker van gedetecteerde wijzigingen.
        page_num: Paginanummer (1-gebaseerd).
        scale: Schaal van de tekening (bijv. 50).
        pixels_per_mm: Pixels per mm werkelijkheid.
        displacements: Lijst van berekende verschuivingen.

    Returns:
        Lijst van interpretatie dictionaries met type, description, location.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY niet geconfigureerd, interpretatie overgeslagen")
        return []

    if displacements is None:
        displacements = []

    h, w = diff_mask.shape[:2]
    strip_height = h // NUM_STRIPS

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    all_changes: list[dict[str, Any]] = []

    for strip_idx in range(NUM_STRIPS):
        y_start = strip_idx * strip_height
        y_end = h if strip_idx == NUM_STRIPS - 1 else (strip_idx + 1) * strip_height

        # Check of er wijzigingen in deze strook zitten
        if not _strip_has_changes(diff_mask, y_start, y_end):
            logger.info(
                "Pagina %d strook %d/%d: geen wijzigingen, overgeslagen",
                page_num, strip_idx + 1, NUM_STRIPS,
            )
            continue

        # Verschuivingen in deze strook
        strip_displacements = _get_displacements_in_strip(
            displacements, y_start, y_end
        )
        verschuivingen_tekst = _format_displacements(strip_displacements)

        logger.info(
            "Pagina %d strook %d/%d: wijzigingen gevonden, "
            "%d verschuivingen, stuur naar AI",
            page_num, strip_idx + 1, NUM_STRIPS, len(strip_displacements),
        )

        # Maak vergelijkingsafbeelding (oud boven, nieuw onder)
        try:
            png_bytes = _create_strip_comparison(
                old_bgr, new_bgr, y_start, y_end
            )
        except Exception as e:
            logger.error(
                "Strook-afbeelding mislukt pagina %d strook %d: %s",
                page_num, strip_idx + 1, str(e),
            )
            continue

        # Verklein als nodig
        image_bytes = _downscale_to_fit(png_bytes, MAX_STRIP_IMAGE_BYTES)
        logger.info(
            "Pagina %d strook %d: afbeelding %d KB%s",
            page_num, strip_idx + 1, len(image_bytes) // 1024,
            " (verkleind)" if len(image_bytes) < len(png_bytes) else "",
        )

        # Bouw prompt en stuur naar Claude
        prompt = _build_vision_prompt(scale, verschuivingen_tekst)
        changes = _call_claude_vision(
            client, image_bytes, prompt, page_num, strip_idx + 1
        )

        for change in changes:
            all_changes.append({
                "type": change["type"],
                "description": change["description"],
                "location": change["location"],
                "strip": strip_idx + 1,
            })

    logger.info(
        "Pagina %d: %d wijzigingen gevonden over alle stroken",
        page_num, len(all_changes),
    )

    return all_changes

"""AI-interpretatie via blok-gebaseerde vergelijking met Claude Vision API."""

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
    GRID_COLS,
    GRID_ROWS,
    MAX_BLOCK_IMAGE_BYTES,
    MIN_CONTOUR_AREA,
)

logger = logging.getLogger(__name__)

CHANGE_TYPES = (
    "WANDDIKTE", "MAATVOERING", "KOZIJNHOOGTE",
    "WANDVERSCHUIVING", "INDELING", "SPARING", "MATERIAALCODE",
)


def _build_vision_prompt(scale: int, verschuivingen_tekst: str) -> str:
    """Bouw de Claude Vision prompt voor een blok-vergelijking."""
    return (
        "Je analyseert twee blokken van een demarcatietekening voor gibowanden. "
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
        '      "location": "waar op het blok, bijv. \'linkerdeel bij ruimte '
        "0.B09' of 'midden bij kozijn H14'\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Alleen JSON, geen andere tekst."
    )


def _block_has_changes(
    diff_mask: np.ndarray,
    y_start: int, y_end: int,
    x_start: int, x_end: int,
) -> bool:
    """Check of een blok wijzigingen bevat in het diff masker."""
    block = diff_mask[y_start:y_end, x_start:x_end]
    return int(np.count_nonzero(block)) > 0


def _get_displacements_in_block(
    displacements: list[dict[str, Any]],
    y_start: int, y_end: int,
    x_start: int, x_end: int,
) -> list[dict[str, Any]]:
    """Filter verschuivingen die binnen een blok vallen."""
    result = []
    for d in displacements:
        cx, cy = d["centroid"]
        if y_start <= cy < y_end and x_start <= cx < x_end:
            result.append(d)
    return result


def _format_displacements(block_displacements: list[dict[str, Any]]) -> str:
    """Formatteer verschuivingsdata als leesbare tekst voor de prompt."""
    if not block_displacements:
        return "Geen verschuivingen gemeten in dit gebied."

    lines = []
    for i, d in enumerate(block_displacements):
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


def _create_block_comparison(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    y_start: int, y_end: int,
    x_start: int, x_end: int,
) -> bytes:
    """
    Maak een vergelijkingsafbeelding: oud boven, nieuw onder met labels.

    Returns:
        PNG bytes van het gecombineerde blok.
    """
    old_block = old_bgr[y_start:y_end, x_start:x_end]
    new_block = new_bgr[y_start:y_end, x_start:x_end]

    block_h, block_w = old_block.shape[:2]
    label_h = 40
    separator_h = 4

    total_h = label_h + block_h + separator_h + label_h + block_h
    total_w = block_w

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Label "OUD"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "OUD", (10, 28), font, 0.9, (0, 0, 200), 2)
    cv2.line(canvas, (0, label_h - 2), (total_w, label_h - 2), (180, 180, 180), 1)

    # Oud blok
    canvas[label_h:label_h + block_h, 0:block_w] = old_block

    # Scheidingslijn
    sep_y = label_h + block_h
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

    # Nieuw blok
    new_block_y = new_label_y + label_h
    canvas[new_block_y:new_block_y + block_h, 0:block_w] = new_block

    # Encodeer als PNG
    success, buffer = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("PNG encoding mislukt voor blok")
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
    row: int,
    col: int,
) -> list[dict[str, str]]:
    """
    Stuur een blok-vergelijking naar Claude Vision API en parse de response.

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
            "AI call pagina %d blok r%d/c%d: %.1fs, %d in / %d out tokens",
            page_num, row, col, elapsed,
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
            "AI response pagina %d blok r%d/c%d: ongeldige JSON: %s",
            page_num, row, col, str(e),
        )
        return []
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "AI call mislukt pagina %d blok r%d/c%d na %.1fs: %s",
            page_num, row, col, elapsed, str(e),
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
    Interpreteer wijzigingen per blok via Claude Vision.

    Splits de pagina in een raster van GRID_COLS x GRID_ROWS blokken.
    Alleen blokken met wijzigingen worden naar Claude gestuurd.

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
    block_h = h // GRID_ROWS
    block_w = w // GRID_COLS

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    all_changes: list[dict[str, Any]] = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y_start = row * block_h
            y_end = h if row == GRID_ROWS - 1 else (row + 1) * block_h
            x_start = col * block_w
            x_end = w if col == GRID_COLS - 1 else (col + 1) * block_w

            # Check of er wijzigingen in dit blok zitten
            if not _block_has_changes(diff_mask, y_start, y_end, x_start, x_end):
                logger.info(
                    "Pagina %d blok r%d/c%d: geen wijzigingen, overgeslagen",
                    page_num, row + 1, col + 1,
                )
                continue

            # Verschuivingen in dit blok
            block_displacements = _get_displacements_in_block(
                displacements, y_start, y_end, x_start, x_end,
            )
            verschuivingen_tekst = _format_displacements(block_displacements)

            logger.info(
                "Pagina %d blok r%d/c%d: wijzigingen gevonden, "
                "%d verschuivingen, stuur naar AI",
                page_num, row + 1, col + 1, len(block_displacements),
            )

            # Maak vergelijkingsafbeelding (oud boven, nieuw onder)
            try:
                png_bytes = _create_block_comparison(
                    old_bgr, new_bgr, y_start, y_end, x_start, x_end,
                )
            except Exception as e:
                logger.error(
                    "Blok-afbeelding mislukt pagina %d blok r%d/c%d: %s",
                    page_num, row + 1, col + 1, str(e),
                )
                continue

            # Verklein als nodig
            image_bytes = _downscale_to_fit(png_bytes, MAX_BLOCK_IMAGE_BYTES)
            logger.info(
                "Pagina %d blok r%d/c%d: afbeelding %d KB%s",
                page_num, row + 1, col + 1, len(image_bytes) // 1024,
                " (verkleind)" if len(image_bytes) < len(png_bytes) else "",
            )

            # Bouw prompt en stuur naar Claude
            prompt = _build_vision_prompt(scale, verschuivingen_tekst)
            changes = _call_claude_vision(
                client, image_bytes, prompt, page_num, row + 1, col + 1,
            )

            for change in changes:
                all_changes.append({
                    "type": change["type"],
                    "description": change["description"],
                    "location": change["location"],
                    "row": row + 1,
                    "col": col + 1,
                })

    logger.info(
        "Pagina %d: %d wijzigingen gevonden over alle blokken",
        page_num, len(all_changes),
    )

    return all_changes

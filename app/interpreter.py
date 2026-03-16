"""AI-interpretatie van gedetecteerde wijzigingen via Claude Vision API."""

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

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MIN_CONTOUR_AREA

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "Je bent een expert in het lezen van demarcatietekeningen voor gibowanden "
    "bij woningbouwprojecten. Je krijgt twee uitsnedes van dezelfde locatie op "
    "een plattegrond: links de oude versie, rechts de nieuwe versie.\n\n"
    "Analyseer de verschillen met deze prioriteiten:\n\n"
    "KRITIEK (direct impact op uitvoering, ALTIJD melden):\n"
    "- Wanddiktes: herken visueel of een wand dikker of dunner is geworden, "
    "ook als er geen maat bij staat. Typische diktes zijn 50, 70 en 100mm. "
    "Een dikkere lijn betekent een dikkere wand.\n"
    "- Maatvoering: elk gewijzigd getal is kritiek. Let specifiek op "
    "afmetingen in millimeters.\n"
    "- Kozijnhoogtes: getallen tussen 2060 en 2364mm bepalen of er giebouw "
    "boven een kozijn komt. Een verschil van zelfs 10mm kan de bouwkeuze "
    "veranderen.\n"
    "- Wandverschuivingen: als een wand van positie is veranderd.\n"
    "- Indelingswijzigingen: toegevoegde of verwijderde kamers, muren, "
    "badkamer of slaapkamer aanpassingen.\n"
    "- Sparingen: toegevoegd, verwijderd of verplaatst.\n"
    "- Materiaalcodes: wijzigingen in codes zoals GNL70, GZL70, GHL70, "
    "GHL50, GHL100.\n\n"
    "NEGEREN (niet vermelden, overslaan):\n"
    "- Verschoven labels of tekst zonder inhoudelijke wijziging.\n"
    "- Gewijzigde arceringen of kleurcoderingen zonder bouwkundige impact.\n"
    "- Andere annotaties.\n"
    "- Minimale pixelverschillen zonder zichtbare bouwkundige wijziging.\n"
    "- Render artefacten of lichte kleurverschillen.\n"
    "- Revisienummers, datums, titelblok wijzigingen.\n\n"
    "BELANGRIJK:\n"
    "- Als je een wanddikte verschil ziet maar geen exact getal kunt lezen, "
    "beschrijf dan wat je visueel ziet (bijv. 'wand lijkt dunner geworden, "
    "mogelijk van 100 naar 70mm').\n"
    "- Kleine getalverschillen zijn ALTIJD kritiek bij maatvoering en "
    "kozijnhoogtes.\n"
    "- Als je niet kunt bepalen wat er veranderd is, zeg dat eerlijk en "
    "classificeer als KRITIEK met een vraagteken.\n"
    "- Wees liever te voorzichtig dan dat je iets mist. Bij twijfel: "
    "KRITIEK.\n\n"
    'Antwoord in JSON:\n'
    '{\n'
    '  "changes": [\n'
    '    {\n'
    '      "description": "beschrijving in het Nederlands, max 2 zinnen",\n'
    '      "classification": "KRITIEK" of "NEGEREN"\n'
    '    }\n'
    '  ]\n'
    '}\n'
    'Alleen JSON, geen andere tekst.'
)

MAX_INTERPRETATIONS_PER_PAGE = 20
CLUSTER_DISTANCE_PX = 100
MIN_CLUSTER_AREA = 200
CROP_PADDING = 500
AI_TIMEOUT = 30.0


def _cluster_bounding_boxes(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Groepeer bounding boxes die dicht bij elkaar liggen tot clusters.

    Parameters:
        bboxes: Lijst van (x, y, w, h) tuples.

    Returns:
        Lijst van geclusterde (x, y, w, h) tuples.
    """
    if not bboxes:
        return []

    # Converteer naar (x1, y1, x2, y2)
    rects = [(x, y, x + w, y + h) for x, y, w, h in bboxes]
    merged = True

    while merged:
        merged = False
        new_rects: list[tuple[int, int, int, int]] = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, x2, y2 = rects[i]

            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                jx1, jy1, jx2, jy2 = rects[j]

                # Check of de afstand kleiner is dan CLUSTER_DISTANCE_PX
                dx = max(0, max(x1, jx1) - min(x2, jx2))
                dy = max(0, max(y1, jy1) - min(y2, jy2))

                if dx <= CLUSTER_DISTANCE_PX and dy <= CLUSTER_DISTANCE_PX:
                    x1 = min(x1, jx1)
                    y1 = min(y1, jy1)
                    x2 = max(x2, jx2)
                    y2 = max(y2, jy2)
                    used[j] = True
                    merged = True

            new_rects.append((x1, y1, x2, y2))
            used[i] = True

        rects = new_rects

    # Terug naar (x, y, w, h) en filter kleine clusters
    result = []
    for x1, y1, x2, y2 in rects:
        w = x2 - x1
        h = y2 - y1
        if w * h >= MIN_CLUSTER_AREA:
            result.append((x1, y1, w, h))

    # Sorteer op grootte (grootste eerst) en limiteer
    result.sort(key=lambda r: r[2] * r[3], reverse=True)
    return result[:MAX_INTERPRETATIONS_PER_PAGE]


def _create_side_by_side_crop(
    old_img: np.ndarray,
    new_img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> str:
    """
    Maak een crop uit beide afbeeldingen en plak ze naast elkaar met labels.

    Returns:
        Base64-encoded PNG string van de gecombineerde crop.
    """
    img_h, img_w = new_img.shape[:2]

    # Crop coördinaten met padding
    cx1 = max(0, x - CROP_PADDING)
    cy1 = max(0, y - CROP_PADDING)
    cx2 = min(img_w, x + w + CROP_PADDING)
    cy2 = min(img_h, y + h + CROP_PADDING)

    # Pas oude afbeelding aan als die kleiner is
    old_h, old_w = old_img.shape[:2]
    old_cx2 = min(old_w, cx2)
    old_cy2 = min(old_h, cy2)

    old_crop = old_img[cy1:old_cy2, cx1:old_cx2]
    new_crop = new_img[cy1:cy2, cx1:cx2]

    # Zorg dat beide crops dezelfde hoogte hebben
    crop_h = max(old_crop.shape[0], new_crop.shape[0])
    crop_w_old = old_crop.shape[1]
    crop_w_new = new_crop.shape[1]

    if old_crop.shape[0] < crop_h:
        pad = np.ones((crop_h - old_crop.shape[0], crop_w_old, 3), dtype=np.uint8) * 255
        old_crop = np.vstack([old_crop, pad])
    if new_crop.shape[0] < crop_h:
        pad = np.ones((crop_h - new_crop.shape[0], crop_w_new, 3), dtype=np.uint8) * 255
        new_crop = np.vstack([new_crop, pad])

    # Label hoogte
    label_h = 40
    total_w = crop_w_old + crop_w_new + 4  # 4px scheiding
    total_h = crop_h + label_h

    # Canvas maken
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Labels tekenen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "OUD", (10, 28), font, 0.8, (0, 0, 200), 2)
    cv2.putText(
        canvas, "NIEUW", (crop_w_old + 14, 28), font, 0.8, (0, 150, 0), 2
    )

    # Scheidingslijn onder labels
    cv2.line(canvas, (0, label_h - 2), (total_w, label_h - 2), (180, 180, 180), 1)

    # Crops plaatsen
    canvas[label_h:label_h + crop_h, 0:crop_w_old] = old_crop
    # Verticale scheidingslijn
    canvas[label_h:, crop_w_old:crop_w_old + 4] = (180, 180, 180)
    canvas[label_h:label_h + crop_h, crop_w_old + 4:crop_w_old + 4 + crop_w_new] = new_crop

    # Naar base64
    success, buffer = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("PNG encoding mislukt voor crop")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _call_claude_vision(
    client: anthropic.Anthropic,
    crop_b64: str,
    page_num: int,
    region_idx: int,
) -> list[dict[str, str]]:
    """
    Stuur een crop naar Claude Vision API en parse de response.

    Returns:
        Lijst van {"description": ..., "classification": ...} dicts.
    """
    start = time.time()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
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
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                }
            ],
            timeout=AI_TIMEOUT,
        )

        elapsed = time.time() - start
        usage = response.usage
        logger.info(
            "AI call pagina %d regio %d: %.1fs, %d input tokens, %d output tokens",
            page_num,
            region_idx,
            elapsed,
            usage.input_tokens,
            usage.output_tokens,
        )

        # Parse JSON uit response
        text = response.content[0].text.strip()
        # Strip eventuele markdown code block markers
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
            if "description" in c and "classification" in c:
                classification = c["classification"].upper()
                if classification not in ("KRITIEK", "NEGEREN"):
                    classification = "NEGEREN"
                valid.append({
                    "description": c["description"],
                    "classification": classification,
                })
        return valid

    except json.JSONDecodeError as e:
        logger.warning(
            "AI response pagina %d regio %d: ongeldige JSON: %s",
            page_num, region_idx, str(e),
        )
        return []
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "AI call mislukt pagina %d regio %d na %.1fs: %s",
            page_num, region_idx, elapsed, str(e),
        )
        return []


def interpret_page(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    diff_mask: np.ndarray,
    page_num: int,
) -> list[dict[str, Any]]:
    """
    Interpreteer alle gedetecteerde wijzigingen op een pagina via Claude Vision.

    Parameters:
        old_bgr: Oude afbeelding in BGR formaat.
        new_bgr: Nieuwe afbeelding in BGR formaat.
        diff_mask: Binair masker van gedetecteerde wijzigingen.
        page_num: Paginanummer (1-gebaseerd).

    Returns:
        Lijst van interpretatie dictionaries met region, crop_image,
        description en classification.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY niet geconfigureerd, interpretatie overgeslagen")
        return []

    # Bounding boxes vinden
    contours, _ = cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
            bboxes.append(cv2.boundingRect(contour))

    if not bboxes:
        return []

    # Cluster bounding boxes
    clusters = _cluster_bounding_boxes(bboxes)
    logger.info(
        "Pagina %d: %d contours → %d clusters voor AI interpretatie",
        page_num, len(bboxes), len(clusters),
    )

    # Anthropic client aanmaken
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        base_url="https://api.anthropic.com",
    )

    interpretations: list[dict[str, Any]] = []

    # Sequentieel verwerken (rate limits)
    for idx, (x, y, w, h) in enumerate(clusters):
        logger.info(
            "Pagina %d: interpretatie regio %d/%d (x=%d, y=%d, %dx%d)",
            page_num, idx + 1, len(clusters), x, y, w, h,
        )

        try:
            crop_b64 = _create_side_by_side_crop(old_bgr, new_bgr, x, y, w, h)
        except Exception as e:
            logger.error(
                "Crop mislukt pagina %d regio %d: %s", page_num, idx, str(e)
            )
            continue

        changes = _call_claude_vision(client, crop_b64, page_num, idx)

        for change in changes:
            if change["classification"] == "NEGEREN":
                continue
            interpretations.append({
                "region": {"x": x, "y": y, "width": w, "height": h},
                "crop_image": crop_b64,
                "description": change["description"],
                "classification": change["classification"],
            })

    return interpretations

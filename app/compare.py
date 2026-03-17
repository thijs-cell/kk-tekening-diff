"""Vergelijkingslogica per pagina-paar: diff berekenen, overlay genereren."""

import base64
import io
import logging
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from .alignment import align_images, compute_homography
from .config import (
    BBOX_LINE_THICKNESS,
    BBOX_PADDING,
    DISPLACEMENT_MATCH_RADIUS_PX,
    ENABLE_AI_INTERPRETATION,
    MIN_CONTOUR_AREA,
    MORPH_ITERATIONS,
    MORPH_KERNEL_SIZE,
    OVERLAY_ALPHA,
    TITLE_BLOCK_MASK_PERCENT,
)
from .interpreter import interpret_page

logger = logging.getLogger(__name__)


def _pil_to_cv2_gray(pil_img: Image.Image) -> np.ndarray:
    """Converteer PIL Image naar OpenCV grijswaarden numpy array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _pil_to_cv2_bgr(pil_img: Image.Image) -> np.ndarray:
    """Converteer PIL Image naar OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_base64_png(img: np.ndarray) -> str:
    """Encodeer OpenCV afbeelding naar base64 PNG string."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("PNG encoding mislukt")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _pil_to_base64_png(pil_img: Image.Image) -> str:
    """Encodeer PIL Image naar base64 PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _create_title_block_mask(h: int, w: int) -> np.ndarray:
    """
    Maak een masker dat het titelblok (rechter percentage) uitsluit.

    Returns:
        Masker waar 255 = zichtbaar gebied, 0 = gemaskeerd (titelblok).
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask_start = int(w * (1.0 - TITLE_BLOCK_MASK_PERCENT / 100.0))
    mask[:, mask_start:] = 0
    return mask


def _create_overlay(
    new_bgr: np.ndarray,
    diff_mask: np.ndarray,
) -> np.ndarray:
    """
    Genereer overlay afbeelding: nieuwe tekening met rode highlights op wijzigingen.
    """
    overlay = new_bgr.copy()

    red_layer = np.zeros_like(overlay)
    red_layer[:, :] = (0, 0, 255)

    change_mask_3ch = cv2.merge([diff_mask, diff_mask, diff_mask])
    overlay = np.where(
        change_mask_3ch > 0,
        cv2.addWeighted(overlay, 1.0 - OVERLAY_ALPHA, red_layer, OVERLAY_ALPHA, 0),
        overlay,
    )

    contours, _ = cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_CONTOUR_AREA:
            x, y, bw, bh = cv2.boundingRect(contour)
            x = max(0, x - BBOX_PADDING)
            y = max(0, y - BBOX_PADDING)
            x2 = min(overlay.shape[1], x + bw + 2 * BBOX_PADDING)
            y2 = min(overlay.shape[0], y + bh + 2 * BBOX_PADDING)
            cv2.rectangle(
                overlay, (x, y), (x2, y2), (0, 0, 255), BBOX_LINE_THICKNESS
            )

    return overlay


def _compute_displacements(
    old_diff_mask: np.ndarray,
    new_diff_mask: np.ndarray,
    pixels_per_mm: float,
) -> list[dict[str, Any]]:
    """
    Bereken verschuivingen door contouren in oud en nieuw diff masker te matchen.

    Voor elke contour in het nieuwe masker: zoek een overeenkomstige contour
    in het oude masker binnen DISPLACEMENT_MATCH_RADIUS_PX. Bereken de
    pixelverschuiving tussen centroids en reken om naar mm.

    Returns:
        Lijst van dicts met positie, grootte en verschuiving_mm info.
    """
    # Vind contouren in nieuwe diff
    new_contours, _ = cv2.findContours(
        new_diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Vind contouren in oude diff
    old_contours, _ = cv2.findContours(
        old_diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Bereken centroids van oude contouren
    old_centroids: list[tuple[float, float, int]] = []
    for c in old_contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        old_centroids.append((cx, cy, len(old_centroids)))

    displacements: list[dict[str, Any]] = []

    for contour in new_contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        ncx = M["m10"] / M["m00"]
        ncy = M["m01"] / M["m00"]

        # Zoek dichtstbijzijnde oude contour
        best_dist = float("inf")
        best_old: Optional[tuple[float, float]] = None
        for ocx, ocy, _ in old_centroids:
            dist = ((ncx - ocx) ** 2 + (ncy - ocy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_old = (ocx, ocy)

        entry: dict[str, Any] = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "centroid": (round(ncx), round(ncy)),
        }

        if best_old is not None and best_dist <= DISPLACEMENT_MATCH_RADIUS_PX:
            dx_px = ncx - best_old[0]
            dy_px = ncy - best_old[1]
            dist_px = (dx_px**2 + dy_px**2) ** 0.5
            dist_mm = round(dist_px / pixels_per_mm) if pixels_per_mm > 0 else None
            entry["verschuiving_px"] = round(dist_px, 1)
            entry["verschuiving_mm"] = dist_mm
        else:
            entry["verschuiving_px"] = None
            entry["verschuiving_mm"] = None

        displacements.append(entry)

    return displacements


def compare_page(
    old_img: Optional[Image.Image],
    new_img: Optional[Image.Image],
    page_num: int,
    sensitivity: int,
    scale: int = 50,
    pixels_per_mm: float = 0.157,
) -> dict[str, Any]:
    """
    Vergelijk een pagina-paar en genereer diff + overlay afbeeldingen.

    Parameters:
        old_img: PIL Image van de oude pagina (None bij nieuwe pagina).
        new_img: PIL Image van de nieuwe pagina (None bij verwijderde pagina).
        page_num: Paginanummer (1-gebaseerd).
        sensitivity: Drempelwaarde voor verschildetectie (0-255).
        scale: Schaal van de tekening (bijv. 50 voor 1:50).
        pixels_per_mm: Pixels per mm werkelijkheid.

    Returns:
        Dictionary met vergelijkingsresultaat voor deze pagina.
    """
    result: dict[str, Any] = {
        "page": page_num,
        "status": "compared",
        "changes_detected": False,
        "change_percentage": 0.0,
        "diff_image": None,
        "overlay_image": None,
    }

    # Nieuwe pagina zonder oud equivalent
    if old_img is None and new_img is not None:
        result["status"] = "new_page"
        result["changes_detected"] = True
        result["change_percentage"] = 100.0
        result["overlay_image"] = _pil_to_base64_png(new_img)
        return result

    # Verwijderde pagina
    if new_img is None and old_img is not None:
        result["status"] = "removed_page"
        result["changes_detected"] = True
        result["change_percentage"] = 100.0
        return result

    # Beide None zou niet moeten voorkomen
    if old_img is None or new_img is None:
        result["status"] = "alignment_failed"
        return result

    # Converteer naar grijswaarden voor alignment
    old_gray = _pil_to_cv2_gray(old_img)
    new_gray = _pil_to_cv2_gray(new_img)
    new_bgr = _pil_to_cv2_bgr(new_img)

    h, w = new_gray.shape[:2]

    # Feature-based alignment
    aligned_old, success = align_images(old_gray, new_gray)

    if not success:
        logger.warning("Pagina %d: uitlijning mislukt", page_num)
        result["status"] = "alignment_failed"
        return result

    # Absoluut verschil berekenen
    diff_raw = cv2.absdiff(aligned_old, new_gray)

    # Threshold toepassen
    _, diff_thresh = cv2.threshold(diff_raw, sensitivity, 255, cv2.THRESH_BINARY)

    # Morphologische operaties om ruis te verwijderen
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    diff_clean = cv2.morphologyEx(
        diff_thresh, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS
    )
    diff_clean = cv2.morphologyEx(
        diff_clean, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS
    )

    # Titelblok masker toepassen
    title_mask = _create_title_block_mask(h, w)
    diff_masked = cv2.bitwise_and(diff_clean, title_mask)

    # Change percentage berekenen
    visible_pixels = int(np.count_nonzero(title_mask))
    changed_pixels = int(np.count_nonzero(diff_masked))

    if visible_pixels > 0:
        change_pct = round((changed_pixels / visible_pixels) * 100, 2)
    else:
        change_pct = 0.0

    result["change_percentage"] = change_pct

    if changed_pixels == 0 or change_pct < 0.05:
        result["status"] = "no_changes"
        result["changes_detected"] = False
        return result

    result["changes_detected"] = True

    # Diff afbeelding genereren (zwart/wit)
    result["diff_image"] = _cv2_to_base64_png(diff_masked)

    # Overlay afbeelding genereren
    overlay = _create_overlay(new_bgr, diff_masked)
    result["overlay_image"] = _cv2_to_base64_png(overlay)

    # Verschuivingen berekenen op basis van aligned old vs new diff
    # Bereken ook diff van aligned_old t.o.v. original old voor verschuiving
    old_bgr_raw = _pil_to_cv2_bgr(old_img)
    H = compute_homography(_pil_to_cv2_gray(old_img), new_gray)
    if H is not None:
        aligned_old_bgr = cv2.warpPerspective(old_bgr_raw, H, (w, h))
    else:
        aligned_old_bgr = old_bgr_raw

    displacements = _compute_displacements(diff_masked, diff_masked, pixels_per_mm)
    logger.info(
        "Pagina %d: %d verschuivingen berekend (schaal 1:%d, %.4f px/mm)",
        page_num, len(displacements), scale, pixels_per_mm,
    )

    # AI interpretatie van wijzigingen (strook-gebaseerd)
    if ENABLE_AI_INTERPRETATION:
        result["interpretations"] = interpret_page(
            aligned_old_bgr, new_bgr, diff_masked, page_num,
            scale=scale, pixels_per_mm=pixels_per_mm,
            displacements=displacements,
        )
    else:
        result["interpretations"] = []

    # Geheugen vrijgeven
    del aligned_old, diff_raw, diff_thresh, diff_clean, diff_masked
    del old_gray, new_gray, new_bgr, overlay, title_mask
    del old_bgr_raw, aligned_old_bgr

    return result

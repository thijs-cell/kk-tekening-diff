"""ORB-gebaseerde feature matching en homografie-uitlijning voor bouwtekeningen."""

import logging
from typing import Optional

import cv2
import numpy as np

from .config import MIN_GOOD_MATCHES, ORB_FEATURES, RANSAC_THRESHOLD

logger = logging.getLogger(__name__)


def align_images(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
) -> tuple[Optional[np.ndarray], bool]:
    """
    Lijn de oude afbeelding uit op de nieuwe via ORB feature matching en homografie.

    Parameters:
        old_gray: Grijswaarden afbeelding van de oude tekening.
        new_gray: Grijswaarden afbeelding van de nieuwe tekening.

    Returns:
        Tuple van (uitgelijnde oude afbeelding, succes boolean).
        Bij falen: (None, False).
    """
    h, w = new_gray.shape[:2]

    # ORB detector initialiseren
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    # Keypoints en descriptors detecteren
    kp_old, desc_old = orb.detectAndCompute(old_gray, None)
    kp_new, desc_new = orb.detectAndCompute(new_gray, None)

    if desc_old is None or desc_new is None:
        logger.warning("Geen features gevonden in een van de afbeeldingen")
        return None, False

    if len(kp_old) < MIN_GOOD_MATCHES or len(kp_new) < MIN_GOOD_MATCHES:
        logger.warning(
            "Te weinig keypoints: oud=%d, nieuw=%d", len(kp_old), len(kp_new)
        )
        return None, False

    # Brute-force matcher met Hamming afstand (voor ORB binaire descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc_old, desc_new, k=2)

    # Lowe's ratio test om goede matches te filteren
    good_matches: list[cv2.DMatch] = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    logger.info("Goede matches gevonden: %d", len(good_matches))

    if len(good_matches) < MIN_GOOD_MATCHES:
        logger.warning(
            "Te weinig goede matches (%d < %d), uitlijning mislukt",
            len(good_matches),
            MIN_GOOD_MATCHES,
        )
        return None, False

    # Punten verzamelen voor homografie
    src_pts = np.float32(
        [kp_old[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_new[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Homografie berekenen met RANSAC
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)

    if matrix is None:
        logger.warning("Homografie berekening mislukt")
        return None, False

    inliers = int(mask.sum()) if mask is not None else 0
    logger.info("Homografie inliers: %d / %d", inliers, len(good_matches))

    if inliers < MIN_GOOD_MATCHES:
        logger.warning("Te weinig inliers (%d), uitlijning onbetrouwbaar", inliers)
        return None, False

    # Oude afbeelding transformeren naar het coördinatensysteem van de nieuwe
    aligned = cv2.warpPerspective(old_gray, matrix, (w, h))

    return aligned, True

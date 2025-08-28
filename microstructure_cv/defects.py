from __future__ import annotations

import cv2
import numpy as np


def detect_defects(
    img_gray: np.ndarray,
    min_area_px: int = 20,
    max_area_px: int | None = None,
    mode: str = "dark",  # "dark" for voids/pores, "bright" for inclusions
) -> tuple[np.ndarray, list[dict]]:
    """Detect defects via simple intensity thresholding + morphology.

    Returns:
        mask: uint8 binary mask of defects
        defects: list of dicts with area, centroid, bbox
    """
    if img_gray.ndim != 2:
        raise ValueError("detect_defects expects grayscale image")

    if mode not in {"dark", "bright"}:
        raise ValueError("mode must be 'dark' or 'bright'")

    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    if mode == "dark":
        # Defects darker than matrix
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 35, 5)
    else:
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 35, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Filter by area
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)
    defects: list[dict] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        final_mask[labels == label] = 255
        defects.append({
            "area_px": area,
            "centroid": [float(cx), float(cy)],
            "bbox": [x, y, w, h],
            "label": int(label),
        })

    return final_mask, defects


from __future__ import annotations

import cv2
import numpy as np


def segment_phases_kmeans(
    image_bgr: np.ndarray,
    k: int = 2,
    space: str = "lab",
    attempts: int = 5,
) -> tuple[np.ndarray, dict]:
    """Segment phases by k-means clustering in color space.

    Returns:
        labels: int32 label map [0..k-1]
        info: dict with centroid colors and area fractions
    """
    if space not in {"lab", "hsv", "rgb"}:
        raise ValueError("space must be 'lab', 'hsv', or 'rgb'")

    img = image_bgr.copy()
    if space == "lab":
        feat = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif space == "hsv":
        feat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        feat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    Z = feat.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((feat.shape[0], feat.shape[1]))
    labels = labels.astype(np.int32)

    # Area fractions
    counts = np.bincount(labels.ravel(), minlength=k)
    total = float(labels.size)
    fractions = (counts / total).tolist()

    info = {
        "centers": centers.tolist(),
        "fractions": fractions,
    }
    return labels, info


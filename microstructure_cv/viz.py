from __future__ import annotations

import cv2
import numpy as np


def overlay_boundaries(image_bgr: np.ndarray, boundaries_mask: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = image_bgr.copy()
    if boundaries_mask.ndim == 2:
        mask = boundaries_mask > 0
        out[mask] = color
    else:
        raise ValueError("boundaries_mask must be single-channel")
    return out


def labels_to_random_colors(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(labels)
    for lab in unique:
        if lab <= 0:
            continue
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        out[labels == lab] = color
    return out


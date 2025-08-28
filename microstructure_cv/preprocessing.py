from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(
    image_bgr: np.ndarray,
    to_gray: bool = True,
    denoise_ksize: int = 3,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    """Preprocess image for microstructure analysis.

    Steps:
      - Optional grayscale conversion
      - Bilateral filter denoising preserving edges
      - Optional CLAHE for contrast normalization
    """
    img = image_bgr.copy()

    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise while preserving edges
    if denoise_ksize and denoise_ksize > 0:
        if img.ndim == 2:
            img = cv2.bilateralFilter(img, d=denoise_ksize, sigmaColor=35, sigmaSpace=35)
        else:
            img = cv2.bilateralFilter(img, d=denoise_ksize, sigmaColor=35, sigmaSpace=35)

    # Contrast enhancement
    if use_clahe:
        if img.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
            img = clahe.apply(img)
        else:
            # Apply CLAHE per channel in LAB space for color images
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


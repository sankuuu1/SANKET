from __future__ import annotations

import cv2
import numpy as np


def segment_grains(
    img_gray: np.ndarray,
    gaussian_ksize: int = 3,
    otsu: bool = True,
    morph_open_ksize: int = 3,
    morph_close_ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Segment grains using threshold + watershed.

    Returns: (labels, boundaries_mask, num_grains)
    labels: int32 label image with 0 as background
    boundaries_mask: uint8 mask of grain boundaries (for visualization)
    """
    if img_gray.ndim != 2:
        raise ValueError("segment_grains expects grayscale image")

    # Smooth slightly to reduce noise
    if gaussian_ksize > 0:
        img_blur = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize), 0)
    else:
        img_blur = img_gray

    # Thresholding
    if otsu:
        _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive threshold for uneven illumination
        thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 35, 2)

    # Ensure grains are foreground (white)
    # Heuristic: pick inversion that yields larger foreground area but bounded
    if np.sum(thresh == 255) < np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)

    # Morphological cleanup
    if morph_open_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    if morph_close_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Distance transform and markers
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    # Use fraction of max distance for sure foreground
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(thresh, sure_fg)

    # Marker labelling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # make background 1
    markers[unknown == 255] = 0

    # Watershed requires 3-channel image
    img_color_for_ws = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color_for_ws, markers)

    # Post process: negative markers are boundaries
    boundaries_mask = (markers == -1).astype(np.uint8) * 255

    # Relabel to have consecutive labels starting at 1 for grains
    labels = markers.copy()
    labels[labels == -1] = 0
    labels[labels == 1] = 0  # watershed background revert to 0

    # Convert to compact labels
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 1]
    relabel_map = {old: i + 1 for i, old in enumerate(unique_labels)}
    compact = np.zeros_like(labels, dtype=np.int32)
    for old, new in relabel_map.items():
        compact[labels == old] = new

    num_grains = len(relabel_map)
    return compact.astype(np.int32), boundaries_mask, num_grains


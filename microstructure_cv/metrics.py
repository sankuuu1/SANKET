from __future__ import annotations

import numpy as np
import cv2


def compute_grain_metrics(labels: np.ndarray, pixel_size_um: float | None = None) -> dict:
    """Compute grain count and size statistics from label image.

    Args:
        labels: int32 label map where 0 is background and positive labels are grains
        pixel_size_um: if provided, convert area and equivalent diameter to microns
    Returns:
        dict with count, area stats, equiv_diameter stats, and optional ASTM-like number
    """
    if labels.dtype != np.int32:
        labels = labels.astype(np.int32)

    grain_ids = np.unique(labels)
    grain_ids = grain_ids[grain_ids > 0]
    num_grains = int(grain_ids.size)
    if num_grains == 0:
        return {
            "num_grains": 0,
            "areas_px": [],
            "equiv_diam_px": [],
        }

    areas_px = []
    equiv_diam_px = []
    for gid in grain_ids:
        mask = (labels == gid).astype(np.uint8)
        area = int(np.sum(mask))
        areas_px.append(area)
        # Equivalent diameter from area
        eq_d = np.sqrt(4 * area / np.pi)
        equiv_diam_px.append(eq_d)

    areas_px = np.array(areas_px, dtype=float)
    equiv_diam_px = np.array(equiv_diam_px, dtype=float)

    def stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    result = {
        "num_grains": num_grains,
        "areas_px": areas_px.tolist(),
        "equiv_diam_px": equiv_diam_px.tolist(),
        "area_stats_px": stats(areas_px),
        "equiv_diam_stats_px": stats(equiv_diam_px),
    }

    if pixel_size_um is not None and pixel_size_um > 0:
        areas_um2 = areas_px * (pixel_size_um ** 2)
        equiv_diam_um = equiv_diam_px * pixel_size_um
        result.update({
            "areas_um2": areas_um2.tolist(),
            "equiv_diam_um": equiv_diam_um.tolist(),
            "area_stats_um2": stats(areas_um2),
            "equiv_diam_stats_um": stats(equiv_diam_um),
        })

        # Approximate ASTM grain size number G using mean area at 1x reference
        # This is highly approximate without calibration at 1 in^2 area.
        # Provide a placeholder using mean diameter in um.
        mean_d_um = float(np.mean(equiv_diam_um))
        if mean_d_um > 0:
            # Not a true ASTM conversion; include as auxiliary indicator.
            result["approx_grain_number"] = float(max(0.0, 10.0 - np.log2(mean_d_um)))

    return result


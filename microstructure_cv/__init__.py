"""Microstructure CV toolkit.

Provides utilities to analyze microstructural images:
- Grain segmentation and metrics
- Phase clustering and area fractions
- Defect detection (voids/inclusions)
- Visualization and CLI
"""

from .preprocessing import preprocess_image
from .grains import segment_grains
from .phases import segment_phases_kmeans
from .defects import detect_defects
from .metrics import compute_grain_metrics

__all__ = [
    "preprocess_image",
    "segment_grains",
    "segment_phases_kmeans",
    "detect_defects",
    "compute_grain_metrics",
]


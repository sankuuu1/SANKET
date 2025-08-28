from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import preprocess_image
from .grains import segment_grains
from .metrics import compute_grain_metrics
from .defects import detect_defects
from .phases import segment_phases_kmeans
from .viz import overlay_boundaries, labels_to_random_colors


def run_cli():
    parser = argparse.ArgumentParser(description="Microstructure CV analysis")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--pixel_size_um", type=float, default=None, help="Microns per pixel")
    parser.add_argument("--k_phases", type=int, default=0, help="If >0, run k-means phase segmentation with k clusters")
    parser.add_argument("--defects", action="store_true", help="Detect dark defects (voids)")
    parser.add_argument("--defects_mode", type=str, default="dark", choices=["dark", "bright"], help="Defect type")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to write outputs")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess for grains
    img_p = preprocess_image(img, to_gray=True)

    # Grain segmentation
    labels, boundaries, n_grains = segment_grains(img_p)
    metrics = compute_grain_metrics(labels, pixel_size_um=args.pixel_size_um)

    # Visuals
    overlay = overlay_boundaries(img, boundaries)
    color_labels = labels_to_random_colors(labels)

    # Save visuals
    stem = Path(args.image).stem
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_labels.png"), color_labels)

    results = {
        "file": str(args.image),
        "grain_metrics": metrics,
    }

    # Phase analysis
    if args.k_phases and args.k_phases > 0:
        phase_labels, phase_info = segment_phases_kmeans(img, k=args.k_phases, space="lab")
        results["phases"] = phase_info
        phase_rgb = labels_to_random_colors(phase_labels)
        cv2.imwrite(str(out_dir / f"{stem}_phases.png"), phase_rgb)

    # Defects
    if args.defects:
        defects_mask, defects_list = detect_defects(img_p, mode=args.defects_mode)
        results["defects"] = {
            "count": len(defects_list),
            "items": defects_list,
        }
        cv2.imwrite(str(out_dir / f"{stem}_defects.png"), defects_mask)

    with open(out_dir / f"{stem}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_cli()


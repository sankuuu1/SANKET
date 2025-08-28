## Microstructure CV Toolkit

Analyze metallography/microstructural images to extract:
- Grain count, size distribution, equivalent diameter, approximate grain number
- Phase clustering (k-means) and area fractions
- Defect detection (voids/inclusions) with counts and areas

### Install

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```
python -m microstructure_cv.cli /path/to/image.png --pixel_size_um 0.2 --k_phases 3 --defects --defects_mode dark --output_dir output
```

Outputs in `output/`:
- `<name>_overlay.png`: Grain boundaries overlay
- `<name>_labels.png`: Colored grains
- `<name>_phases.png`: Colored phase clusters (if requested)
- `<name>_defects.png`: Defect mask (if requested)
- `<name>_results.json`: Numeric results

### Notes
- Provide correct `--pixel_size_um` for physical units.
- For etched images, watershed works well; adjust thresholds if needed.
- For phase segmentation, tune `--k_phases`.

# SANKET
# Microstructural Analysis by Computer Vision

A comprehensive Python toolkit for analyzing microstructural images using computer vision and machine learning techniques. This tool can extract detailed features from metallographic images including grain characteristics, defects, phases, and advanced statistical properties.

## Features

### Core Analysis Capabilities
- **Grain Analysis**: Detection, segmentation, and characterization of grains
- **Defect Detection**: Identification of pores, cracks, and inclusions
- **Phase Analysis**: Multi-phase identification and quantification
- **Texture Analysis**: GLCM and LBP texture feature extraction
- **Statistical Analysis**: Comprehensive statistical characterization

### Advanced Features
- **Machine Learning**: Anomaly detection and grain clustering
- **Spatial Analysis**: Spatial distribution and clustering statistics
- **Distribution Fitting**: Grain size distribution analysis
- **Material Classification**: Material-specific microstructure classification
- **Comprehensive Reporting**: Automated report generation with visualizations

## Installation

1. Clone this repository or download the files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- OpenCV (cv2)
- scikit-image
- NumPy
- Matplotlib
- SciPy
- Pandas
- Seaborn
- Pillow
- scikit-learn
- plotly

## Quick Start

### Basic Usage

```python
from microstructural_analyzer import MicrostructuralAnalyzer

# Initialize analyzer with your image
analyzer = MicrostructuralAnalyzer('your_image.png', scale_pixels_per_micron=2.0)

# Run complete analysis
results = analyzer.run_complete_analysis()

# Generate report and visualizations
analyzer.generate_report(save_path="analysis_report.txt")
analyzer.visualize_results(save_path="analysis_plots.png")
```

### Advanced Analysis

```python
from advanced_features import AdvancedMicrostructuralAnalyzer, run_advanced_analysis

# Run basic analysis first
analyzer = MicrostructuralAnalyzer('your_image.png', scale_pixels_per_micron=2.0)
analyzer.run_complete_analysis()

# Run advanced analysis
advanced = run_advanced_analysis(analyzer)
```

### Example with Sample Data

```python
from microstructural_analyzer import create_sample_microstructure, MicrostructuralAnalyzer

# Create a sample microstructure
create_sample_microstructure(save_path="sample.png")

# Analyze it
analyzer = MicrostructuralAnalyzer("sample.png", scale_pixels_per_micron=2.0)
results = analyzer.run_complete_analysis()
```

## Detailed Usage

### 1. Image Preprocessing

```python
# Load and preprocess image
analyzer.load_image("microstructure.png")
analyzer.preprocess_image(
    denoise=True,
    enhance_contrast=True,
    gaussian_sigma=1.0,
    clahe_clip_limit=2.0
)
```

### 2. Grain Analysis

```python
# Detect grain boundaries
analyzer.detect_grain_boundaries(method='sobel', threshold_factor=0.5)

# Segment individual grains
analyzer.segment_grains(min_distance=20, watershed_compactness=0.1)

# Analyze grain properties
grain_data = analyzer.analyze_grains()
print(f"Found {len(grain_data)} grains")
```

### 3. Defect Detection

```python
# Detect various defects
defects = analyzer.detect_defects(
    pore_threshold=0.3,
    crack_min_length=20
)

print(f"Found {len(defects['pores'])} pores")
print(f"Found {len(defects['cracks'])} cracks")
print(f"Found {len(defects['inclusions'])} inclusions")
```

### 4. Phase Analysis

```python
# Identify phases
phases = analyzer.analyze_phases(n_phases=2, method='kmeans')
print(f"Identified {len(phases['data'])} phases")
```

### 5. Advanced Features

```python
from advanced_features import AdvancedMicrostructuralAnalyzer

# Initialize advanced analyzer
advanced = AdvancedMicrostructuralAnalyzer(analyzer)

# Extract ML features
features = advanced.extract_ml_features()

# Detect anomalous grains
anomalies = advanced.detect_anomalous_grains(contamination=0.1)

# Cluster grains
clusters = advanced.cluster_grains(n_clusters=3, method='kmeans')

# Analyze grain size distribution
dist_analysis = advanced.analyze_grain_size_distribution()

# Calculate spatial statistics
spatial_stats = advanced.calculate_spatial_statistics()
```

## Analysis Results

### Grain Analysis Output
- Total number of grains
- Average, median, min, max grain sizes
- Grain size distribution statistics
- Average aspect ratio and circularity
- Grain area fraction

### Defect Analysis Output
- Pore count and characteristics
- Crack detection and measurements
- Inclusion identification
- Defect area fractions
- Statistical summaries

### Phase Analysis Output
- Number of phases detected
- Phase area fractions
- Phase intensity characteristics
- Phase distribution maps

### Advanced Analysis Output
- Anomalous grain detection
- Grain clustering results
- Spatial distribution analysis
- Grain size distribution fitting
- Material classification
- Comprehensive statistical analysis

## Customization

### Custom Parameters

You can customize analysis parameters for different materials and imaging conditions:

```python
# Custom preprocessing
preprocess_params = {
    'denoise': True,
    'enhance_contrast': True,
    'gaussian_sigma': 0.8,
    'clahe_clip_limit': 3.0
}

# Custom grain detection
grain_params = {
    'min_distance': 15,
    'watershed_compactness': 0.2
}

# Custom defect detection
defect_params = {
    'pore_threshold': 0.25,
    'crack_min_length': 30
}

# Run with custom parameters
results = analyzer.run_complete_analysis(
    preprocess_params=preprocess_params,
    grain_params=grain_params,
    defect_params=defect_params
)
```

### Batch Processing

```python
from example_usage import batch_analysis

# Analyze multiple images
results = batch_analysis(
    image_folder="./images/",
    file_extension=".png",
    scale=2.0
)
```

## Output Files

The analysis generates several output files:

1. **Text Reports** (`*_report.txt`): Detailed numerical analysis results
2. **Visualization Plots** (`*_analysis.png`): Comprehensive visual analysis
3. **CSV Summary** (`batch_analysis_summary.csv`): Batch analysis results
4. **Advanced Reports** (`advanced_analysis_report.txt`): ML and statistical analysis

## Examples and Use Cases

### Steel Microstructure Analysis
- Grain size measurement (ASTM standards)
- Phase fraction analysis (ferrite/pearlite)
- Defect quantification
- Quality assessment

### Aluminum Alloy Analysis
- Precipitate detection
- Grain boundary analysis
- Texture characterization
- Processing optimization

### Ceramic Analysis
- Porosity measurement
- Grain size distribution
- Phase identification
- Sintering quality assessment

### Research Applications
- Microstructure-property relationships
- Processing parameter optimization
- Quality control and inspection
- Failure analysis

## Technical Details

### Image Processing Pipeline
1. **Preprocessing**: Noise reduction, contrast enhancement
2. **Segmentation**: Edge detection, watershed segmentation
3. **Feature Extraction**: Geometric, morphological, and texture features
4. **Classification**: Statistical and ML-based analysis

### Algorithms Used
- **Edge Detection**: Sobel, Canny, Scharr filters
- **Segmentation**: Watershed, Otsu thresholding, K-means clustering
- **Machine Learning**: Isolation Forest, DBSCAN, Random Forest
- **Statistical Analysis**: Distribution fitting, spatial statistics

## Troubleshooting

### Common Issues

1. **Poor Grain Detection**
   - Adjust preprocessing parameters
   - Try different edge detection methods
   - Modify watershed parameters

2. **Inaccurate Defect Detection**
   - Adjust threshold parameters
   - Check image quality and contrast
   - Modify morphological operation parameters

3. **Scale Calibration**
   - Ensure correct `scale_pixels_per_micron` value
   - Use calibration standards or scale bars

### Tips for Best Results

1. **Image Quality**: Use high-quality, well-contrasted images
2. **Preprocessing**: Experiment with different preprocessing parameters
3. **Scale Calibration**: Always calibrate pixel-to-micron conversion
4. **Parameter Tuning**: Adjust analysis parameters based on material type
5. **Validation**: Manually verify results on a subset of data

## Contributing

This is an open framework that can be extended for specific applications. Feel free to:
- Add new analysis methods
- Implement material-specific algorithms
- Improve visualization capabilities
- Add new machine learning models

## Citation

If you use this tool in your research, please cite it appropriately and reference the underlying algorithms and libraries used.

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with the licenses of all dependencies.

## Support

For questions, issues, or suggestions, please refer to the documentation or create an issue in the repository.
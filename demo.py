#!/usr/bin/env python3
"""
Simple Demo of Microstructural Analysis
=======================================

This script demonstrates the basic usage of the microstructural analysis toolkit.
It creates a sample image and performs analysis without requiring external dependencies
that might not be installed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def create_simple_demo():
    """Create a simple demonstration without heavy dependencies."""
    
    print("Microstructural Analysis Toolkit Demo")
    print("="*50)
    
    # Create a simple synthetic microstructure
    print("1. Creating synthetic microstructure...")
    
    # Simple grain simulation using random circles
    size = (400, 400)
    img = np.ones(size, dtype=np.uint8) * 128  # Gray background
    
    # Add some grains as circles with different intensities
    np.random.seed(42)
    n_grains = 30
    
    for i in range(n_grains):
        # Random center
        center_x = np.random.randint(50, size[1]-50)
        center_y = np.random.randint(50, size[0]-50)
        
        # Random radius
        radius = np.random.randint(15, 35)
        
        # Random intensity
        intensity = np.random.randint(80, 200)
        
        # Draw filled circle
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = intensity
    
    # Add some noise
    noise = np.random.normal(0, 10, size).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"   Created synthetic image of size {size}")
    
    # Basic analysis simulation
    print("\n2. Performing basic analysis...")
    
    # Simulate grain detection
    unique_intensities = len(np.unique(img))
    estimated_grains = max(10, unique_intensities // 8)
    
    # Simulate measurements
    avg_grain_size = np.random.uniform(20, 40)  # microns
    grain_size_std = avg_grain_size * 0.3
    
    # Simulate defect detection
    n_pores = np.random.randint(2, 8)
    n_cracks = np.random.randint(0, 3)
    
    print(f"   Estimated grains: {estimated_grains}")
    print(f"   Average grain size: {avg_grain_size:.1f} ± {grain_size_std:.1f} μm")
    print(f"   Detected defects: {n_pores} pores, {n_cracks} cracks")
    
    # Create visualization
    print("\n3. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Synthetic Microstructure')
    axes[0, 0].axis('off')
    
    # Histogram
    axes[0, 1].hist(img.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Intensity Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Simulated grain size distribution
    grain_sizes = np.random.normal(avg_grain_size, grain_size_std, estimated_grains)
    grain_sizes = grain_sizes[grain_sizes > 0]  # Remove negative values
    
    axes[1, 0].hist(grain_sizes, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Grain Size (μm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Simulated Grain Size Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
ANALYSIS SUMMARY
================

Image Size: {size[0]} × {size[1]} pixels
Estimated Grains: {estimated_grains}
Avg Grain Size: {avg_grain_size:.1f} μm
Grain Size Range: {grain_sizes.min():.1f} - {grain_sizes.max():.1f} μm
Standard Deviation: {grain_size_std:.1f} μm

DEFECTS DETECTED:
• Pores: {n_pores}
• Cracks: {n_cracks}

QUALITY METRICS:
• Grain Uniformity: {'Good' if grain_size_std/avg_grain_size < 0.4 else 'Poor'}
• Defect Density: {'Low' if n_pores + n_cracks < 5 else 'High'}
    """
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('demo_analysis.png', dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'demo_analysis.png'")
    
    # Save the synthetic image
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title('Synthetic Microstructure for Analysis')
    plt.axis('off')
    plt.savefig('synthetic_microstructure.png', dpi=300, bbox_inches='tight')
    print("   Synthetic image saved as 'synthetic_microstructure.png'")
    
    # Generate text report
    report = f"""
MICROSTRUCTURAL ANALYSIS DEMO REPORT
====================================

Date: Demo Version
Image: synthetic_microstructure.png
Scale: 2.0 pixels/micron (assumed)

GRAIN ANALYSIS:
--------------
Total Grains Detected: {estimated_grains}
Average Grain Size: {avg_grain_size:.2f} ± {grain_size_std:.2f} μm
Median Grain Size: {np.median(grain_sizes):.2f} μm
Grain Size Range: {grain_sizes.min():.2f} - {grain_sizes.max():.2f} μm
Grain Size Distribution: Normal (simulated)

DEFECT ANALYSIS:
---------------
Total Pores: {n_pores}
Total Cracks: {n_cracks}
Total Inclusions: 0
Defect Area Fraction: {(n_pores + n_cracks) * 0.001:.4f} (estimated)

QUALITY ASSESSMENT:
------------------
Grain Uniformity: {'Excellent' if grain_size_std/avg_grain_size < 0.2 else 'Good' if grain_size_std/avg_grain_size < 0.4 else 'Poor'}
Overall Quality: {'Good' if n_pores + n_cracks < 5 and grain_size_std/avg_grain_size < 0.4 else 'Acceptable'}

RECOMMENDATIONS:
---------------
- This is a demonstration with simulated data
- For real analysis, use the full MicrostructuralAnalyzer class
- Calibrate the scale factor for accurate measurements
- Adjust preprocessing parameters based on image quality

Note: This report was generated from simulated data for demonstration purposes.
    """
    
    with open('demo_report.txt', 'w') as f:
        f.write(report)
    
    print("   Text report saved as 'demo_report.txt'")
    
    print("\n4. Demo completed successfully!")
    print("\nGenerated files:")
    print("- synthetic_microstructure.png: Sample microstructure image")
    print("- demo_analysis.png: Analysis visualization")
    print("- demo_report.txt: Analysis report")
    
    print("\nTo use the full analysis toolkit:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: python example_usage.py")
    print("3. Or use: from microstructural_analyzer import MicrostructuralAnalyzer")
    
    return img, grain_sizes

def show_approach():
    """Show the general approach for microstructural analysis."""
    
    print("\n\nMICROSTRUCTURAL ANALYSIS APPROACH")
    print("="*50)
    
    approach = """
1. IMAGE PREPROCESSING:
   - Noise reduction (bilateral filtering, Gaussian)
   - Contrast enhancement (CLAHE, histogram equalization)
   - Image calibration (pixel-to-micron conversion)

2. GRAIN ANALYSIS:
   - Edge detection (Sobel, Canny, Scharr filters)
   - Boundary detection and cleaning
   - Watershed segmentation for individual grains
   - Feature extraction (size, shape, orientation)

3. DEFECT DETECTION:
   - Pore detection (dark circular regions)
   - Crack detection (linear morphological operations)
   - Inclusion detection (intensity-based thresholding)

4. PHASE ANALYSIS:
   - Multi-level thresholding (Otsu)
   - K-means clustering
   - Watershed-based segmentation
   - Phase fraction calculation

5. FEATURE EXTRACTION:
   - Geometric features (area, perimeter, circularity)
   - Morphological features (aspect ratio, solidity)
   - Texture features (GLCM, LBP)
   - Statistical measures

6. ADVANCED ANALYSIS:
   - Machine learning for anomaly detection
   - Spatial statistics and clustering
   - Distribution fitting and modeling
   - Material-specific classification

7. VISUALIZATION & REPORTING:
   - Comprehensive plots and charts
   - Statistical summaries
   - Quality metrics
   - Automated report generation

KEY ALGORITHMS USED:
===================

• Edge Detection: Sobel, Canny, Scharr operators
• Segmentation: Watershed, Otsu thresholding, K-means
• Morphological Operations: Opening, closing, erosion, dilation
• Machine Learning: Isolation Forest, DBSCAN, Random Forest
• Statistical Analysis: Distribution fitting, hypothesis testing
• Texture Analysis: Gray-level co-occurrence matrix (GLCM)
• Spatial Analysis: Nearest neighbor, Ripley's K function

APPLICATIONS:
============

• Steel: Grain size (ASTM), phase fractions, defect analysis
• Aluminum: Precipitate analysis, texture characterization
• Ceramics: Porosity measurement, sintering quality
• Composites: Fiber distribution, matrix characterization
• Quality Control: Automated inspection, process optimization
• Research: Structure-property relationships, failure analysis

BEST PRACTICES:
==============

1. Use high-quality, well-prepared samples
2. Calibrate the measurement scale accurately
3. Adjust parameters based on material type
4. Validate results with manual measurements
5. Consider multiple analysis methods for confirmation
6. Document all processing parameters used
    """
    
    print(approach)
    
    # Save approach to file
    with open('analysis_approach.txt', 'w') as f:
        f.write("MICROSTRUCTURAL ANALYSIS APPROACH\n")
        f.write("="*50 + "\n")
        f.write(approach)
    
    print("\nApproach documentation saved as 'analysis_approach.txt'")

if __name__ == "__main__":
    # Run the demo
    try:
        img, grain_sizes = create_simple_demo()
        show_approach()
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("This might be due to missing dependencies.")
        print("Install requirements with: pip install -r requirements.txt")
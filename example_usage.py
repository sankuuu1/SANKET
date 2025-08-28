#!/usr/bin/env python3
"""
Example Usage of Microstructural Analyzer
==========================================

This script demonstrates how to use the MicrostructuralAnalyzer class
for comprehensive analysis of metallographic images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from microstructural_analyzer import MicrostructuralAnalyzer, create_sample_microstructure

def analyze_real_image(image_path, scale_pixels_per_micron=1.0):
    """
    Analyze a real microstructural image.
    
    Parameters:
    -----------
    image_path : str
        Path to the microstructural image
    scale_pixels_per_micron : float
        Calibration factor for converting pixels to microns
    """
    print(f"Analyzing real image: {image_path}")
    print("="*50)
    
    # Initialize analyzer
    analyzer = MicrostructuralAnalyzer(image_path, scale_pixels_per_micron)
    
    # Custom parameters for different analysis steps
    preprocess_params = {
        'denoise': True,
        'enhance_contrast': True,
        'gaussian_sigma': 1.0,
        'clahe_clip_limit': 3.0
    }
    
    grain_params = {
        'min_distance': 15,
        'watershed_compactness': 0.1
    }
    
    defect_params = {
        'pore_threshold': 0.4,
        'crack_min_length': 25
    }
    
    phase_params = {
        'n_phases': 2,
        'method': 'kmeans'
    }
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        preprocess_params=preprocess_params,
        grain_params=grain_params,
        defect_params=defect_params,
        phase_params=phase_params
    )
    
    # Generate outputs
    base_name = image_path.split('.')[0]
    analyzer.generate_report(save_path=f"{base_name}_report.txt")
    analyzer.visualize_results(save_path=f"{base_name}_analysis.png")
    
    return analyzer, results

def analyze_sample_image():
    """Create and analyze a sample microstructural image."""
    print("Creating and analyzing sample microstructure...")
    print("="*50)
    
    # Create sample microstructure with different parameters
    sample_path = "advanced_sample.png"
    create_sample_microstructure(
        size=(800, 800),
        n_grains=75,
        noise_level=0.05,
        save_path=sample_path
    )
    
    # Analyze the sample
    analyzer = MicrostructuralAnalyzer(sample_path, scale_pixels_per_micron=2.5)
    
    # Run analysis with default parameters
    results = analyzer.run_complete_analysis()
    
    # Generate outputs
    analyzer.generate_report(save_path="sample_analysis_report.txt")
    analyzer.visualize_results(save_path="sample_analysis_visualization.png")
    
    return analyzer, results

def batch_analysis(image_folder, file_extension='.png', scale=1.0):
    """
    Perform batch analysis on multiple images.
    
    Parameters:
    -----------
    image_folder : str
        Path to folder containing images
    file_extension : str
        File extension to look for
    scale : float
        Scale factor for all images
    """
    import os
    import glob
    
    print(f"Performing batch analysis on {image_folder}")
    print("="*50)
    
    # Find all images
    pattern = os.path.join(image_folder, f"*{file_extension}")
    image_files = glob.glob(pattern)
    
    if not image_files:
        print(f"No images found with extension {file_extension} in {image_folder}")
        return
    
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            analyzer = MicrostructuralAnalyzer(image_path, scale)
            results = analyzer.run_complete_analysis()
            
            # Extract key metrics
            summary = {
                'filename': os.path.basename(image_path),
                'total_grains': results.get('grain_analysis', {}).get('total_grains', 0),
                'avg_grain_size': results.get('grain_analysis', {}).get('avg_grain_size_microns', 0),
                'total_pores': results.get('defect_analysis', {}).get('total_pores', 0),
                'total_cracks': results.get('defect_analysis', {}).get('total_cracks', 0),
                'pore_area_fraction': results.get('defect_analysis', {}).get('pore_area_fraction', 0)
            }
            results_summary.append(summary)
            
            # Save individual results
            base_name = os.path.splitext(image_path)[0]
            analyzer.generate_report(save_path=f"{base_name}_report.txt")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Create batch summary
    if results_summary:
        import pandas as pd
        df = pd.DataFrame(results_summary)
        df.to_csv('batch_analysis_summary.csv', index=False)
        print(f"\nBatch analysis complete! Summary saved to batch_analysis_summary.csv")
        print(df.describe())
    
    return results_summary

def compare_processing_methods(image_path, scale=1.0):
    """
    Compare different processing methods on the same image.
    
    Parameters:
    -----------
    image_path : str
        Path to the microstructural image
    scale : float
        Scale factor
    """
    print(f"Comparing processing methods for: {image_path}")
    print("="*50)
    
    methods = {
        'Standard': {
            'preprocess': {'denoise': True, 'enhance_contrast': True, 'gaussian_sigma': 1.0},
            'grain': {'min_distance': 20},
            'defect': {'pore_threshold': 0.3},
            'phase': {'n_phases': 2, 'method': 'kmeans'}
        },
        'Aggressive': {
            'preprocess': {'denoise': True, 'enhance_contrast': True, 'gaussian_sigma': 0.5, 'clahe_clip_limit': 4.0},
            'grain': {'min_distance': 10},
            'defect': {'pore_threshold': 0.2},
            'phase': {'n_phases': 3, 'method': 'otsu'}
        },
        'Conservative': {
            'preprocess': {'denoise': False, 'enhance_contrast': False, 'gaussian_sigma': 2.0},
            'grain': {'min_distance': 30},
            'defect': {'pore_threshold': 0.4},
            'phase': {'n_phases': 2, 'method': 'watershed'}
        }
    }
    
    comparison_results = {}
    
    for method_name, params in methods.items():
        print(f"\nAnalyzing with {method_name} method...")
        
        analyzer = MicrostructuralAnalyzer(image_path, scale)
        results = analyzer.run_complete_analysis(
            preprocess_params=params['preprocess'],
            grain_params=params['grain'],
            defect_params=params['defect'],
            phase_params=params['phase']
        )
        
        comparison_results[method_name] = results
        
        # Save results
        analyzer.generate_report(save_path=f"comparison_{method_name.lower()}_report.txt")
    
    # Create comparison summary
    print("\nCOMPARISON SUMMARY")
    print("="*50)
    for method, results in comparison_results.items():
        print(f"\n{method} Method:")
        if 'grain_analysis' in results:
            ga = results['grain_analysis']
            print(f"  Grains: {ga['total_grains']}, Avg Size: {ga['avg_grain_size_microns']:.2f} μm")
        if 'defect_analysis' in results:
            da = results['defect_analysis']
            print(f"  Defects: {da['total_pores']} pores, {da['total_cracks']} cracks")
    
    return comparison_results

def interactive_analysis():
    """Interactive analysis with user input."""
    print("Interactive Microstructural Analysis")
    print("="*50)
    
    # Get user input
    image_path = input("Enter image path (or press Enter for sample): ").strip()
    if not image_path:
        # Create sample
        image_path = "interactive_sample.png"
        create_sample_microstructure(save_path=image_path)
        print(f"Created sample image: {image_path}")
    
    try:
        scale = float(input("Enter scale (pixels per micron) [default: 1.0]: ") or "1.0")
    except ValueError:
        scale = 1.0
    
    # Analysis options
    print("\nSelect analysis options:")
    run_grains = input("Analyze grains? (y/n) [default: y]: ").lower() != 'n'
    run_defects = input("Detect defects? (y/n) [default: y]: ").lower() != 'n'
    run_phases = input("Analyze phases? (y/n) [default: y]: ").lower() != 'n'
    run_texture = input("Calculate texture features? (y/n) [default: y]: ").lower() != 'n'
    
    # Initialize analyzer
    analyzer = MicrostructuralAnalyzer(image_path, scale)
    
    # Preprocessing
    print("\nPreprocessing image...")
    analyzer.preprocess_image()
    
    # Run selected analyses
    if run_grains:
        print("Analyzing grains...")
        analyzer.detect_grain_boundaries()
        analyzer.segment_grains()
        analyzer.analyze_grains()
    
    if run_defects:
        print("Detecting defects...")
        analyzer.detect_defects()
    
    if run_phases:
        print("Analyzing phases...")
        n_phases = int(input("Number of phases to detect [default: 2]: ") or "2")
        analyzer.analyze_phases(n_phases=n_phases)
    
    if run_texture:
        print("Calculating texture features...")
        analyzer.calculate_texture_features()
    
    # Generate outputs
    analyzer.generate_report(save_path="interactive_analysis_report.txt")
    analyzer.visualize_results(save_path="interactive_analysis_visualization.png")
    
    print("\nInteractive analysis complete!")
    return analyzer

def main():
    """Main function demonstrating different usage scenarios."""
    print("Microstructural Analysis Examples")
    print("="*50)
    
    # Example 1: Sample image analysis
    print("\n1. Sample Image Analysis")
    print("-"*30)
    analyzer1, results1 = analyze_sample_image()
    
    # Example 2: Method comparison (using the sample)
    print("\n\n2. Processing Method Comparison")
    print("-"*30)
    comparison_results = compare_processing_methods("advanced_sample.png", scale=2.5)
    
    # Example 3: Interactive analysis
    print("\n\n3. Interactive Analysis")
    print("-"*30)
    choice = input("Run interactive analysis? (y/n): ").lower()
    if choice == 'y':
        interactive_analyzer = interactive_analysis()
    
    print("\n\nAll examples completed!")
    print("Check the generated files for detailed results:")
    print("- *_report.txt: Text reports")
    print("- *_analysis.png: Visualization plots")
    print("- batch_analysis_summary.csv: Batch analysis summary (if applicable)")

if __name__ == "__main__":
    main()
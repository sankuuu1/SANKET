"""
Microstructural Analysis Tool using Computer Vision
==================================================

This module provides comprehensive analysis of microstructural images including:
- Grain detection and characterization
- Defect identification (pores, cracks, inclusions)
- Phase analysis
- Quantitative feature extraction
- Statistical analysis and reporting

Author: AI Assistant
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import (
    filters, morphology, measure, segmentation, 
    feature, restoration, exposure, util
)
from skimage.segmentation import watershed, mark_boundaries
from skimage.feature import peak_local_maxima, corner_harris, corner_peaks
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class MicrostructuralAnalyzer:
    """
    A comprehensive tool for microstructural analysis of metallographic images.
    """
    
    def __init__(self, image_path=None, scale_pixels_per_micron=1.0):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        image_path : str, optional
            Path to the microstructural image
        scale_pixels_per_micron : float
            Calibration factor for converting pixels to microns
        """
        self.image_path = image_path
        self.scale = scale_pixels_per_micron
        self.original_image = None
        self.processed_image = None
        self.grain_labels = None
        self.grain_properties = None
        self.defects = None
        self.phases = None
        self.results = {}
        
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load and preprocess the microstructural image."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale for processing
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        print(f"Image loaded: {self.processed_image.shape}")
    
    def preprocess_image(self, denoise=True, enhance_contrast=True, 
                        gaussian_sigma=1.0, clahe_clip_limit=2.0):
        """
        Preprocess the image for better analysis.
        
        Parameters:
        -----------
        denoise : bool
            Apply denoising filter
        enhance_contrast : bool
            Apply contrast enhancement
        gaussian_sigma : float
            Sigma for Gaussian filter
        clahe_clip_limit : float
            Clip limit for CLAHE contrast enhancement
        """
        if self.processed_image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        img = self.processed_image.copy()
        
        # Denoising
        if denoise:
            img = restoration.denoise_bilateral(img, sigma_color=0.1, sigma_spatial=1.0)
            img = (img * 255).astype(np.uint8)
        
        # Contrast enhancement
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8,8))
            img = clahe.apply(img)
        
        # Gaussian smoothing
        if gaussian_sigma > 0:
            img = filters.gaussian(img, sigma=gaussian_sigma, preserve_range=True).astype(np.uint8)
        
        self.processed_image = img
        print("Image preprocessing completed")
        return img
    
    def detect_grain_boundaries(self, method='sobel', threshold_factor=0.5):
        """
        Detect grain boundaries using edge detection.
        
        Parameters:
        -----------
        method : str
            Edge detection method ('sobel', 'canny', 'scharr')
        threshold_factor : float
            Factor for automatic threshold calculation
        """
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        img = self.processed_image
        
        if method == 'sobel':
            edges = filters.sobel(img)
        elif method == 'canny':
            # Auto threshold for Canny
            v = np.median(img)
            lower = int(max(0, (1.0 - threshold_factor) * v))
            upper = int(min(255, (1.0 + threshold_factor) * v))
            edges = feature.canny(img, low_threshold=lower, high_threshold=upper)
        elif method == 'scharr':
            edges = filters.scharr(img)
        else:
            raise ValueError("Method must be 'sobel', 'canny', or 'scharr'")
        
        # Clean up edges
        if method != 'canny':
            threshold = filters.threshold_otsu(edges)
            edges = edges > threshold * threshold_factor
        
        # Morphological operations to clean boundaries
        edges = morphology.binary_closing(edges, morphology.disk(1))
        edges = morphology.remove_small_objects(edges, min_size=10)
        
        self.grain_boundaries = edges
        print(f"Grain boundaries detected using {method}")
        return edges
    
    def segment_grains(self, min_distance=20, watershed_compactness=0.1):
        """
        Segment individual grains using watershed algorithm.
        
        Parameters:
        -----------
        min_distance : int
            Minimum distance between grain centers
        watershed_compactness : float
            Compactness parameter for watershed
        """
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        img = self.processed_image
        
        # Create distance transform
        if not hasattr(self, 'grain_boundaries'):
            self.detect_grain_boundaries()
        
        # Fill holes in boundaries and invert
        boundaries_filled = ndi.binary_fill_holes(self.grain_boundaries)
        distance = ndi.distance_transform_edt(~boundaries_filled)
        
        # Find local maxima (grain centers)
        local_maxima = peak_local_maxima(distance, min_distance=min_distance, 
                                       threshold_abs=distance.max()*0.1,
                                       indices=False)
        markers = measure.label(local_maxima)
        
        # Watershed segmentation
        labels = watershed(-distance, markers, mask=~boundaries_filled, 
                          compactness=watershed_compactness)
        
        self.grain_labels = labels
        print(f"Segmented {len(np.unique(labels)) - 1} grains")
        return labels
    
    def analyze_grains(self):
        """Analyze grain properties and extract features."""
        if self.grain_labels is None:
            self.segment_grains()
        
        # Extract region properties
        properties = measure.regionprops(self.grain_labels, 
                                       intensity_image=self.processed_image)
        
        grain_data = []
        for prop in properties:
            if prop.area > 50:  # Filter out very small regions
                grain_data.append({
                    'label': prop.label,
                    'area_pixels': prop.area,
                    'area_microns': prop.area / (self.scale ** 2),
                    'perimeter_pixels': prop.perimeter,
                    'perimeter_microns': prop.perimeter / self.scale,
                    'equivalent_diameter_pixels': prop.equivalent_diameter,
                    'equivalent_diameter_microns': prop.equivalent_diameter / self.scale,
                    'major_axis_length': prop.major_axis_length / self.scale,
                    'minor_axis_length': prop.minor_axis_length / self.scale,
                    'aspect_ratio': prop.major_axis_length / prop.minor_axis_length,
                    'solidity': prop.solidity,
                    'eccentricity': prop.eccentricity,
                    'circularity': 4 * np.pi * prop.area / (prop.perimeter ** 2),
                    'mean_intensity': prop.mean_intensity,
                    'centroid_x': prop.centroid[1],
                    'centroid_y': prop.centroid[0]
                })
        
        self.grain_properties = pd.DataFrame(grain_data)
        
        # Calculate summary statistics
        if len(grain_data) > 0:
            self.results['grain_analysis'] = {
                'total_grains': len(grain_data),
                'avg_grain_size_microns': self.grain_properties['equivalent_diameter_microns'].mean(),
                'std_grain_size_microns': self.grain_properties['equivalent_diameter_microns'].std(),
                'median_grain_size_microns': self.grain_properties['equivalent_diameter_microns'].median(),
                'min_grain_size_microns': self.grain_properties['equivalent_diameter_microns'].min(),
                'max_grain_size_microns': self.grain_properties['equivalent_diameter_microns'].max(),
                'avg_aspect_ratio': self.grain_properties['aspect_ratio'].mean(),
                'avg_circularity': self.grain_properties['circularity'].mean(),
                'total_grain_area_fraction': self.grain_properties['area_pixels'].sum() / 
                                           (self.processed_image.shape[0] * self.processed_image.shape[1])
            }
        
        print(f"Grain analysis completed: {len(grain_data)} grains analyzed")
        return self.grain_properties
    
    def detect_defects(self, pore_threshold=0.3, crack_min_length=20):
        """
        Detect various types of defects in the microstructure.
        
        Parameters:
        -----------
        pore_threshold : float
            Threshold for pore detection (relative to image intensity)
        crack_min_length : int
            Minimum length for crack detection
        """
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        img = self.processed_image
        defects = {}
        
        # 1. Pore Detection (dark circular/elliptical regions)
        # Use adaptive thresholding for pore detection
        binary_threshold = filters.threshold_otsu(img)
        pore_mask = img < (binary_threshold * pore_threshold)
        
        # Clean up pore mask
        pore_mask = morphology.remove_small_objects(pore_mask, min_size=10)
        pore_mask = morphology.binary_closing(pore_mask, morphology.disk(3))
        
        # Label and analyze pores
        pore_labels = measure.label(pore_mask)
        pore_props = measure.regionprops(pore_labels)
        
        pores = []
        for prop in pore_props:
            if prop.solidity > 0.7 and prop.area > 20:  # Circular-ish and significant size
                pores.append({
                    'type': 'pore',
                    'area_pixels': prop.area,
                    'area_microns': prop.area / (self.scale ** 2),
                    'equivalent_diameter_microns': prop.equivalent_diameter / self.scale,
                    'circularity': 4 * np.pi * prop.area / (prop.perimeter ** 2),
                    'centroid': prop.centroid
                })
        
        defects['pores'] = pores
        
        # 2. Crack Detection (elongated dark regions)
        # Use morphological operations to detect linear features
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (crack_min_length, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, crack_min_length))
        
        # Detect horizontal and vertical cracks
        crack_h = cv2.morphologyEx(255 - img, cv2.MORPH_OPEN, kernel_line_h)
        crack_v = cv2.morphologyEx(255 - img, cv2.MORPH_OPEN, kernel_line_v)
        cracks = cv2.bitwise_or(crack_h, crack_v)
        
        # Threshold and clean
        _, cracks = cv2.threshold(cracks, 30, 255, cv2.THRESH_BINARY)
        cracks = morphology.remove_small_objects(cracks.astype(bool), min_size=crack_min_length)
        
        crack_labels = measure.label(cracks)
        crack_props = measure.regionprops(crack_labels)
        
        crack_list = []
        for prop in crack_props:
            if prop.major_axis_length / prop.minor_axis_length > 3:  # Elongated
                crack_list.append({
                    'type': 'crack',
                    'length_microns': prop.major_axis_length / self.scale,
                    'width_microns': prop.minor_axis_length / self.scale,
                    'area_microns': prop.area / (self.scale ** 2),
                    'aspect_ratio': prop.major_axis_length / prop.minor_axis_length,
                    'centroid': prop.centroid
                })
        
        defects['cracks'] = crack_list
        
        # 3. Inclusion Detection (bright or unusually colored regions)
        # Detect regions with significantly different intensity
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        
        # Bright inclusions
        bright_mask = img > (mean_intensity + 2 * std_intensity)
        bright_mask = morphology.remove_small_objects(bright_mask, min_size=15)
        
        bright_labels = measure.label(bright_mask)
        bright_props = measure.regionprops(bright_labels)
        
        inclusions = []
        for prop in bright_props:
            inclusions.append({
                'type': 'bright_inclusion',
                'area_pixels': prop.area,
                'area_microns': prop.area / (self.scale ** 2),
                'equivalent_diameter_microns': prop.equivalent_diameter / self.scale,
                'mean_intensity': prop.mean_intensity,
                'centroid': prop.centroid
            })
        
        defects['inclusions'] = inclusions
        
        self.defects = defects
        
        # Summary statistics
        self.results['defect_analysis'] = {
            'total_pores': len(pores),
            'total_cracks': len(crack_list),
            'total_inclusions': len(inclusions),
            'pore_area_fraction': sum(p['area_pixels'] for p in pores) / 
                                (img.shape[0] * img.shape[1]) if pores else 0,
            'avg_pore_size_microns': np.mean([p['equivalent_diameter_microns'] for p in pores]) if pores else 0,
            'total_crack_length_microns': sum(c['length_microns'] for c in crack_list) if crack_list else 0
        }
        
        print(f"Defect detection completed: {len(pores)} pores, {len(crack_list)} cracks, {len(inclusions)} inclusions")
        return defects
    
    def analyze_phases(self, n_phases=2, method='kmeans'):
        """
        Identify and analyze different phases in the microstructure.
        
        Parameters:
        -----------
        n_phases : int
            Number of phases to identify
        method : str
            Method for phase identification ('kmeans', 'otsu', 'watershed')
        """
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        img = self.processed_image
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            
            # Prepare data for clustering
            pixel_values = img.reshape((-1, 1))
            pixel_values = np.float32(pixel_values)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_phases, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixel_values)
            labels = labels.reshape(img.shape)
            
        elif method == 'otsu':
            # Multi-level Otsu thresholding
            if n_phases == 2:
                threshold = filters.threshold_otsu(img)
                labels = (img > threshold).astype(int)
            else:
                thresholds = filters.threshold_multiotsu(img, classes=n_phases)
                labels = np.digitize(img, thresholds)
        
        elif method == 'watershed':
            # Watershed-based phase separation
            # Use gradient as elevation map
            gradient = filters.sobel(img)
            markers = np.zeros_like(img)
            
            # Create markers for each phase based on intensity
            for i in range(n_phases):
                threshold = np.percentile(img, (i+1) * 100 / (n_phases + 1))
                markers[img < threshold] = i + 1
            
            labels = watershed(gradient, markers)
        
        # Analyze each phase
        phase_data = []
        for phase_id in range(n_phases):
            phase_mask = (labels == phase_id)
            phase_area = np.sum(phase_mask)
            
            if phase_area > 0:
                phase_props = measure.regionprops(phase_mask.astype(int))
                mean_intensity = np.mean(img[phase_mask])
                
                phase_data.append({
                    'phase_id': phase_id,
                    'area_pixels': phase_area,
                    'area_fraction': phase_area / (img.shape[0] * img.shape[1]),
                    'mean_intensity': mean_intensity,
                    'std_intensity': np.std(img[phase_mask])
                })
        
        self.phases = {
            'labels': labels,
            'data': pd.DataFrame(phase_data),
            'method': method,
            'n_phases': n_phases
        }
        
        self.results['phase_analysis'] = {
            'n_phases_detected': len(phase_data),
            'phase_fractions': {f'phase_{p["phase_id"]}': p['area_fraction'] 
                              for p in phase_data}
        }
        
        print(f"Phase analysis completed: {len(phase_data)} phases identified")
        return self.phases
    
    def calculate_texture_features(self):
        """Calculate texture features using GLCM and LBP."""
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
        
        img = self.processed_image
        
        # GLCM features
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = greycomatrix(img, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        texture_features = {
            'contrast': np.mean(greycoprops(glcm, 'contrast')),
            'dissimilarity': np.mean(greycoprops(glcm, 'dissimilarity')),
            'homogeneity': np.mean(greycoprops(glcm, 'homogeneity')),
            'energy': np.mean(greycoprops(glcm, 'energy')),
            'correlation': np.mean(greycoprops(glcm, 'correlation')),
            'asm': np.mean(greycoprops(glcm, 'ASM'))
        }
        
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                  range=(0, n_points + 2), density=True)
        
        texture_features.update({
            'lbp_uniformity': np.sum(lbp_hist ** 2),
            'lbp_entropy': -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
        })
        
        self.results['texture_features'] = texture_features
        print("Texture analysis completed")
        return texture_features
    
    def generate_report(self, save_path=None):
        """Generate a comprehensive analysis report."""
        if not self.results:
            print("No analysis results available. Run analysis methods first.")
            return
        
        report = []
        report.append("="*60)
        report.append("MICROSTRUCTURAL ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Image: {self.image_path}")
        report.append(f"Scale: {self.scale} pixels/micron")
        report.append("")
        
        # Grain Analysis
        if 'grain_analysis' in self.results:
            report.append("GRAIN ANALYSIS")
            report.append("-" * 40)
            ga = self.results['grain_analysis']
            report.append(f"Total Grains: {ga['total_grains']}")
            report.append(f"Average Grain Size: {ga['avg_grain_size_microns']:.2f} ± {ga['std_grain_size_microns']:.2f} μm")
            report.append(f"Median Grain Size: {ga['median_grain_size_microns']:.2f} μm")
            report.append(f"Grain Size Range: {ga['min_grain_size_microns']:.2f} - {ga['max_grain_size_microns']:.2f} μm")
            report.append(f"Average Aspect Ratio: {ga['avg_aspect_ratio']:.2f}")
            report.append(f"Average Circularity: {ga['avg_circularity']:.2f}")
            report.append(f"Grain Area Fraction: {ga['total_grain_area_fraction']:.3f}")
            report.append("")
        
        # Defect Analysis
        if 'defect_analysis' in self.results:
            report.append("DEFECT ANALYSIS")
            report.append("-" * 40)
            da = self.results['defect_analysis']
            report.append(f"Total Pores: {da['total_pores']}")
            report.append(f"Total Cracks: {da['total_cracks']}")
            report.append(f"Total Inclusions: {da['total_inclusions']}")
            report.append(f"Pore Area Fraction: {da['pore_area_fraction']:.4f}")
            if da['avg_pore_size_microns'] > 0:
                report.append(f"Average Pore Size: {da['avg_pore_size_microns']:.2f} μm")
            if da['total_crack_length_microns'] > 0:
                report.append(f"Total Crack Length: {da['total_crack_length_microns']:.2f} μm")
            report.append("")
        
        # Phase Analysis
        if 'phase_analysis' in self.results:
            report.append("PHASE ANALYSIS")
            report.append("-" * 40)
            pa = self.results['phase_analysis']
            report.append(f"Phases Detected: {pa['n_phases_detected']}")
            for phase, fraction in pa['phase_fractions'].items():
                report.append(f"{phase}: {fraction:.3f} ({fraction*100:.1f}%)")
            report.append("")
        
        # Texture Features
        if 'texture_features' in self.results:
            report.append("TEXTURE FEATURES")
            report.append("-" * 40)
            tf = self.results['texture_features']
            for feature, value in tf.items():
                report.append(f"{feature}: {value:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        print(report_text)
        return report_text
    
    def visualize_results(self, figsize=(20, 15), save_path=None):
        """Create comprehensive visualization of analysis results."""
        if self.original_image is None:
            raise ValueError("No image loaded")
        
        # Create subplots
        fig = plt.figure(figsize=figsize)
        
        # Original image
        plt.subplot(3, 4, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed image
        plt.subplot(3, 4, 2)
        plt.imshow(self.processed_image, cmap='gray')
        plt.title('Processed Image')
        plt.axis('off')
        
        # Grain boundaries
        if hasattr(self, 'grain_boundaries'):
            plt.subplot(3, 4, 3)
            plt.imshow(self.grain_boundaries, cmap='gray')
            plt.title('Grain Boundaries')
            plt.axis('off')
        
        # Grain segmentation
        if self.grain_labels is not None:
            plt.subplot(3, 4, 4)
            plt.imshow(mark_boundaries(self.processed_image, self.grain_labels))
            plt.title('Grain Segmentation')
            plt.axis('off')
        
        # Grain size distribution
        if self.grain_properties is not None and len(self.grain_properties) > 0:
            plt.subplot(3, 4, 5)
            plt.hist(self.grain_properties['equivalent_diameter_microns'], 
                    bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Grain Size (μm)')
            plt.ylabel('Frequency')
            plt.title('Grain Size Distribution')
            plt.grid(True, alpha=0.3)
        
        # Defects visualization
        if self.defects is not None:
            plt.subplot(3, 4, 6)
            img_defects = self.processed_image.copy()
            
            # Mark pores in red
            for pore in self.defects.get('pores', []):
                y, x = pore['centroid']
                cv2.circle(img_defects, (int(x), int(y)), 3, 255, -1)
            
            # Mark cracks in blue
            for crack in self.defects.get('cracks', []):
                y, x = crack['centroid']
                cv2.circle(img_defects, (int(x), int(y)), 2, 128, -1)
            
            plt.imshow(img_defects, cmap='gray')
            plt.title('Defects (Pores: white, Cracks: gray)')
            plt.axis('off')
        
        # Phase analysis
        if self.phases is not None:
            plt.subplot(3, 4, 7)
            plt.imshow(self.phases['labels'], cmap='viridis')
            plt.title('Phase Segmentation')
            plt.axis('off')
            
            # Phase fractions pie chart
            plt.subplot(3, 4, 8)
            if len(self.phases['data']) > 0:
                phases = self.phases['data']
                plt.pie(phases['area_fraction'], 
                       labels=[f'Phase {i}' for i in phases['phase_id']], 
                       autopct='%1.1f%%')
                plt.title('Phase Fractions')
        
        # Grain aspect ratio vs size
        if self.grain_properties is not None and len(self.grain_properties) > 0:
            plt.subplot(3, 4, 9)
            plt.scatter(self.grain_properties['equivalent_diameter_microns'],
                       self.grain_properties['aspect_ratio'], 
                       alpha=0.6, s=30)
            plt.xlabel('Grain Size (μm)')
            plt.ylabel('Aspect Ratio')
            plt.title('Grain Size vs Aspect Ratio')
            plt.grid(True, alpha=0.3)
        
        # Circularity distribution
        if self.grain_properties is not None and len(self.grain_properties) > 0:
            plt.subplot(3, 4, 10)
            plt.hist(self.grain_properties['circularity'], 
                    bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('Circularity')
            plt.ylabel('Frequency')
            plt.title('Grain Circularity Distribution')
            plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(3, 4, 11)
        plt.axis('off')
        summary_text = []
        if 'grain_analysis' in self.results:
            ga = self.results['grain_analysis']
            summary_text.extend([
                f"Grains: {ga['total_grains']}",
                f"Avg Size: {ga['avg_grain_size_microns']:.1f} μm",
                f"Size Range: {ga['min_grain_size_microns']:.1f}-{ga['max_grain_size_microns']:.1f} μm"
            ])
        
        if 'defect_analysis' in self.results:
            da = self.results['defect_analysis']
            summary_text.extend([
                f"Pores: {da['total_pores']}",
                f"Cracks: {da['total_cracks']}",
                f"Inclusions: {da['total_inclusions']}"
            ])
        
        if summary_text:
            plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
            plt.title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        return fig
    
    def run_complete_analysis(self, preprocess_params=None, 
                            grain_params=None, defect_params=None, 
                            phase_params=None):
        """
        Run complete microstructural analysis pipeline.
        
        Parameters:
        -----------
        preprocess_params : dict
            Parameters for preprocessing
        grain_params : dict
            Parameters for grain analysis
        defect_params : dict
            Parameters for defect detection
        phase_params : dict
            Parameters for phase analysis
        """
        print("Starting complete microstructural analysis...")
        
        # Set default parameters
        preprocess_params = preprocess_params or {}
        grain_params = grain_params or {}
        defect_params = defect_params or {}
        phase_params = phase_params or {}
        
        # 1. Preprocessing
        print("\n1. Preprocessing image...")
        self.preprocess_image(**preprocess_params)
        
        # 2. Grain analysis
        print("\n2. Analyzing grains...")
        self.detect_grain_boundaries()
        self.segment_grains(**grain_params)
        self.analyze_grains()
        
        # 3. Defect detection
        print("\n3. Detecting defects...")
        self.detect_defects(**defect_params)
        
        # 4. Phase analysis
        print("\n4. Analyzing phases...")
        self.analyze_phases(**phase_params)
        
        # 5. Texture analysis
        print("\n5. Calculating texture features...")
        self.calculate_texture_features()
        
        print("\nComplete analysis finished!")
        return self.results


def create_sample_microstructure(size=(512, 512), n_grains=50, noise_level=0.1, 
                                save_path='sample_microstructure.png'):
    """
    Create a synthetic microstructural image for testing.
    
    Parameters:
    -----------
    size : tuple
        Image dimensions
    n_grains : int
        Number of grains to simulate
    noise_level : float
        Amount of noise to add
    save_path : str
        Path to save the synthetic image
    """
    from scipy.spatial import Voronoi
    
    # Generate random grain centers
    np.random.seed(42)
    points = np.random.rand(n_grains, 2) * np.array(size)
    
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create image
    img = np.zeros(size)
    y, x = np.mgrid[:size[0], :size[1]]
    
    # Assign each pixel to nearest grain center
    for i, point in enumerate(points):
        distances = np.sqrt((x - point[1])**2 + (y - point[0])**2)
        mask = True
        for j, other_point in enumerate(points):
            if i != j:
                other_distances = np.sqrt((x - other_point[1])**2 + (y - other_point[0])**2)
                mask &= distances <= other_distances
        
        # Assign grain intensity
        grain_intensity = 100 + i * 3  # Different intensities for different grains
        img[mask] = grain_intensity
    
    # Add grain boundaries
    gradient = filters.sobel(img)
    boundaries = gradient > filters.threshold_otsu(gradient) * 0.5
    img[boundaries] = 50  # Dark boundaries
    
    # Add some defects
    # Pores (dark circles)
    for _ in range(5):
        center = np.random.randint(20, size[0]-20, 2)
        radius = np.random.randint(3, 8)
        rr, cc = morphology.disk(radius)
        rr, cc = rr + center[0], cc + center[1]
        valid = (rr >= 0) & (rr < size[0]) & (cc >= 0) & (cc < size[1])
        img[rr[valid], cc[valid]] = 20
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, size)
        img = np.clip(img + noise, 0, 255)
    
    # Convert to uint8 and save
    img = img.astype(np.uint8)
    cv2.imwrite(save_path, img)
    
    print(f"Sample microstructure created and saved to {save_path}")
    return img


if __name__ == "__main__":
    # Example usage
    print("Microstructural Analysis Tool")
    print("=============================")
    
    # Create sample data if needed
    sample_path = "sample_microstructure.png"
    create_sample_microstructure(save_path=sample_path)
    
    # Initialize analyzer
    analyzer = MicrostructuralAnalyzer(sample_path, scale_pixels_per_micron=2.0)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Generate report
    analyzer.generate_report(save_path="microstructure_report.txt")
    
    # Create visualizations
    analyzer.visualize_results(save_path="microstructure_analysis.png")
    
    print("\nAnalysis complete! Check the generated files for results.")
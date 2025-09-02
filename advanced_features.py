"""
Advanced Features for Microstructural Analysis
==============================================

This module provides additional advanced features for microstructural analysis
including machine learning approaches, advanced statistical analysis, and 
specialized algorithms for specific materials.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology, filters
import warnings
warnings.filterwarnings('ignore')


class AdvancedMicrostructuralAnalyzer:
    """
    Advanced analyzer with machine learning and statistical methods.
    """
    
    def __init__(self, base_analyzer):
        """
        Initialize with a base MicrostructuralAnalyzer instance.
        
        Parameters:
        -----------
        base_analyzer : MicrostructuralAnalyzer
            Base analyzer with completed analysis
        """
        self.base = base_analyzer
        self.ml_features = None
        self.anomaly_scores = None
        self.grain_clusters = None
    
    def extract_ml_features(self):
        """Extract comprehensive features for machine learning."""
        if self.base.grain_properties is None:
            raise ValueError("Grain analysis must be completed first")
        
        grains = self.base.grain_properties
        
        # Basic geometric features
        features = grains[['area_microns', 'perimeter_microns', 'equivalent_diameter_microns',
                          'major_axis_length', 'minor_axis_length', 'aspect_ratio',
                          'solidity', 'eccentricity', 'circularity', 'mean_intensity']].copy()
        
        # Add derived features
        features['elongation'] = 1 - (grains['minor_axis_length'] / grains['major_axis_length'])
        features['compactness'] = grains['area_microns'] / (grains['perimeter_microns'] ** 2)
        features['form_factor'] = 4 * np.pi * grains['area_microns'] / (grains['perimeter_microns'] ** 2)
        features['roundness'] = 4 * grains['area_microns'] / (np.pi * grains['major_axis_length'] ** 2)
        
        # Neighborhood features (if spatial information available)
        if 'centroid_x' in grains.columns:
            features = self._add_spatial_features(features, grains)
        
        # Statistical features
        features = self._add_statistical_features(features, grains)
        
        self.ml_features = features
        return features
    
    def _add_spatial_features(self, features, grains):
        """Add spatial neighborhood features."""
        centroids = grains[['centroid_x', 'centroid_y']].values
        
        if len(centroids) < 2:
            return features
        
        # Calculate distances to nearest neighbors
        distances = pdist(centroids)
        dist_matrix = squareform(distances)
        
        # For each grain, find nearest neighbors
        nearest_distances = []
        avg_neighbor_sizes = []
        neighbor_counts = []
        
        for i in range(len(grains)):
            # Find 5 nearest neighbors (excluding self)
            neighbor_indices = np.argsort(dist_matrix[i])[1:6]
            neighbor_dists = dist_matrix[i][neighbor_indices]
            
            nearest_distances.append(np.mean(neighbor_dists))
            
            # Average size of neighbors
            neighbor_sizes = grains.iloc[neighbor_indices]['equivalent_diameter_microns']
            avg_neighbor_sizes.append(np.mean(neighbor_sizes))
            
            # Count neighbors within certain distance
            threshold_distance = 50  # microns
            close_neighbors = np.sum(dist_matrix[i] < threshold_distance) - 1
            neighbor_counts.append(close_neighbors)
        
        features['avg_neighbor_distance'] = nearest_distances
        features['avg_neighbor_size'] = avg_neighbor_sizes
        features['neighbor_count'] = neighbor_counts
        features['local_density'] = features['neighbor_count'] / (np.pi * (features['avg_neighbor_distance'] ** 2))
        
        return features
    
    def _add_statistical_features(self, features, grains):
        """Add statistical features based on grain population."""
        # Percentile ranks
        for col in ['area_microns', 'equivalent_diameter_microns', 'aspect_ratio']:
            if col in features.columns:
                percentiles = stats.rankdata(features[col]) / len(features) * 100
                features[f'{col}_percentile'] = percentiles
        
        # Z-scores
        for col in ['area_microns', 'equivalent_diameter_microns', 'circularity']:
            if col in features.columns:
                z_scores = stats.zscore(features[col])
                features[f'{col}_zscore'] = z_scores
        
        return features
    
    def detect_anomalous_grains(self, contamination=0.1, method='isolation_forest'):
        """
        Detect anomalous grains using machine learning.
        
        Parameters:
        -----------
        contamination : float
            Expected fraction of outliers
        method : str
            Method for anomaly detection ('isolation_forest', 'local_outlier')
        """
        if self.ml_features is None:
            self.extract_ml_features()
        
        # Prepare features
        feature_cols = self.ml_features.select_dtypes(include=[np.number]).columns
        X = self.ml_features[feature_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.score_samples(X_scaled)
        
        elif method == 'local_outlier':
            from sklearn.neighbors import LocalOutlierFactor
            detector = LocalOutlierFactor(contamination=contamination)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.negative_outlier_factor_
        
        # Store results
        self.anomaly_scores = pd.DataFrame({
            'grain_label': self.base.grain_properties['label'],
            'is_anomaly': anomaly_labels == -1,
            'anomaly_score': anomaly_scores
        })
        
        print(f"Detected {np.sum(anomaly_labels == -1)} anomalous grains using {method}")
        return self.anomaly_scores
    
    def cluster_grains(self, n_clusters=3, method='kmeans'):
        """
        Cluster grains based on their features.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (for kmeans)
        method : str
            Clustering method ('kmeans', 'dbscan', 'hierarchical')
        """
        if self.ml_features is None:
            self.extract_ml_features()
        
        # Prepare features
        feature_cols = ['area_microns', 'aspect_ratio', 'circularity', 'solidity', 'eccentricity']
        available_cols = [col for col in feature_cols if col in self.ml_features.columns]
        
        if not available_cols:
            raise ValueError("No suitable features available for clustering")
        
        X = self.ml_features[available_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X_scaled)
        
        elif method == 'dbscan':
            # Estimate eps using k-distance graph
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(X_scaled)
            distances, indices = neighbors_fit.kneighbors(X_scaled)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]  # k=4 distances
            eps = np.percentile(distances, 90)
            
            clusterer = DBSCAN(eps=eps, min_samples=3)
            cluster_labels = clusterer.fit_predict(X_scaled)
        
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Store results
        self.grain_clusters = pd.DataFrame({
            'grain_label': self.base.grain_properties['label'],
            'cluster': cluster_labels,
            'cluster_name': [f'Cluster_{i}' if i >= 0 else 'Noise' for i in cluster_labels]
        })
        
        # Add cluster information to features
        self.ml_features['cluster'] = cluster_labels
        
        print(f"Clustered grains using {method}: {len(np.unique(cluster_labels))} clusters")
        return self.grain_clusters
    
    def analyze_grain_size_distribution(self):
        """Detailed grain size distribution analysis."""
        if self.base.grain_properties is None:
            raise ValueError("Grain analysis must be completed first")
        
        sizes = self.base.grain_properties['equivalent_diameter_microns']
        
        # Fit different distributions
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'weibull': stats.weibull_min
        }
        
        distribution_results = {}
        
        for name, dist in distributions.items():
            try:
                # Fit distribution
                params = dist.fit(sizes)
                
                # Calculate goodness of fit (KS test)
                ks_stat, ks_p = stats.kstest(sizes, lambda x: dist.cdf(x, *params))
                
                # Calculate AIC
                log_likelihood = np.sum(dist.logpdf(sizes, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                
                distribution_results[name] = {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'aic': aic,
                    'log_likelihood': log_likelihood
                }
            except Exception as e:
                print(f"Failed to fit {name} distribution: {e}")
        
        # Find best fitting distribution
        if distribution_results:
            best_dist = min(distribution_results.items(), key=lambda x: x[1]['aic'])
            print(f"Best fitting distribution: {best_dist[0]} (AIC: {best_dist[1]['aic']:.2f})")
        
        return distribution_results
    
    def calculate_spatial_statistics(self):
        """Calculate spatial statistics for grain distribution."""
        if self.base.grain_properties is None or 'centroid_x' not in self.base.grain_properties.columns:
            raise ValueError("Grain analysis with spatial information required")
        
        centroids = self.base.grain_properties[['centroid_x', 'centroid_y']].values
        
        # Calculate nearest neighbor distances
        from scipy.spatial.distance import cdist
        distances = cdist(centroids, centroids)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        
        # Clark-Evans test for spatial randomness
        observed_mean = np.mean(nearest_distances)
        area = self.base.processed_image.shape[0] * self.base.processed_image.shape[1]
        density = len(centroids) / area
        expected_mean = 1 / (2 * np.sqrt(density))
        
        clark_evans_R = observed_mean / expected_mean
        
        # Interpretation
        if clark_evans_R < 1:
            pattern = "Clustered"
        elif clark_evans_R > 1:
            pattern = "Dispersed"
        else:
            pattern = "Random"
        
        # Ripley's K function (simplified)
        def ripleys_k(points, distances_range, area):
            k_values = []
            for r in distances_range:
                count = 0
                for i, point in enumerate(points):
                    # Count points within distance r
                    dists = np.sqrt(np.sum((points - point)**2, axis=1))
                    count += np.sum(dists <= r) - 1  # Exclude self
                
                k = (area / (len(points) * (len(points) - 1))) * count
                k_values.append(k)
            
            return np.array(k_values)
        
        r_values = np.linspace(10, 100, 20)
        k_values = ripleys_k(centroids, r_values, area)
        
        spatial_stats = {
            'clark_evans_R': clark_evans_R,
            'spatial_pattern': pattern,
            'mean_nearest_distance': observed_mean,
            'grain_density': density,
            'ripleys_k': {'r_values': r_values, 'k_values': k_values}
        }
        
        return spatial_stats
    
    def grain_boundary_analysis(self):
        """Analyze grain boundary characteristics."""
        if not hasattr(self.base, 'grain_boundaries'):
            self.base.detect_grain_boundaries()
        
        boundaries = self.base.grain_boundaries
        
        # Skeletonize boundaries
        skeleton = morphology.skeletonize(boundaries)
        
        # Analyze boundary segments
        boundary_props = measure.regionprops(measure.label(skeleton))
        
        # Calculate boundary statistics
        boundary_lengths = [prop.major_axis_length for prop in boundary_props if prop.area > 5]
        boundary_orientations = [prop.orientation for prop in boundary_props if prop.area > 5]
        
        # Junction analysis (triple points)
        # Dilate skeleton to find intersections
        dilated = morphology.dilation(skeleton, morphology.disk(2))
        intersections = morphology.label(dilated)
        
        # Count junctions (regions with more than 2 neighbors)
        junction_count = 0
        for region in measure.regionprops(intersections):
            if region.area > 4:  # Significant intersection
                junction_count += 1
        
        boundary_analysis = {
            'total_boundary_length': np.sum(boundaries) / self.base.scale,
            'avg_boundary_segment_length': np.mean(boundary_lengths) / self.base.scale if boundary_lengths else 0,
            'boundary_density': np.sum(boundaries) / (boundaries.shape[0] * boundaries.shape[1]),
            'junction_count': junction_count,
            'avg_boundary_orientation': np.mean(boundary_orientations) if boundary_orientations else 0,
            'boundary_orientation_std': np.std(boundary_orientations) if boundary_orientations else 0
        }
        
        return boundary_analysis
    
    def material_classification(self, material_type='steel'):
        """
        Classify microstructure based on material-specific criteria.
        
        Parameters:
        -----------
        material_type : str
            Type of material ('steel', 'aluminum', 'copper', 'titanium')
        """
        if self.base.grain_properties is None:
            raise ValueError("Grain analysis must be completed first")
        
        grains = self.base.grain_properties
        
        if material_type == 'steel':
            # Steel microstructure classification
            avg_grain_size = grains['equivalent_diameter_microns'].mean()
            
            # ASTM grain size number
            astm_grain_size = 1 + 3.322 * np.log10(avg_grain_size)
            
            # Classify based on grain size
            if avg_grain_size < 10:
                grain_class = "Fine"
            elif avg_grain_size < 50:
                grain_class = "Medium"
            else:
                grain_class = "Coarse"
            
            # Estimate microstructure type based on phases and grain characteristics
            if hasattr(self.base, 'phases') and self.base.phases is not None:
                phase_data = self.base.phases['data']
                if len(phase_data) >= 2:
                    phase_fraction_diff = abs(phase_data.iloc[0]['area_fraction'] - 
                                            phase_data.iloc[1]['area_fraction'])
                    
                    if phase_fraction_diff < 0.2:
                        microstructure_type = "Pearlite"
                    elif phase_data.iloc[0]['area_fraction'] > 0.8:
                        microstructure_type = "Ferrite"
                    else:
                        microstructure_type = "Mixed"
                else:
                    microstructure_type = "Single Phase"
            else:
                microstructure_type = "Unknown"
            
            classification = {
                'material': 'Steel',
                'avg_grain_size_microns': avg_grain_size,
                'astm_grain_size': astm_grain_size,
                'grain_size_class': grain_class,
                'estimated_microstructure': microstructure_type
            }
        
        else:
            # Generic classification for other materials
            avg_grain_size = grains['equivalent_diameter_microns'].mean()
            grain_size_std = grains['equivalent_diameter_microns'].std()
            avg_aspect_ratio = grains['aspect_ratio'].mean()
            
            # Uniformity index
            uniformity = 1 - (grain_size_std / avg_grain_size)
            
            classification = {
                'material': material_type.capitalize(),
                'avg_grain_size_microns': avg_grain_size,
                'grain_size_uniformity': uniformity,
                'avg_aspect_ratio': avg_aspect_ratio,
                'microstructure_quality': 'Good' if uniformity > 0.7 else 'Poor'
            }
        
        return classification
    
    def generate_advanced_report(self, save_path=None):
        """Generate comprehensive advanced analysis report."""
        report = []
        report.append("="*70)
        report.append("ADVANCED MICROSTRUCTURAL ANALYSIS REPORT")
        report.append("="*70)
        
        # Anomaly detection results
        if self.anomaly_scores is not None:
            report.append("\nANOMALY DETECTION")
            report.append("-" * 50)
            anomalous_count = np.sum(self.anomaly_scores['is_anomaly'])
            total_grains = len(self.anomaly_scores)
            report.append(f"Anomalous grains detected: {anomalous_count}/{total_grains} ({anomalous_count/total_grains*100:.1f}%)")
            
            if anomalous_count > 0:
                worst_anomalies = self.anomaly_scores.nsmallest(5, 'anomaly_score')
                report.append("Most anomalous grains:")
                for _, grain in worst_anomalies.iterrows():
                    report.append(f"  Grain {grain['grain_label']}: Score = {grain['anomaly_score']:.3f}")
        
        # Clustering results
        if self.grain_clusters is not None:
            report.append("\nGRAIN CLUSTERING")
            report.append("-" * 50)
            cluster_counts = self.grain_clusters['cluster_name'].value_counts()
            for cluster, count in cluster_counts.items():
                report.append(f"{cluster}: {count} grains")
        
        # Grain size distribution analysis
        try:
            dist_results = self.analyze_grain_size_distribution()
            if dist_results:
                report.append("\nGRAIN SIZE DISTRIBUTION")
                report.append("-" * 50)
                best_dist = min(dist_results.items(), key=lambda x: x[1]['aic'])
                report.append(f"Best fitting distribution: {best_dist[0]}")
                report.append(f"AIC: {best_dist[1]['aic']:.2f}")
                report.append(f"KS p-value: {best_dist[1]['ks_p_value']:.4f}")
        except Exception as e:
            report.append(f"\nGrain size distribution analysis failed: {e}")
        
        # Spatial statistics
        try:
            spatial_stats = self.calculate_spatial_statistics()
            report.append("\nSPATIAL STATISTICS")
            report.append("-" * 50)
            report.append(f"Clark-Evans R: {spatial_stats['clark_evans_R']:.3f}")
            report.append(f"Spatial pattern: {spatial_stats['spatial_pattern']}")
            report.append(f"Grain density: {spatial_stats['grain_density']:.6f} grains/pixel²")
        except Exception as e:
            report.append(f"\nSpatial analysis failed: {e}")
        
        # Boundary analysis
        try:
            boundary_stats = self.grain_boundary_analysis()
            report.append("\nGRAIN BOUNDARY ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total boundary length: {boundary_stats['total_boundary_length']:.1f} μm")
            report.append(f"Boundary density: {boundary_stats['boundary_density']:.4f}")
            report.append(f"Triple point junctions: {boundary_stats['junction_count']}")
        except Exception as e:
            report.append(f"\nBoundary analysis failed: {e}")
        
        # Material classification
        try:
            classification = self.material_classification()
            report.append("\nMATERIAL CLASSIFICATION")
            report.append("-" * 50)
            for key, value in classification.items():
                report.append(f"{key}: {value}")
        except Exception as e:
            report.append(f"\nMaterial classification failed: {e}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Advanced report saved to {save_path}")
        
        print(report_text)
        return report_text
    
    def visualize_advanced_results(self, figsize=(20, 16), save_path=None):
        """Create advanced visualization plots."""
        fig = plt.figure(figsize=figsize)
        
        # Original image
        plt.subplot(3, 4, 1)
        if self.base.original_image is not None:
            plt.imshow(cv2.cvtColor(self.base.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Anomaly detection
        if self.anomaly_scores is not None and self.base.grain_properties is not None:
            plt.subplot(3, 4, 2)
            
            # Create anomaly visualization
            img_anomaly = self.base.processed_image.copy()
            anomalous_grains = self.anomaly_scores[self.anomaly_scores['is_anomaly']]
            
            for _, anomaly in anomalous_grains.iterrows():
                grain_data = self.base.grain_properties[
                    self.base.grain_properties['label'] == anomaly['grain_label']
                ]
                if not grain_data.empty:
                    y, x = grain_data.iloc[0]['centroid_y'], grain_data.iloc[0]['centroid_x']
                    cv2.circle(img_anomaly, (int(x), int(y)), 5, 255, -1)
            
            plt.imshow(img_anomaly, cmap='gray')
            plt.title('Anomalous Grains (white dots)')
            plt.axis('off')
        
        # Grain clusters
        if self.grain_clusters is not None and self.base.grain_labels is not None:
            plt.subplot(3, 4, 3)
            
            # Color grains by cluster
            cluster_image = np.zeros_like(self.base.grain_labels, dtype=float)
            for _, row in self.grain_clusters.iterrows():
                mask = self.base.grain_labels == row['grain_label']
                cluster_image[mask] = row['cluster']
            
            plt.imshow(cluster_image, cmap='viridis')
            plt.title('Grain Clusters')
            plt.axis('off')
        
        # Grain size distribution with fitted curve
        if self.base.grain_properties is not None:
            plt.subplot(3, 4, 4)
            sizes = self.base.grain_properties['equivalent_diameter_microns']
            
            # Histogram
            plt.hist(sizes, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Fit and plot normal distribution
            try:
                mu, sigma = stats.norm.fit(sizes)
                x = np.linspace(sizes.min(), sizes.max(), 100)
                plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
                plt.legend()
            except:
                pass
            
            plt.xlabel('Grain Size (μm)')
            plt.ylabel('Density')
            plt.title('Grain Size Distribution')
            plt.grid(True, alpha=0.3)
        
        # Feature correlation heatmap
        if self.ml_features is not None:
            plt.subplot(3, 4, 5)
            
            # Select key features for correlation
            key_features = ['area_microns', 'aspect_ratio', 'circularity', 'solidity', 'eccentricity']
            available_features = [f for f in key_features if f in self.ml_features.columns]
            
            if len(available_features) > 1:
                corr_matrix = self.ml_features[available_features].corr()
                im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(im)
                plt.xticks(range(len(available_features)), available_features, rotation=45)
                plt.yticks(range(len(available_features)), available_features)
                plt.title('Feature Correlation')
        
        # PCA visualization
        if self.ml_features is not None and len(self.ml_features) > 2:
            plt.subplot(3, 4, 6)
            
            # Prepare features
            feature_cols = self.ml_features.select_dtypes(include=[np.number]).columns
            X = self.ml_features[feature_cols].fillna(0)
            
            if len(X.columns) > 1:
                # Standardize and apply PCA
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Color by cluster if available
                if self.grain_clusters is not None:
                    colors = self.grain_clusters['cluster']
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis', alpha=0.6)
                    plt.colorbar(scatter)
                else:
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
                
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                plt.title('PCA Visualization')
        
        # Spatial distribution
        if self.base.grain_properties is not None and 'centroid_x' in self.base.grain_properties.columns:
            plt.subplot(3, 4, 7)
            
            x = self.base.grain_properties['centroid_x']
            y = self.base.grain_properties['centroid_y']
            sizes = self.base.grain_properties['equivalent_diameter_microns']
            
            scatter = plt.scatter(x, y, c=sizes, s=sizes*2, alpha=0.6, cmap='plasma')
            plt.colorbar(scatter, label='Grain Size (μm)')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Spatial Grain Distribution')
            plt.axis('equal')
        
        # Defect analysis
        if self.base.defects is not None:
            plt.subplot(3, 4, 8)
            
            defect_types = ['pores', 'cracks', 'inclusions']
            defect_counts = [len(self.base.defects.get(dt, [])) for dt in defect_types]
            
            bars = plt.bar(defect_types, defect_counts, color=['red', 'blue', 'green'], alpha=0.7)
            plt.ylabel('Count')
            plt.title('Defect Summary')
            plt.xticks(rotation=45)
            
            # Add count labels on bars
            for bar, count in zip(bars, defect_counts):
                if count > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
        
        # Aspect ratio vs circularity
        if self.base.grain_properties is not None:
            plt.subplot(3, 4, 9)
            
            plt.scatter(self.base.grain_properties['aspect_ratio'],
                       self.base.grain_properties['circularity'],
                       alpha=0.6, s=30)
            plt.xlabel('Aspect Ratio')
            plt.ylabel('Circularity')
            plt.title('Grain Shape Analysis')
            plt.grid(True, alpha=0.3)
        
        # Texture features radar chart
        if 'texture_features' in self.base.results:
            plt.subplot(3, 4, 10, projection='polar')
            
            tf = self.base.results['texture_features']
            features = ['contrast', 'homogeneity', 'energy', 'correlation']
            available_tf = [f for f in features if f in tf]
            
            if available_tf:
                values = [tf[f] for f in available_tf]
                # Normalize values to 0-1 range
                values = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values))
                
                angles = np.linspace(0, 2*np.pi, len(available_tf), endpoint=False)
                values = np.concatenate((values, [values[0]]))  # Close the plot
                angles = np.concatenate((angles, [angles[0]]))
                
                plt.plot(angles, values, 'o-', linewidth=2)
                plt.fill(angles, values, alpha=0.25)
                plt.xticks(angles[:-1], available_tf)
                plt.title('Texture Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Advanced visualization saved to {save_path}")
        
        plt.show()
        return fig


# Example usage function
def run_advanced_analysis(base_analyzer):
    """Run complete advanced analysis pipeline."""
    print("Running Advanced Microstructural Analysis...")
    print("="*50)
    
    # Initialize advanced analyzer
    advanced = AdvancedMicrostructuralAnalyzer(base_analyzer)
    
    # Extract ML features
    print("Extracting machine learning features...")
    features = advanced.extract_ml_features()
    
    # Detect anomalies
    print("Detecting anomalous grains...")
    anomalies = advanced.detect_anomalous_grains()
    
    # Cluster grains
    print("Clustering grains...")
    clusters = advanced.cluster_grains()
    
    # Generate advanced report
    print("Generating advanced report...")
    advanced.generate_advanced_report(save_path="advanced_analysis_report.txt")
    
    # Create advanced visualizations
    print("Creating advanced visualizations...")
    advanced.visualize_advanced_results(save_path="advanced_analysis_plots.png")
    
    print("Advanced analysis complete!")
    return advanced
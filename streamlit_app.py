#!/usr/bin/env python3
"""
Streamlit Web Application for Microstructural Analysis
=====================================================

A user-friendly web interface for microstructural analysis using Streamlit.
Deploy locally or on cloud platforms like Streamlit Cloud, Heroku, or AWS.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from PIL import Image
import zipfile
import tempfile
import os

# Import our analysis modules
try:
    from microstructural_analyzer import MicrostructuralAnalyzer, create_sample_microstructure
    from advanced_features import AdvancedMicrostructuralAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    st.error(f"Analysis modules not available: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="Microstructural Analysis Toolkit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Microstructural Analysis Toolkit</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive computer vision analysis for materials science**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Home", "Upload & Analyze", "Sample Analysis", "Batch Processing", "Advanced Features", "Documentation"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Upload & Analyze":
        show_upload_analysis()
    elif page == "Sample Analysis":
        show_sample_analysis()
    elif page == "Batch Processing":
        show_batch_processing()
    elif page == "Advanced Features":
        show_advanced_features()
    elif page == "Documentation":
        show_documentation()

def show_home_page():
    """Display the home page with overview and features."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎯 What Can This Tool Do?</h2>', unsafe_allow_html=True)
        
        features = {
            "🔍 Grain Analysis": [
                "Automatic grain detection and segmentation",
                "Size distribution analysis (ASTM standards)",
                "Shape parameters (circularity, aspect ratio)",
                "Grain boundary characterization"
            ],
            "⚠️ Defect Detection": [
                "Pore identification and quantification",
                "Crack detection and measurement",
                "Inclusion analysis",
                "Statistical defect characterization"
            ],
            "🧪 Phase Analysis": [
                "Multi-phase identification",
                "Phase fraction calculations",
                "Phase distribution mapping",
                "Material classification"
            ],
            "🤖 Advanced Features": [
                "Machine learning anomaly detection",
                "Spatial statistics analysis",
                "Grain clustering and classification",
                "Distribution fitting and modeling"
            ]
        }
        
        for category, items in features.items():
            with st.expander(category, expanded=True):
                for item in items:
                    st.write(f"• {item}")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Sample Results</h2>', unsafe_allow_html=True)
        
        # Sample metrics
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Avg Grain Size", "28.5 μm", "±8.2")
            st.metric("Total Grains", "247", "+15")
        with col2_2:
            st.metric("Defect Density", "0.23%", "-0.05")
            st.metric("Quality Score", "Excellent", "↑")
        
        # Sample applications
        st.markdown('<h3 class="sub-header">🏭 Applications</h3>', unsafe_allow_html=True)
        applications = [
            "Steel microstructure analysis",
            "Aluminum alloy characterization", 
            "Ceramic quality control",
            "Composite material analysis",
            "Research and development",
            "Quality assurance"
        ]
        for app in applications:
            st.write(f"• {app}")
    
    # Quick start section
    st.markdown('<h2 class="sub-header">🚀 Quick Start</h2>', unsafe_allow_html=True)
    
    start_cols = st.columns(3)
    with start_cols[0]:
        st.markdown("""
        <div class="info-box">
        <h4>1. Upload Image</h4>
        <p>Upload your microstructural image (PNG, JPG, TIFF supported)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with start_cols[1]:
        st.markdown("""
        <div class="info-box">
        <h4>2. Set Parameters</h4>
        <p>Configure analysis parameters and scale calibration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with start_cols[2]:
        st.markdown("""
        <div class="info-box">
        <h4>3. Analyze & Download</h4>
        <p>Get comprehensive results and visualizations</p>
        </div>
        """, unsafe_allow_html=True)

def show_upload_analysis():
    """Handle image upload and analysis."""
    
    st.markdown('<h2 class="sub-header">📤 Upload & Analyze Your Image</h2>', unsafe_allow_html=True)
    
    if not ANALYSIS_AVAILABLE:
        st.error("Analysis modules not available. Please install dependencies.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a microstructural image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload high-quality microstructural images for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Format:** {image.format}")
        
        with col2:
            st.subheader("⚙️ Analysis Parameters")
            
            # Scale calibration
            scale = st.number_input(
                "Scale (pixels per micron)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Calibration factor for converting pixels to microns"
            )
            
            # Analysis options
            st.subheader("Analysis Options")
            run_grain_analysis = st.checkbox("Grain Analysis", value=True)
            run_defect_detection = st.checkbox("Defect Detection", value=True)
            run_phase_analysis = st.checkbox("Phase Analysis", value=True)
            run_texture_analysis = st.checkbox("Texture Analysis", value=False)
            
            # Advanced parameters
            with st.expander("Advanced Parameters"):
                enhance_contrast = st.checkbox("Enhance Contrast", value=True)
                denoise = st.checkbox("Apply Denoising", value=True)
                gaussian_sigma = st.slider("Gaussian Sigma", 0.5, 3.0, 1.0, 0.1)
                min_grain_distance = st.slider("Min Grain Distance", 10, 50, 20, 5)
        
        # Analysis button
        if st.button("🔬 Start Analysis", type="primary"):
            with st.spinner("Analyzing microstructure... This may take a few minutes."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        image.save(tmp_file.name)
                        
                        # Initialize analyzer
                        analyzer = MicrostructuralAnalyzer(tmp_file.name, scale_pixels_per_micron=scale)
                        
                        # Set parameters
                        preprocess_params = {
                            'enhance_contrast': enhance_contrast,
                            'denoise': denoise,
                            'gaussian_sigma': gaussian_sigma
                        }
                        
                        grain_params = {
                            'min_distance': min_grain_distance
                        }
                        
                        # Run analysis
                        results = analyzer.run_complete_analysis(
                            preprocess_params=preprocess_params,
                            grain_params=grain_params
                        )
                        
                        # Display results
                        display_analysis_results(analyzer, results)
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Try adjusting the analysis parameters or check image quality.")

def show_sample_analysis():
    """Show analysis with sample data."""
    
    st.markdown('<h2 class="sub-header">🧪 Sample Analysis Demonstration</h2>', unsafe_allow_html=True)
    
    if not ANALYSIS_AVAILABLE:
        st.error("Analysis modules not available. Please install dependencies.")
        return
    
    st.info("This demonstrates the analysis using a computer-generated sample microstructure.")
    
    # Sample parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Parameters")
        sample_size = st.selectbox("Image Size", [400, 600, 800], index=1)
        n_grains = st.slider("Number of Grains", 20, 100, 50, 5)
        noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)
    
    with col2:
        st.subheader("Analysis Parameters")
        scale = st.number_input("Scale (pixels/micron)", 1.0, 5.0, 2.5, 0.1)
        run_advanced = st.checkbox("Run Advanced Analysis", value=False)
    
    if st.button("🔬 Generate & Analyze Sample", type="primary"):
        with st.spinner("Generating sample and analyzing..."):
            try:
                # Create sample
                sample_path = "temp_sample.png"
                create_sample_microstructure(
                    size=(sample_size, sample_size),
                    n_grains=n_grains,
                    noise_level=noise_level,
                    save_path=sample_path
                )
                
                # Analyze
                analyzer = MicrostructuralAnalyzer(sample_path, scale_pixels_per_micron=scale)
                results = analyzer.run_complete_analysis()
                
                # Display results
                display_analysis_results(analyzer, results)
                
                # Advanced analysis
                if run_advanced:
                    st.subheader("🤖 Advanced Analysis")
                    with st.spinner("Running advanced analysis..."):
                        from advanced_features import AdvancedMicrostructuralAnalyzer
                        advanced = AdvancedMicrostructuralAnalyzer(analyzer)
                        
                        # Extract features and run ML analysis
                        features = advanced.extract_ml_features()
                        anomalies = advanced.detect_anomalous_grains()
                        clusters = advanced.cluster_grains()
                        
                        display_advanced_results(advanced, features, anomalies, clusters)
                
                # Clean up
                if os.path.exists(sample_path):
                    os.unlink(sample_path)
                    
            except Exception as e:
                st.error(f"Sample analysis failed: {str(e)}")

def display_analysis_results(analyzer, results):
    """Display comprehensive analysis results."""
    
    st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Key metrics
    if 'grain_analysis' in results:
        ga = results['grain_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Grains", ga.get('total_grains', 0))
        with col2:
            st.metric("Avg Grain Size", f"{ga.get('avg_grain_size_microns', 0):.1f} μm")
        with col3:
            st.metric("Std Deviation", f"{ga.get('std_grain_size_microns', 0):.1f} μm")
        with col4:
            st.metric("Grain Area Fraction", f"{ga.get('total_grain_area_fraction', 0):.3f}")
    
    # Defect metrics
    if 'defect_analysis' in results:
        da = results['defect_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pores", da.get('total_pores', 0))
        with col2:
            st.metric("Cracks", da.get('total_cracks', 0))
        with col3:
            st.metric("Inclusions", da.get('total_inclusions', 0))
        with col4:
            st.metric("Pore Area %", f"{da.get('pore_area_fraction', 0)*100:.3f}%")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Images", "📊 Distributions", "📈 Statistics", "📋 Report"])
    
    with tab1:
        display_image_results(analyzer)
    
    with tab2:
        display_distribution_plots(analyzer)
    
    with tab3:
        display_statistical_analysis(analyzer, results)
    
    with tab4:
        display_text_report(analyzer, results)

def display_image_results(analyzer):
    """Display image analysis results."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if analyzer.original_image is not None:
            st.subheader("Original Image")
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(analyzer.original_image, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_column_width=True)
    
    with col2:
        if analyzer.processed_image is not None:
            st.subheader("Processed Image")
            st.image(analyzer.processed_image, use_column_width=True, cmap='gray')
    
    if hasattr(analyzer, 'grain_boundaries') and analyzer.grain_labels is not None:
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Grain Boundaries")
            st.image(analyzer.grain_boundaries, use_column_width=True)
        
        with col4:
            st.subheader("Grain Segmentation")
            # Create colored segmentation
            segmented = np.zeros_like(analyzer.grain_labels, dtype=np.uint8)
            unique_labels = np.unique(analyzer.grain_labels)
            for i, label in enumerate(unique_labels[1:]):  # Skip background
                segmented[analyzer.grain_labels == label] = (i * 255 // len(unique_labels)) % 255
            st.image(segmented, use_column_width=True)

def display_distribution_plots(analyzer):
    """Display distribution plots using Plotly."""
    
    if analyzer.grain_properties is not None and len(analyzer.grain_properties) > 0:
        grain_data = analyzer.grain_properties
        
        # Grain size distribution
        fig1 = px.histogram(
            grain_data, 
            x='equivalent_diameter_microns',
            nbins=20,
            title='Grain Size Distribution',
            labels={'equivalent_diameter_microns': 'Grain Size (μm)', 'count': 'Frequency'}
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Aspect ratio vs circularity
        fig2 = px.scatter(
            grain_data,
            x='aspect_ratio',
            y='circularity',
            size='area_microns',
            title='Grain Shape Analysis',
            labels={'aspect_ratio': 'Aspect Ratio', 'circularity': 'Circularity'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Multiple distributions
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Area Distribution', 'Aspect Ratio', 'Circularity', 'Solidity']
        )
        
        # Area
        fig3.add_trace(
            go.Histogram(x=grain_data['area_microns'], name='Area'),
            row=1, col=1
        )
        
        # Aspect Ratio
        fig3.add_trace(
            go.Histogram(x=grain_data['aspect_ratio'], name='Aspect Ratio'),
            row=1, col=2
        )
        
        # Circularity
        fig3.add_trace(
            go.Histogram(x=grain_data['circularity'], name='Circularity'),
            row=2, col=1
        )
        
        # Solidity
        fig3.add_trace(
            go.Histogram(x=grain_data['solidity'], name='Solidity'),
            row=2, col=2
        )
        
        fig3.update_layout(height=600, showlegend=False, title_text="Grain Property Distributions")
        st.plotly_chart(fig3, use_container_width=True)

def display_statistical_analysis(analyzer, results):
    """Display statistical analysis."""
    
    if analyzer.grain_properties is not None and len(analyzer.grain_properties) > 0:
        grain_data = analyzer.grain_properties
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        numeric_cols = grain_data.select_dtypes(include=[np.number]).columns
        summary_stats = grain_data[numeric_cols].describe()
        
        # Format for display
        formatted_stats = summary_stats.round(3)
        st.dataframe(formatted_stats, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Feature Correlations")
        key_features = ['area_microns', 'aspect_ratio', 'circularity', 'solidity', 'eccentricity']
        available_features = [f for f in key_features if f in grain_data.columns]
        
        if len(available_features) > 1:
            corr_matrix = grain_data[available_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    # Phase analysis results
    if analyzer.phases is not None:
        st.subheader("🧪 Phase Analysis")
        phase_data = analyzer.phases['data']
        
        if len(phase_data) > 0:
            # Phase fractions pie chart
            fig = px.pie(
                phase_data,
                values='area_fraction',
                names=[f'Phase {i}' for i in phase_data['phase_id']],
                title='Phase Fractions'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Phase data table
            st.dataframe(phase_data, use_container_width=True)

def display_text_report(analyzer, results):
    """Display comprehensive text report."""
    
    # Generate report
    report = analyzer.generate_report()
    
    # Display in text area
    st.text_area("📋 Analysis Report", report, height=400)
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="microstructure_analysis_report.txt",
        mime="text/plain"
    )
    
    # Download data as CSV
    if analyzer.grain_properties is not None:
        csv_data = analyzer.grain_properties.to_csv(index=False)
        st.download_button(
            label="📥 Download Grain Data (CSV)",
            data=csv_data,
            file_name="grain_analysis_data.csv",
            mime="text/csv"
        )

def display_advanced_results(advanced, features, anomalies, clusters):
    """Display advanced analysis results."""
    
    st.subheader("🤖 Machine Learning Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Anomaly Detection")
        anomaly_count = np.sum(anomalies['is_anomaly'])
        total_grains = len(anomalies)
        st.metric("Anomalous Grains", f"{anomaly_count}/{total_grains}")
        st.metric("Anomaly Rate", f"{anomaly_count/total_grains*100:.1f}%")
        
        if anomaly_count > 0:
            worst_anomalies = anomalies.nsmallest(5, 'anomaly_score')
            st.write("Most Anomalous Grains:")
            for _, grain in worst_anomalies.iterrows():
                st.write(f"Grain {grain['grain_label']}: Score = {grain['anomaly_score']:.3f}")
    
    with col2:
        st.subheader("🎯 Grain Clustering")
        cluster_counts = clusters['cluster_name'].value_counts()
        
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Grain Clusters"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_batch_processing():
    """Handle batch processing of multiple images."""
    
    st.markdown('<h2 class="sub-header">📦 Batch Processing</h2>', unsafe_allow_html=True)
    
    st.info("Upload multiple images for batch analysis. Results will be compiled into a summary report.")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Choose microstructural images",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")
        
        # Batch parameters
        col1, col2 = st.columns(2)
        
        with col1:
            scale = st.number_input("Scale (pixels/micron)", 1.0, 5.0, 2.0, 0.1)
            max_files = st.number_input("Max files to process", 1, len(uploaded_files), min(len(uploaded_files), 10))
        
        with col2:
            analysis_options = st.multiselect(
                "Analysis Options",
                ["Grain Analysis", "Defect Detection", "Phase Analysis"],
                default=["Grain Analysis", "Defect Detection"]
            )
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files[:max_files]):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / max_files)
                
                try:
                    # Process each file
                    image = Image.open(uploaded_file)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        image.save(tmp_file.name)
                        
                        analyzer = MicrostructuralAnalyzer(tmp_file.name, scale_pixels_per_micron=scale)
                        results = analyzer.run_complete_analysis()
                        
                        # Extract key metrics
                        file_results = {
                            'filename': uploaded_file.name,
                            'total_grains': results.get('grain_analysis', {}).get('total_grains', 0),
                            'avg_grain_size': results.get('grain_analysis', {}).get('avg_grain_size_microns', 0),
                            'total_pores': results.get('defect_analysis', {}).get('total_pores', 0),
                            'total_cracks': results.get('defect_analysis', {}).get('total_cracks', 0),
                            'pore_area_fraction': results.get('defect_analysis', {}).get('pore_area_fraction', 0)
                        }
                        batch_results.append(file_results)
                        
                        os.unlink(tmp_file.name)
                
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    continue
            
            # Display batch results
            if batch_results:
                st.success(f"✅ Processed {len(batch_results)} files successfully!")
                
                df = pd.DataFrame(batch_results)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Batch Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Grains per Image", f"{df['total_grains'].mean():.1f}")
                    st.metric("Total Grains", f"{df['total_grains'].sum()}")
                
                with col2:
                    st.metric("Avg Grain Size", f"{df['avg_grain_size'].mean():.1f} μm")
                    st.metric("Size Std Dev", f"{df['avg_grain_size'].std():.1f} μm")
                
                with col3:
                    st.metric("Avg Defects", f"{(df['total_pores'] + df['total_cracks']).mean():.1f}")
                    st.metric("Max Defects", f"{(df['total_pores'] + df['total_cracks']).max()}")
                
                # Download batch results
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Batch Results",
                    data=csv_data,
                    file_name="batch_analysis_results.csv",
                    mime="text/csv"
                )

def show_advanced_features():
    """Show advanced features and ML capabilities."""
    
    st.markdown('<h2 class="sub-header">🤖 Advanced Features</h2>', unsafe_allow_html=True)
    
    st.info("Advanced features require a completed basic analysis first.")
    
    # Feature descriptions
    features = {
        "🔍 Anomaly Detection": {
            "description": "Identify unusual grains using machine learning",
            "methods": ["Isolation Forest", "Local Outlier Factor"],
            "outputs": ["Anomaly scores", "Anomalous grain identification"]
        },
        "🎯 Grain Clustering": {
            "description": "Group grains with similar characteristics",
            "methods": ["K-means", "DBSCAN", "Hierarchical"],
            "outputs": ["Cluster assignments", "Cluster visualization"]
        },
        "📊 Distribution Analysis": {
            "description": "Fit statistical distributions to grain sizes",
            "methods": ["Normal", "Log-normal", "Gamma", "Weibull"],
            "outputs": ["Best-fit parameters", "Goodness-of-fit tests"]
        },
        "🗺️ Spatial Statistics": {
            "description": "Analyze spatial distribution patterns",
            "methods": ["Clark-Evans test", "Ripley's K function"],
            "outputs": ["Spatial randomness", "Clustering indices"]
        },
        "🧪 Material Classification": {
            "description": "Classify microstructure type",
            "methods": ["Steel classification", "ASTM standards"],
            "outputs": ["Microstructure type", "Quality assessment"]
        }
    }
    
    for feature_name, feature_info in features.items():
        with st.expander(feature_name, expanded=False):
            st.write(f"**Description:** {feature_info['description']}")
            st.write(f"**Methods:** {', '.join(feature_info['methods'])}")
            st.write(f"**Outputs:** {', '.join(feature_info['outputs'])}")
    
    st.markdown('<h3 class="sub-header">🔬 Try Advanced Analysis</h3>', unsafe_allow_html=True)
    st.write("Upload an image in the 'Upload & Analyze' section, then return here to run advanced analysis on the results.")

def show_documentation():
    """Show documentation and help."""
    
    st.markdown('<h2 class="sub-header">📚 Documentation</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "🔧 Installation", "💡 Usage Tips", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ## Overview
        
        This toolkit provides comprehensive computer vision analysis for microstructural images from materials science. 
        It can extract detailed quantitative features including:
        
        - **Grain Characteristics**: Size, shape, distribution, boundaries
        - **Defect Analysis**: Pores, cracks, inclusions with quantification
        - **Phase Analysis**: Multi-phase identification and fractions
        - **Advanced Features**: ML-based anomaly detection and clustering
        
        ### Key Algorithms
        - Edge detection: Sobel, Canny, Scharr filters
        - Segmentation: Watershed, Otsu thresholding, K-means
        - Machine Learning: Isolation Forest, DBSCAN, clustering
        - Statistical Analysis: Distribution fitting, spatial statistics
        """)
    
    with tab2:
        st.markdown("""
        ## Installation
        
        ### Local Installation
        ```bash
        # Clone the repository
        git clone <repository-url>
        cd microstructural-analysis
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Run Streamlit app
        streamlit run streamlit_app.py
        ```
        
        ### Docker Installation
        ```bash
        # Build Docker image
        docker build -t microstructure-analyzer .
        
        # Run container
        docker run -p 8501:8501 microstructure-analyzer
        ```
        
        ### Cloud Deployment
        - **Streamlit Cloud**: Connect GitHub repository
        - **Heroku**: Use provided Procfile
        - **AWS/GCP**: Use Docker container
        """)
    
    with tab3:
        st.markdown("""
        ## Usage Tips
        
        ### Image Quality
        - Use high-contrast, well-focused images
        - Ensure proper illumination without shadows
        - Optimal resolution: 800x600 to 2000x2000 pixels
        - Supported formats: PNG, JPG, TIFF, BMP
        
        ### Scale Calibration
        - Accurately measure pixel-to-micron conversion
        - Use scale bars or calibration standards
        - Typical values: 1-5 pixels per micron
        
        ### Parameter Tuning
        - Start with default parameters
        - Adjust based on material type:
          - Steel: Higher contrast enhancement
          - Aluminum: Lower noise reduction
          - Ceramics: Adjust pore detection threshold
        
        ### Best Results
        - Preprocess images for consistent lighting
        - Use ROI selection for large images
        - Validate results with manual measurements
        """)
    
    with tab4:
        st.markdown("""
        ## Frequently Asked Questions
        
        **Q: What image formats are supported?**
        A: PNG, JPG, JPEG, TIFF, and BMP formats are supported.
        
        **Q: How accurate are the measurements?**
        A: Accuracy depends on image quality and scale calibration. Typically ±5-10% for well-prepared samples.
        
        **Q: Can I process multiple images at once?**
        A: Yes, use the Batch Processing feature for multiple images.
        
        **Q: What if grain detection is poor?**
        A: Try adjusting preprocessing parameters, particularly contrast enhancement and denoising.
        
        **Q: How do I calibrate the scale?**
        A: Measure a known distance in pixels and divide by the actual distance in microns.
        
        **Q: Can I export the results?**
        A: Yes, all results can be downloaded as text reports and CSV data files.
        
        **Q: Is this suitable for all materials?**
        A: The toolkit works best with metallic and ceramic microstructures. Some parameters may need adjustment for specific materials.
        """)

if __name__ == "__main__":
    main()
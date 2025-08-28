#!/usr/bin/env python3
"""
Desktop GUI Application for Microstructural Analysis
===================================================

A simple desktop application using tkinter for microstructural analysis.
Provides a user-friendly interface for loading images and running analysis.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import webbrowser
import os
import sys
from pathlib import Path
import json
from PIL import Image, ImageTk

# Import analysis modules
try:
    from microstructural_analyzer import MicrostructuralAnalyzer, create_sample_microstructure
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

class MicrostructureAnalyzerGUI:
    """Main GUI application class."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Microstructural Analysis Toolkit v1.0")
        self.root.geometry("800x600")
        
        # Variables
        self.current_image_path = None
        self.analysis_results = None
        self.scale_var = tk.DoubleVar(value=2.0)
        
        # Setup UI
        self.setup_ui()
        self.update_status("Ready")
    
    def setup_ui(self):
        """Setup the user interface."""
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="🔬 Microstructural Analysis Toolkit", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Computer Vision Analysis for Materials Science", 
                                 font=("Arial", 10))
        subtitle_label.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.setup_analysis_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()
        
        # Tools tab
        self.tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tools_frame, text="Tools")
        self.setup_tools_tab()
        
        # About tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")
        self.setup_about_tab()
        
        # Status bar
        self.status_frame = ttk.Frame(main_container)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(2, 0))
    
    def setup_analysis_tab(self):
        """Setup the main analysis tab."""
        
        # Image selection frame
        image_frame = ttk.LabelFrame(self.analysis_frame, text="Image Selection", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Image path
        path_frame = ttk.Frame(image_frame)
        path_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(path_frame, text="Image:").pack(side=tk.LEFT)
        self.image_path_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.image_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(path_frame, text="Browse", command=self.browse_image).pack(side=tk.RIGHT)
        
        # Sample button
        ttk.Button(image_frame, text="Create Sample Image", command=self.create_sample).pack(pady=(5, 0))
        
        # Image preview
        self.image_label = ttk.Label(image_frame, text="No image selected", background="white", relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scale parameter
        scale_frame = ttk.Frame(params_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(scale_frame, text="Scale (pixels/micron):").pack(side=tk.LEFT)
        scale_entry = ttk.Entry(scale_frame, textvariable=self.scale_var, width=10)
        scale_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Analysis options
        options_frame = ttk.Frame(params_frame)
        options_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.denoise_var = tk.BooleanVar(value=True)
        self.enhance_contrast_var = tk.BooleanVar(value=True)
        self.detect_grains_var = tk.BooleanVar(value=True)
        self.detect_defects_var = tk.BooleanVar(value=True)
        self.analyze_phases_var = tk.BooleanVar(value=True)
        self.run_advanced_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(options_frame, text="Apply Denoising", variable=self.denoise_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Enhance Contrast", variable=self.enhance_contrast_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Detect Grains", variable=self.detect_grains_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Detect Defects", variable=self.detect_defects_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Analyze Phases", variable=self.analyze_phases_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Advanced ML Analysis", variable=self.run_advanced_var).pack(anchor=tk.W)
        
        # Control buttons
        control_frame = ttk.Frame(self.analysis_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT)
    
    def setup_results_tab(self):
        """Setup the results display tab."""
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results buttons
        results_buttons = ttk.Frame(self.results_frame)
        results_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(results_buttons, text="Save Report", command=self.save_report).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(results_buttons, text="View Plots", command=self.view_plots).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(results_buttons, text="Export Data", command=self.export_data).pack(side=tk.LEFT)
    
    def setup_tools_tab(self):
        """Setup the tools tab."""
        
        tools_label = ttk.Label(self.tools_frame, text="Additional Tools", font=("Arial", 14, "bold"))
        tools_label.pack(pady=10)
        
        # Tool buttons
        tool_buttons = [
            ("Open Web Interface", self.open_web_interface, "Launch Streamlit web application"),
            ("Open Jupyter Notebook", self.open_jupyter, "Open interactive Jupyter tutorial"),
            ("Start API Server", self.start_api_server, "Start REST API server"),
            ("View Documentation", self.view_documentation, "Open documentation in browser"),
            ("Sample Gallery", self.show_sample_gallery, "View sample microstructures")
        ]
        
        for text, command, description in tool_buttons:
            button_frame = ttk.Frame(self.tools_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=5)
            
            ttk.Button(button_frame, text=text, command=command, width=20).pack(side=tk.LEFT)
            ttk.Label(button_frame, text=description).pack(side=tk.LEFT, padx=(10, 0))
    
    def setup_about_tab(self):
        """Setup the about tab."""
        
        about_text = """
Microstructural Analysis Toolkit
=================================

Version: 1.0.0
Author: AI Assistant
License: Educational Use

Features:
• Comprehensive grain analysis
• Defect detection and characterization
• Phase analysis and identification
• Advanced machine learning features
• Statistical analysis and reporting
• Multiple deployment options

Supported Materials:
• Steel and iron alloys
• Aluminum alloys
• Ceramics and composites
• Custom materials

Applications:
• Quality control and inspection
• Materials research and development
• Process optimization
• Academic research and education

Dependencies:
• OpenCV for image processing
• scikit-image for advanced analysis
• scikit-learn for machine learning
• matplotlib for visualization
• pandas for data management

For more information, documentation, and updates,
please refer to the project repository and README file.
        """
        
        about_label = ttk.Label(self.about_frame, text=about_text, justify=tk.LEFT, font=("Courier", 9))
        about_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def browse_image(self):
        """Open file dialog to select an image."""
        
        file_path = filedialog.askopenfilename(
            title="Select Microstructural Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("TIFF files", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_path_var.set(file_path)
            self.load_image_preview()
    
    def load_image_preview(self):
        """Load and display image preview."""
        
        if not self.current_image_path:
            return
        
        try:
            # Load image
            image = Image.open(self.current_image_path)
            
            # Resize for preview (maintain aspect ratio)
            preview_size = (200, 150)
            image.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            self.update_status(f"Image loaded: {os.path.basename(self.current_image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image preview: {str(e)}")
    
    def create_sample(self):
        """Create a sample microstructure image."""
        
        if not ANALYSIS_AVAILABLE:
            messagebox.showerror("Error", "Analysis modules not available. Please install dependencies.")
            return
        
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                title="Save Sample Image",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if file_path:
                self.update_status("Creating sample image...")
                self.progress.start()
                
                # Create sample in a thread
                def create_thread():
                    try:
                        create_sample_microstructure(
                            size=(600, 600),
                            n_grains=50,
                            noise_level=0.05,
                            save_path=file_path
                        )
                        
                        # Update UI in main thread
                        self.root.after(0, lambda: self.sample_created(file_path))
                        
                    except Exception as e:
                        self.root.after(0, lambda: self.sample_creation_failed(str(e)))
                
                threading.Thread(target=create_thread, daemon=True).start()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample: {str(e)}")
            self.progress.stop()
    
    def sample_created(self, file_path):
        """Handle successful sample creation."""
        self.progress.stop()
        self.current_image_path = file_path
        self.image_path_var.set(file_path)
        self.load_image_preview()
        self.update_status("Sample image created successfully")
        messagebox.showinfo("Success", f"Sample image created: {os.path.basename(file_path)}")
    
    def sample_creation_failed(self, error):
        """Handle failed sample creation."""
        self.progress.stop()
        self.update_status("Failed to create sample")
        messagebox.showerror("Error", f"Failed to create sample: {error}")
    
    def start_analysis(self):
        """Start the microstructural analysis."""
        
        if not ANALYSIS_AVAILABLE:
            messagebox.showerror("Error", "Analysis modules not available. Please install dependencies.")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        if not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "Selected image file does not exist.")
            return
        
        # Confirm analysis
        if not messagebox.askyesno("Confirm Analysis", "Start microstructural analysis? This may take several minutes."):
            return
        
        # Start analysis in background thread
        self.update_status("Starting analysis...")
        self.progress.start()
        
        def analysis_thread():
            try:
                # Initialize analyzer
                analyzer = MicrostructuralAnalyzer(
                    self.current_image_path, 
                    scale_pixels_per_micron=self.scale_var.get()
                )
                
                # Set parameters
                preprocess_params = {
                    'denoise': self.denoise_var.get(),
                    'enhance_contrast': self.enhance_contrast_var.get()
                }
                
                # Run analysis
                results = analyzer.run_complete_analysis(preprocess_params=preprocess_params)
                
                # Generate report
                report = analyzer.generate_report()
                
                # Store results
                self.analysis_results = {
                    'analyzer': analyzer,
                    'results': results,
                    'report': report
                }
                
                # Update UI in main thread
                self.root.after(0, self.analysis_completed)
                
            except Exception as e:
                self.root.after(0, lambda: self.analysis_failed(str(e)))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def analysis_completed(self):
        """Handle successful analysis completion."""
        self.progress.stop()
        self.update_status("Analysis completed successfully")
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, self.analysis_results['report'])
        
        # Switch to results tab
        self.notebook.select(self.results_frame)
        
        messagebox.showinfo("Success", "Analysis completed successfully! Check the Results tab for details.")
    
    def analysis_failed(self, error):
        """Handle failed analysis."""
        self.progress.stop()
        self.update_status("Analysis failed")
        messagebox.showerror("Analysis Error", f"Analysis failed: {error}")
    
    def clear_results(self):
        """Clear analysis results."""
        self.analysis_results = None
        self.results_text.delete(1.0, tk.END)
        self.update_status("Results cleared")
    
    def export_results(self):
        """Export analysis results."""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No results to export. Run analysis first.")
            return
        
        # Ask for export directory
        directory = filedialog.askdirectory(title="Select Export Directory")
        if not directory:
            return
        
        try:
            analyzer = self.analysis_results['analyzer']
            
            # Export report
            report_path = os.path.join(directory, "analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write(self.analysis_results['report'])
            
            # Export grain data if available
            if analyzer.grain_properties is not None:
                csv_path = os.path.join(directory, "grain_data.csv")
                analyzer.grain_properties.to_csv(csv_path, index=False)
            
            # Export visualization
            plot_path = os.path.join(directory, "analysis_plots.png")
            analyzer.visualize_results(save_path=plot_path)
            
            messagebox.showinfo("Success", f"Results exported to: {directory}")
            self.update_status(f"Results exported to {directory}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def save_report(self):
        """Save the analysis report."""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No report to save. Run analysis first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.analysis_results['report'])
                messagebox.showinfo("Success", f"Report saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def view_plots(self):
        """View analysis plots."""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No plots to view. Run analysis first.")
            return
        
        try:
            analyzer = self.analysis_results['analyzer']
            analyzer.visualize_results()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display plots: {str(e)}")
    
    def export_data(self):
        """Export analysis data as CSV."""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No data to export. Run analysis first.")
            return
        
        analyzer = self.analysis_results['analyzer']
        if analyzer.grain_properties is None:
            messagebox.showwarning("Warning", "No grain data available.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Grain Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                analyzer.grain_properties.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def open_web_interface(self):
        """Open the Streamlit web interface."""
        def start_streamlit():
            try:
                subprocess.run(['streamlit', 'run', 'streamlit_app.py'], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.root.after(0, lambda: messagebox.showerror("Error", 
                    "Failed to start Streamlit. Make sure it's installed and streamlit_app.py exists."))
        
        # Start Streamlit in background
        threading.Thread(target=start_streamlit, daemon=True).start()
        
        # Open browser after a delay
        self.root.after(3000, lambda: webbrowser.open('http://localhost:8501'))
        
        messagebox.showinfo("Web Interface", "Starting web interface... It will open in your browser shortly.")
    
    def open_jupyter(self):
        """Open Jupyter notebook."""
        try:
            subprocess.Popen(['jupyter', 'notebook', 'Microstructural_Analysis_Tutorial.ipynb'])
            messagebox.showinfo("Jupyter", "Opening Jupyter notebook...")
        except FileNotFoundError:
            messagebox.showerror("Error", "Jupyter not found. Make sure it's installed.")
    
    def start_api_server(self):
        """Start the API server."""
        def start_api():
            try:
                subprocess.run(['python', 'api_server.py'], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.root.after(0, lambda: messagebox.showerror("Error", 
                    "Failed to start API server. Make sure api_server.py exists."))
        
        threading.Thread(target=start_api, daemon=True).start()
        messagebox.showinfo("API Server", "Starting API server... It will be available at http://localhost:8000")
    
    def view_documentation(self):
        """Open documentation."""
        try:
            # Look for README.md
            if os.path.exists("README.md"):
                webbrowser.open(f"file://{os.path.abspath('README.md')}")
            else:
                messagebox.showinfo("Documentation", "README.md not found in current directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open documentation: {str(e)}")
    
    def show_sample_gallery(self):
        """Show sample microstructures gallery."""
        messagebox.showinfo("Sample Gallery", 
            "Feature coming soon! Use 'Create Sample Image' to generate synthetic microstructures.")
    
    def update_status(self, message):
        """Update the status bar."""
        self.status_label.configure(text=message)
        self.root.update_idletasks()

def main():
    """Main function to run the GUI application."""
    
    # Create and configure root window
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme
    
    # Create application
    app = MicrostructureAnalyzerGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        root.destroy()

if __name__ == "__main__":
    main()
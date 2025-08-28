#!/usr/bin/env python3
"""
REST API Server for Microstructural Analysis
===========================================

FastAPI-based REST API for programmatic access to microstructural analysis.
Supports file uploads, batch processing, and JSON responses.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
import tempfile
import os
import json
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import shutil

# Import analysis modules
try:
    from microstructural_analyzer import MicrostructuralAnalyzer, create_sample_microstructure
    from advanced_features import AdvancedMicrostructuralAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    print(f"Analysis modules not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Microstructural Analysis API",
    description="REST API for computer vision analysis of microstructural images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)

# In-memory storage for job status (use Redis/database in production)
job_status = {}

class AnalysisResult:
    """Class to store analysis results."""
    def __init__(self, job_id: str, status: str, results: Dict = None, error: str = None):
        self.job_id = job_id
        self.status = status
        self.results = results
        self.error = error
        self.timestamp = datetime.now().isoformat()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Microstructural Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "analysis_available": ANALYSIS_AVAILABLE,
        "endpoints": {
            "analyze": "/analyze",
            "batch_analyze": "/batch-analyze",
            "sample_analysis": "/sample-analysis",
            "job_status": "/status/{job_id}",
            "download": "/download/{job_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "analysis_available": ANALYSIS_AVAILABLE
    }

@app.post("/analyze")
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: float = Form(2.0),
    denoise: bool = Form(True),
    enhance_contrast: bool = Form(True),
    detect_grains: bool = Form(True),
    detect_defects: bool = Form(True),
    analyze_phases: bool = Form(True),
    run_advanced: bool = Form(False)
):
    """
    Analyze a single microstructural image.
    
    Parameters:
    -----------
    file : UploadFile
        Image file to analyze
    scale : float
        Scale factor (pixels per micron)
    denoise : bool
        Apply denoising
    enhance_contrast : bool
        Apply contrast enhancement
    detect_grains : bool
        Run grain analysis
    detect_defects : bool
        Run defect detection
    analyze_phases : bool
        Run phase analysis
    run_advanced : bool
        Run advanced ML analysis
    
    Returns:
    --------
    dict : Job information with job_id for status tracking
    """
    
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analysis modules not available")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Initialize job status
    job_status[job_id] = AnalysisResult(job_id, "queued")
    
    # Add background task
    background_tasks.add_task(
        run_analysis_task,
        job_id=job_id,
        file=file,
        scale=scale,
        analysis_params={
            'denoise': denoise,
            'enhance_contrast': enhance_contrast,
            'detect_grains': detect_grains,
            'detect_defects': detect_defects,
            'analyze_phases': analyze_phases,
            'run_advanced': run_advanced
        }
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Analysis started. Use /status/{job_id} to check progress.",
        "estimated_time": "2-5 minutes"
    }

@app.post("/batch-analyze")
async def batch_analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    scale: float = Form(2.0),
    max_files: int = Form(10)
):
    """
    Analyze multiple images in batch.
    
    Parameters:
    -----------
    files : List[UploadFile]
        List of image files to analyze
    scale : float
        Scale factor for all images
    max_files : int
        Maximum number of files to process
    
    Returns:
    --------
    dict : Batch job information
    """
    
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analysis modules not available")
    
    # Limit number of files
    if len(files) > max_files:
        files = files[:max_files]
    
    # Validate all files
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
    
    # Generate batch job ID
    job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Initialize job status
    job_status[job_id] = AnalysisResult(job_id, "queued")
    
    # Add background task
    background_tasks.add_task(
        run_batch_analysis_task,
        job_id=job_id,
        files=files,
        scale=scale
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "files_count": len(files),
        "message": "Batch analysis started. Use /status/{job_id} to check progress.",
        "estimated_time": f"{len(files) * 2}-{len(files) * 5} minutes"
    }

@app.get("/sample-analysis")
async def sample_analysis(
    background_tasks: BackgroundTasks,
    size: int = 600,
    n_grains: int = 50,
    noise_level: float = 0.05,
    scale: float = 2.5
):
    """
    Generate and analyze a sample microstructure.
    
    Parameters:
    -----------
    size : int
        Image size (pixels)
    n_grains : int
        Number of grains to generate
    noise_level : float
        Noise level (0.0 to 0.2)
    scale : float
        Scale factor
    
    Returns:
    --------
    dict : Job information
    """
    
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analysis modules not available")
    
    # Generate job ID
    job_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Initialize job status
    job_status[job_id] = AnalysisResult(job_id, "queued")
    
    # Add background task
    background_tasks.add_task(
        run_sample_analysis_task,
        job_id=job_id,
        size=size,
        n_grains=n_grains,
        noise_level=noise_level,
        scale=scale
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Sample analysis started. Use /status/{job_id} to check progress.",
        "estimated_time": "1-3 minutes"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of an analysis job.
    
    Parameters:
    -----------
    job_id : str
        Job identifier
    
    Returns:
    --------
    dict : Job status and results
    """
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = job_status[job_id]
    
    response = {
        "job_id": result.job_id,
        "status": result.status,
        "timestamp": result.timestamp
    }
    
    if result.status == "completed" and result.results:
        response["results"] = result.results
        response["download_url"] = f"/download/{job_id}"
    elif result.status == "failed" and result.error:
        response["error"] = result.error
    
    return response

@app.get("/download/{job_id}")
async def download_results(job_id: str, format: str = "json"):
    """
    Download analysis results.
    
    Parameters:
    -----------
    job_id : str
        Job identifier
    format : str
        Download format ('json', 'csv', 'txt')
    
    Returns:
    --------
    FileResponse : Results file
    """
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = job_status[job_id]
    
    if result.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Create download file
    output_dir = Path("outputs")
    
    if format == "json":
        file_path = output_dir / f"{job_id}_results.json"
        with open(file_path, 'w') as f:
            json.dump(result.results, f, indent=2, default=str)
        media_type = "application/json"
    
    elif format == "csv" and "grain_data" in result.results:
        import pandas as pd
        file_path = output_dir / f"{job_id}_grain_data.csv"
        df = pd.DataFrame(result.results["grain_data"])
        df.to_csv(file_path, index=False)
        media_type = "text/csv"
    
    elif format == "txt":
        file_path = output_dir / f"{job_id}_report.txt"
        with open(file_path, 'w') as f:
            f.write(result.results.get("report", "No report available"))
        media_type = "text/plain"
    
    else:
        raise HTTPException(status_code=400, detail="Invalid format or no data available")
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from memory
    del job_status[job_id]
    
    # Clean up files
    output_dir = Path("outputs")
    for file_path in output_dir.glob(f"{job_id}_*"):
        file_path.unlink(missing_ok=True)
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/jobs")
async def list_jobs():
    """List all jobs and their status."""
    
    jobs = []
    for job_id, result in job_status.items():
        jobs.append({
            "job_id": result.job_id,
            "status": result.status,
            "timestamp": result.timestamp
        })
    
    return {
        "jobs": jobs,
        "total_jobs": len(jobs)
    }

# Background task functions
async def run_analysis_task(job_id: str, file: UploadFile, scale: float, analysis_params: dict):
    """Background task for single image analysis."""
    
    try:
        # Update status
        job_status[job_id].status = "processing"
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Initialize analyzer
        analyzer = MicrostructuralAnalyzer(tmp_file_path, scale_pixels_per_micron=scale)
        
        # Set parameters
        preprocess_params = {
            'denoise': analysis_params['denoise'],
            'enhance_contrast': analysis_params['enhance_contrast']
        }
        
        # Run analysis
        results = analyzer.run_complete_analysis(preprocess_params=preprocess_params)
        
        # Convert results to JSON-serializable format
        json_results = {
            "analysis_summary": results,
            "grain_data": analyzer.grain_properties.to_dict('records') if analyzer.grain_properties is not None else [],
            "defects": analyzer.defects if analyzer.defects else {},
            "phases": analyzer.phases['data'].to_dict('records') if analyzer.phases else [],
            "report": analyzer.generate_report()
        }
        
        # Advanced analysis if requested
        if analysis_params.get('run_advanced', False):
            advanced = AdvancedMicrostructuralAnalyzer(analyzer)
            features = advanced.extract_ml_features()
            anomalies = advanced.detect_anomalous_grains()
            clusters = advanced.cluster_grains()
            
            json_results["advanced"] = {
                "features": features.to_dict('records') if features is not None else [],
                "anomalies": anomalies.to_dict('records') if anomalies is not None else [],
                "clusters": clusters.to_dict('records') if clusters is not None else []
            }
        
        # Update job status
        job_status[job_id] = AnalysisResult(job_id, "completed", json_results)
        
        # Clean up
        os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {str(e)}")
        job_status[job_id] = AnalysisResult(job_id, "failed", error=str(e))

async def run_batch_analysis_task(job_id: str, files: List[UploadFile], scale: float):
    """Background task for batch analysis."""
    
    try:
        # Update status
        job_status[job_id].status = "processing"
        
        batch_results = []
        
        for i, file in enumerate(files):
            try:
                # Save file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                # Analyze
                analyzer = MicrostructuralAnalyzer(tmp_file_path, scale_pixels_per_micron=scale)
                results = analyzer.run_complete_analysis()
                
                # Extract key metrics
                file_results = {
                    'filename': file.filename,
                    'total_grains': results.get('grain_analysis', {}).get('total_grains', 0),
                    'avg_grain_size': results.get('grain_analysis', {}).get('avg_grain_size_microns', 0),
                    'total_pores': results.get('defect_analysis', {}).get('total_pores', 0),
                    'total_cracks': results.get('defect_analysis', {}).get('total_cracks', 0),
                    'pore_area_fraction': results.get('defect_analysis', {}).get('pore_area_fraction', 0)
                }
                
                batch_results.append(file_results)
                
                # Clean up
                os.unlink(tmp_file_path)
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                continue
        
        # Summary statistics
        if batch_results:
            import pandas as pd
            df = pd.DataFrame(batch_results)
            summary = {
                'total_files': len(batch_results),
                'avg_grains_per_image': df['total_grains'].mean(),
                'avg_grain_size': df['avg_grain_size'].mean(),
                'total_defects': (df['total_pores'] + df['total_cracks']).sum()
            }
        else:
            summary = {'total_files': 0, 'error': 'No files processed successfully'}
        
        # Update job status
        json_results = {
            "batch_summary": summary,
            "individual_results": batch_results
        }
        
        job_status[job_id] = AnalysisResult(job_id, "completed", json_results)
        
    except Exception as e:
        logger.error(f"Batch analysis failed for job {job_id}: {str(e)}")
        job_status[job_id] = AnalysisResult(job_id, "failed", error=str(e))

async def run_sample_analysis_task(job_id: str, size: int, n_grains: int, noise_level: float, scale: float):
    """Background task for sample analysis."""
    
    try:
        # Update status
        job_status[job_id].status = "processing"
        
        # Create sample
        sample_path = f"temp/sample_{job_id}.png"
        create_sample_microstructure(
            size=(size, size),
            n_grains=n_grains,
            noise_level=noise_level,
            save_path=sample_path
        )
        
        # Analyze
        analyzer = MicrostructuralAnalyzer(sample_path, scale_pixels_per_micron=scale)
        results = analyzer.run_complete_analysis()
        
        # Convert results
        json_results = {
            "sample_parameters": {
                "size": size,
                "n_grains": n_grains,
                "noise_level": noise_level,
                "scale": scale
            },
            "analysis_summary": results,
            "grain_data": analyzer.grain_properties.to_dict('records') if analyzer.grain_properties is not None else [],
            "defects": analyzer.defects if analyzer.defects else {},
            "report": analyzer.generate_report()
        }
        
        # Update job status
        job_status[job_id] = AnalysisResult(job_id, "completed", json_results)
        
        # Clean up
        os.unlink(sample_path)
        
    except Exception as e:
        logger.error(f"Sample analysis failed for job {job_id}: {str(e)}")
        job_status[job_id] = AnalysisResult(job_id, "failed", error=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
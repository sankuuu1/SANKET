# 🚀 Deployment Guide for Microstructural Analysis Toolkit

This guide provides comprehensive instructions for deploying your microstructural analysis toolkit across different platforms and environments.

## 📋 Table of Contents

1. [Local Development Setup](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Web Application Deployment](#web-application-deployment)
4. [REST API Deployment](#rest-api-deployment)
5. [Cloud Platform Deployment](#cloud-platforms)
6. [Desktop Application](#desktop-application)
7. [Jupyter Notebook Environment](#jupyter-environment)
8. [Troubleshooting](#troubleshooting)

---

## 🖥️ Local Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Installation Steps

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd microstructural-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation:**
   ```bash
   python simple_demo.py
   ```

4. **Run the web application:**
   ```bash
   streamlit run streamlit_app.py
   ```
   Open http://localhost:8501 in your browser.

5. **Run the API server:**
   ```bash
   python api_server.py
   ```
   API documentation available at http://localhost:8000/docs

---

## 🐳 Docker Deployment

### Basic Docker Setup

1. **Build the image:**
   ```bash
   docker build -t microstructure-analyzer .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 microstructure-analyzer
   ```

### Using Docker Compose

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Services available:**
   - Web app: http://localhost:8501
   - Jupyter: http://localhost:8888 (with `development` profile)
   - API: http://localhost:8000 (with `api` profile)

3. **Run specific profiles:**
   ```bash
   # Development with Jupyter
   docker-compose --profile development up -d
   
   # API service
   docker-compose --profile api up -d
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

---

## 🌐 Web Application Deployment

### Streamlit Cloud (Recommended)

1. **Prerequisites:**
   - GitHub account
   - Public repository with your code

2. **Deployment steps:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository and branch
   - Set main file: `streamlit_app.py`
   - Deploy automatically

3. **Configuration:**
   ```toml
   # .streamlit/config.toml
   [server]
   headless = true
   port = $PORT
   enableCORS = false
   ```

### Railway

1. **Connect GitHub repository:**
   - Visit [railway.app](https://railway.app)
   - Connect your GitHub account
   - Select the repository

2. **Configuration:**
   ```json
   # railway.json (already provided)
   {
     "build": {
       "builder": "DOCKERFILE"
     },
     "deploy": {
       "startCommand": "streamlit run streamlit_app.py --server.port=$PORT"
     }
   }
   ```

### Render

1. **Create new Web Service:**
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

### Heroku

1. **Install Heroku CLI and login:**
   ```bash
   heroku login
   ```

2. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

4. **Required files (already provided):**
   - `Procfile`: Process definition
   - `runtime.txt`: Python version
   - `requirements.txt`: Dependencies

---

## 🔌 REST API Deployment

### Local API Server

```bash
# Development server
python api_server.py

# Production server with Gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api_server:app --bind 0.0.0.0:8000
```

### Docker API Deployment

```bash
# Build API image
docker build -f Dockerfile.api -t microstructure-api .

# Run API container
docker run -p 8000:8000 microstructure-api
```

### Cloud API Deployment

#### AWS Lambda + API Gateway

1. **Install serverless framework:**
   ```bash
   npm install -g serverless
   pip install serverless-wsgi
   ```

2. **Create serverless.yml:**
   ```yaml
   service: microstructure-api
   provider:
     name: aws
     runtime: python3.9
   functions:
     api:
       handler: wsgi_handler.handler
       events:
         - http: ANY /
         - http: ANY /{proxy+}
   ```

#### Google Cloud Functions

```bash
gcloud functions deploy microstructure-analysis \
  --runtime python39 \
  --trigger-http \
  --entry-point app
```

---

## ☁️ Cloud Platform Deployment

### Amazon Web Services (AWS)

#### EC2 Instance

1. **Launch EC2 instance:**
   - Choose Amazon Linux 2 or Ubuntu
   - Select appropriate instance type (t3.medium recommended)
   - Configure security groups (ports 8501, 8000)

2. **Setup on instance:**
   ```bash
   # Update system
   sudo yum update -y  # Amazon Linux
   # sudo apt update && sudo apt upgrade -y  # Ubuntu
   
   # Install Python and pip
   sudo yum install python3 python3-pip git -y
   
   # Clone repository
   git clone <your-repo-url>
   cd microstructural-analysis
   
   # Install dependencies
   pip3 install -r requirements.txt
   
   # Run application
   streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   ```

#### ECS (Docker)

1. **Push image to ECR:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker tag microstructure-analyzer:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/microstructure-analyzer:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/microstructure-analyzer:latest
   ```

2. **Create ECS task definition and service**

### Google Cloud Platform (GCP)

#### App Engine

1. **Deploy using app.yaml:**
   ```bash
   gcloud app deploy app.yaml
   ```

#### Cloud Run

1. **Build and deploy:**
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/microstructure-analyzer
   gcloud run deploy --image gcr.io/[PROJECT-ID]/microstructure-analyzer --platform managed
   ```

### Microsoft Azure

#### Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name microstructure-analyzer \
  --image your-registry/microstructure-analyzer:latest \
  --ports 8501 \
  --environment-variables STREAMLIT_SERVER_PORT=8501
```

---

## 🖥️ Desktop Application

### Using PyInstaller

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Create desktop app:**
   ```bash
   pyinstaller --onefile --windowed desktop_app.py
   ```

3. **Create desktop_app.py:**
   ```python
   import tkinter as tk
   from tkinter import ttk, filedialog, messagebox
   import subprocess
   import threading
   import webbrowser
   
   class MicrostructureAnalyzerGUI:
       def __init__(self, root):
           self.root = root
           self.root.title("Microstructural Analysis Toolkit")
           self.setup_ui()
   
       def setup_ui(self):
           # Create main frame
           main_frame = ttk.Frame(self.root, padding="10")
           main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
           
           # Title
           title_label = ttk.Label(main_frame, text="Microstructural Analysis Toolkit", 
                                 font=("Arial", 16, "bold"))
           title_label.grid(row=0, column=0, columnspan=2, pady=10)
           
           # Buttons
           ttk.Button(main_frame, text="Open Web Interface", 
                     command=self.open_web_interface).grid(row=1, column=0, pady=5, padx=5)
           
           ttk.Button(main_frame, text="Open Jupyter Notebook", 
                     command=self.open_jupyter).grid(row=1, column=1, pady=5, padx=5)
           
           ttk.Button(main_frame, text="Run Analysis", 
                     command=self.run_analysis).grid(row=2, column=0, pady=5, padx=5)
           
           ttk.Button(main_frame, text="Exit", 
                     command=self.root.quit).grid(row=2, column=1, pady=5, padx=5)
   
       def open_web_interface(self):
           # Start Streamlit in background
           threading.Thread(target=self.start_streamlit, daemon=True).start()
           # Open browser
           webbrowser.open('http://localhost:8501')
   
       def start_streamlit(self):
           subprocess.run(['streamlit', 'run', 'streamlit_app.py'])
   
       def open_jupyter(self):
           subprocess.run(['jupyter', 'notebook', 'Microstructural_Analysis_Tutorial.ipynb'])
   
       def run_analysis(self):
           file_path = filedialog.askopenfilename(
               title="Select Microstructural Image",
               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp")]
           )
           if file_path:
               # Run analysis
               messagebox.showinfo("Analysis", f"Starting analysis of {file_path}")
   
   if __name__ == "__main__":
       root = tk.Tk()
       app = MicrostructureAnalyzerGUI(root)
       root.mainloop()
   ```

---

## 📓 Jupyter Notebook Environment

### Local Jupyter Setup

1. **Install Jupyter:**
   ```bash
   pip install jupyter jupyterlab
   ```

2. **Start Jupyter:**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. **Open tutorial:**
   Navigate to `Microstructural_Analysis_Tutorial.ipynb`

### JupyterHub Deployment

1. **Docker JupyterHub:**
   ```bash
   docker run -p 8000:8000 -d --name jupyterhub jupyterhub/jupyterhub jupyterhub
   ```

2. **Custom JupyterHub configuration:**
   ```python
   # jupyterhub_config.py
   c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
   c.DockerSpawner.image = 'microstructure-analyzer:latest'
   ```

### Google Colab

1. **Upload notebook to GitHub**
2. **Open in Colab:**
   - Go to https://colab.research.google.com
   - Select "GitHub" tab
   - Enter repository URL
   - Select the notebook

3. **Install dependencies in Colab:**
   ```python
   !pip install -r requirements.txt
   ```

---

## 🔧 Configuration and Environment Variables

### Environment Variables

```bash
# .env file
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
API_HOST=0.0.0.0
API_PORT=8000
SCALE_DEFAULT=2.0
MAX_UPLOAD_SIZE=200
```

### Streamlit Configuration

```toml
# .streamlit/config.toml
[global]
dataFrameSerialization = "legacy"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

---

## 🔍 Monitoring and Logging

### Application Monitoring

1. **Health check endpoints:**
   - Streamlit: `http://localhost:8501/_stcore/health`
   - API: `http://localhost:8000/health`

2. **Logging configuration:**
   ```python
   import logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('app.log'),
           logging.StreamHandler()
       ]
   )
   ```

### Performance Monitoring

1. **Resource usage:**
   ```bash
   # Monitor with htop, top, or ps
   ps aux | grep streamlit
   ```

2. **Application metrics:**
   - Memory usage
   - Processing time
   - Error rates
   - User sessions

---

## 🛠️ Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8501
# Kill process
kill -9 <PID>
```

#### Memory Issues
- Reduce image sizes
- Process images in batches
- Increase container memory limits

#### Dependencies Conflicts
```bash
# Create virtual environment
python -m venv microstructure_env
source microstructure_env/bin/activate  # Linux/Mac
# microstructure_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Docker Build Issues
```bash
# Clear Docker cache
docker system prune -a
# Rebuild with no cache
docker build --no-cache -t microstructure-analyzer .
```

### Performance Optimization

1. **Image optimization:**
   - Resize large images before processing
   - Use appropriate image formats (PNG for analysis)
   - Process regions of interest for large images

2. **Memory management:**
   - Process images in batches
   - Clear variables after processing
   - Use generators for large datasets

3. **Caching:**
   ```python
   @st.cache_data
   def load_and_process_image(image_path):
       # Cached function
       pass
   ```

### Debug Mode

```bash
# Streamlit debug mode
streamlit run streamlit_app.py --logger.level=debug

# API debug mode
python api_server.py --debug
```

---

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

### Cloud Platform Guides
- [AWS App Deployment](https://aws.amazon.com/getting-started/hands-on/deploy-python-application/)
- [Google Cloud App Engine](https://cloud.google.com/appengine/docs/python/)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)

### Support
- Check the project README.md for detailed usage instructions
- Review error logs for specific issues
- Ensure all dependencies are correctly installed
- Verify image formats and quality

---

## 🎯 Quick Deployment Commands

### Local Development
```bash
# Quick start
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Docker Deployment
```bash
# Build and run
docker build -t microstructure-analyzer .
docker run -p 8501:8501 microstructure-analyzer
```

### Cloud Deployment
```bash
# Streamlit Cloud: Push to GitHub and deploy via web interface
# Heroku
git push heroku main

# Railway/Render: Connect GitHub repository via web interface
```

---

*This deployment guide covers the most common scenarios. For specific use cases or custom requirements, refer to the platform-specific documentation or modify the configurations as needed.*
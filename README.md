# 🧠 NueroVision AI

**Advanced 3D Brain Tumor Segmentation System with Deep Learning and Uncertainty Quantification**

A comprehensive AI-powered medical imaging platform for brain tumor detection and segmentation using state-of-the-art 3D U-Net architecture with uncertainty estimation and professional web interface.

## 🔧 Tech Stack & Requirements

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![React](https://img.shields.io/badge/React-18+-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue)
![NiBabel](https://img.shields.io/badge/NiBabel-5.2+-green)
![scikit-image](https://img.shields.io/badge/scikit--image-0.21+-orange)
![Tailwind](https://img.shields.io/badge/Tailwind-CSS-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- � **3D Brain Tumor Segmentation** using advanced 3D U-Net architecture
- 🏥 **Multi-modal MRI Support** (T1, T1ce, T2, FLAIR) with intelligent preprocessing
- 🎯 **Test-Time Augmentation (TTA)** for robust and reliable predictions
- 📊 **Uncertainty Quantification** with Monte Carlo Dropout and confidence estimation
- 🔄 **Advanced Preprocessing Pipeline** with skull stripping, bias correction, and normalization
- � **Post-processing Refinement** with morphological operations and component filtering
- 🎨 **3D Visualization** with comprehensive volume rendering and slice views
- ⚡ **FastAPI Backend** with REST API for seamless integration
- 💻 **Modern React Frontend** with TypeScript and Tailwind CSS
- 📊 **Real-time Metrics** including Dice score, IoU, and volume statistics  

---

## 🏗️ Project Structure

```
NueroVisionAI/
│── backend/                    # FastAPI backend
│   ├── main.py                 # FastAPI application entry point
│   ├── routes/                 # API route handlers
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── predict.py         # Prediction API
│   │   └── history.py         # Prediction history
│   ├── models/                # Pre-trained model files
│   │   ├── tumornet_best.pkl  # Best model checkpoint
│   │   └── tumornet.onnx      # ONNX model for deployment
│   ├── services/              # Business logic
│   └── schemas.py             # Pydantic data models
│
│── core/                      # Core ML architecture
│   ├── 2D architecture/       # 2D CNN models and utilities
│   └── 3D architecture/       # 3D U-Net implementation
│       ├── model_3d.py        # 3D U-Net model architecture
│       ├── train_3d.py        # Training pipeline
│       ├── inference_3d.py    # Inference engine  
│       ├── preprocess_3d.py   # Advanced preprocessing
│       ├── postprocess_3d.py  # Post-processing pipeline
│       ├── visualize_3d.py    # 3D visualization tools
│       ├── metrics_3d.py      # Evaluation metrics
│       └── utils_3d.py        # Utility functions
│
│
│── checkpoints/               # Model checkpoints
│   └── 3d/
│       └── best_model.pth    # Best trained model (Dice: 0.8974)
│
│── data/                     # BraTS dataset
│   ├── BrainWithTumor/      # Tumor cases (T1, T1ce, T2, FLAIR)
│   ├── Healthy/             # Healthy brain scans
│   └── TumorOnly/           # Tumor-only segmentation masks
│
│── frontend/                # React + TypeScript + Tailwind frontend
│   ├── src/
│   │   ├── components/      # UI components
│   │   │   ├── AuthorsSection.tsx    # Team information
│   │   │   ├── HeroSection.tsx       # Landing page hero
│   │   │   ├── ImageUploader.tsx     # File upload component
│   │   │   └── ui/                   # Reusable UI components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API integration
│   │   └── assets/          # Static assets
│   ├── public/              # Public static files
│   ├── package.json         # Frontend dependencies
│   └── vite.config.ts       # Vite configuration
│
│── notebooks/               # Research and development notebooks
│   ├── 2D/                  # 2D CNN experiments
│   └── 3D/                  # 3D U-Net development
│       ├── 01_explore_data_3d.ipynb      # Data exploration
│       ├── 02_train_model_3d.ipynb       # Model training
│       ├── 03_infer_segmentation_3d.ipynb # Inference pipeline
│       └── 04_evaluate_results_3d.ipynb   # Results evaluation
│
│── test_results/            # Test inference outputs
│── outputs/                 # Training outputs and artifacts
│── results/                 # Evaluation results and metrics
│── test_model.py           # Standalone model testing script
│── requirements.txt        # Python dependencies
│── README.md              # Project documentation
└── .gitignore            # Git ignore rules
```

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, 3D U-Net, Light/Standard/Heavy architectures
- **Medical Imaging**: NiBabel, DICOM support, multi-modal MRI processing
- **Uncertainty Estimation**: Monte Carlo Dropout, Test-Time Augmentation
- **Preprocessing**: Skull stripping, bias correction, intensity normalization
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL/SQLite, JWT authentication
- **Frontend**: React 18, TypeScript, Tailwind CSS, Vite, shadcn/ui
- **Visualization**: 3D volume rendering, slice-by-slice views, overlay masks
- **Deployment**: Docker, REST API, Web application  

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/NueroVisionAI.git
cd NueroVisionAI
```

### 2️⃣ Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Backend Setup (FastAPI)

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**
- API Documentation: **http://localhost:8000/docs**

### 4️⃣ Frontend Setup (React + TypeScript)

```bash
cd frontend
npm install
npm run dev
```

The web application will be available at **http://localhost:5173**

### 5️⃣ Model Testing

```bash
# Test the trained model with sample data
python test_model.py
```

### 6️⃣ Training (Optional)

```bash
# Train a new model from scratch
cd "core/3D architecture"
python train_3d.py
```

---

## 🚀 Usage

1. **Open the Frontend UI** at http://localhost:5173
2. **Upload an MRI scan** (T1/T2/FLAIR format)
3. **Get Results**:
   - ✅ Tumor probability score
   - 📉 Epistemic + Aleatoric uncertainty estimates
   - 🔥 Grad-CAM++ heatmap visualization

### API Endpoints

- `POST /predict` - Upload MRI scan for tumor detection
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

---

## 📊 Performance Results

### Model Performance
- **Validation Dice Score**: 0.8974 (89.74%)
- **Training Epochs**: 17 epochs for optimal convergence
- **Model Size**: 1,402,561 parameters (Light architecture)
- **Inference Time**: ~140ms per case (CPU), <50ms (GPU)

### Technical Achievements
- **Robust Preprocessing**: Handles various MRI protocols and scanner types
- **Uncertainty Quantification**: Reliable confidence estimation for clinical decision support
- **Multi-Modal Support**: Seamless integration of T1, T1ce, T2, and FLAIR sequences
- **3D Visualization**: Comprehensive volume rendering with interactive slice navigation
- **Clinical Integration**: DICOM-compatible processing pipeline

---

## 🔬 Model Architecture

### 3D U-Net Implementation

Our system features a sophisticated 3D U-Net architecture with multiple variants:

- **Light Model**: 1.4M parameters - Fast inference, suitable for resource-constrained environments
- **Standard Model**: Balanced architecture for optimal performance-efficiency trade-off  
- **Heavy Model**: Maximum capacity for highest accuracy requirements

### Key Technical Features

- **Multi-Scale Processing**: Handles 3D volumes with adaptive input sizing (64³, 128³, 256³)
- **Advanced Preprocessing**: 
  - Skull stripping and brain extraction
  - Bias field correction using N4ITK
  - Intensity normalization and standardization
  - Isotropic resampling for consistent voxel spacing
- **Uncertainty Quantification**: 
  - Monte Carlo Dropout for epistemic uncertainty
  - Test-Time Augmentation (TTA) for robust predictions
  - Confidence thresholding and reliability estimation
- **Post-Processing Pipeline**:
  - Connected component analysis
  - Morphological operations (opening, closing)
  - Hole filling and surface smoothing
  - Volume statistics and quality metrics

---

## 📁 Key Files

- `core/3D architecture/model_3d.py` - 3D U-Net architecture implementation
- `core/3D architecture/train_3d.py` - Complete training pipeline
- `core/3D architecture/inference_3d.py` - Inference engine with TTA
- `core/3D architecture/preprocess_3d.py` - Advanced preprocessing pipeline
- `backend/main.py` - FastAPI application entry point
- `frontend/src/components/` - React components with TypeScript
- `test_model.py` - Standalone model testing script
- `checkpoints/3d/best_model.pth` - Trained model checkpoint

---

## 📈 Development Roadmap

- [ ] Deploy on cloud platform (Docker + AWS/GCP)
- [ ] Add support for additional MRI modalities (Diffusion, Perfusion)
- [ ] Implement clinical-grade user interface
- [ ] Add model performance benchmarking
- [ ] Integration with medical imaging standards (DICOM)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [BRATS Dataset](https://www.med.upenn.edu/sbia/brats2018.html) for brain tumor MRI scans
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Grad-CAM++](https://arxiv.org/abs/1710.11063) for explainable AI visualizations
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) and [Tailwind CSS](https://tailwindcss.com/) for the frontend

---

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:
- **Email**: team@neurovision.ai
- **Project Repository**: [GitHub](https://github.com/your-username/NueroVisionAI)

---

<div align="center">
  <strong>⭐ Star this repo if you found it helpful!</strong>
</div>

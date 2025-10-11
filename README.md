# 🧠 Brain Tumor Segmentation AI Platform

Advanced AI-powered brain tumor detection and analysis system with professional medical reporting capabilities.

## 🚀 Features

### Core Functionality

- **3D U-Net AI Model**: State-of-the-art brain tumor segmentation
- **Multi-Planar Visualization**: Axial, Coronal, and Sagittal slice viewers
- **Interactive 3D Rendering**: Volume visualization with tumor isosurfaces
- **Professional PDF Reports**: AI-generated medical analysis with diagnostic images

### Enhanced User Experience

- **Smooth Plotly Sliders**: Fluid navigation through brain slices
- **Real-time Processing**: Live data streaming and progress indicators
- **AI-Powered Analysis**: Local LLM integration using Ollama
- **Robust Error Handling**: Fallback analysis when services are unavailable

### Professional Reporting

- **LLM-Selected Diagnostic Slices**: AI chooses most relevant imaging views
- **Bounding Box Analysis**: Precise tumor location and morphometry
- **Consistent Page Layout**: Professional medical report formatting
- **Technical Specifications**: Detailed imaging parameters and QA metrics

## 📁 Project Structure

```
BrainTumorSegmentation/
├── app.py                    # Main Streamlit application
├── model.py                  # 3D U-Net model and prediction logic
├── ai_report.py             # AI-powered medical analysis
├── pdf_report_generator.py  # Professional PDF report creation
├── main.py                  # Original CLI interface
├── Vis3d.py                 # 3D visualization utilities
├── VisSlices.py            # 2D slice visualization utilities
└── Flairbased_2/           # Model weights directory
    └── model_epoch_45 (1).pt
```

## 🛠️ Setup & Installation

### Prerequisites

# Brain Tumor Segmentation — Clinical Analysis Platform

Comprehensive research-grade toolkit for FLAIR brain MRI tumor segmentation, visualization, clinical prediction, and professional PDF reporting.

This repository contains a Streamlit app and supporting modules that perform:

- 3D U-Net segmentation of brain tumors from FLAIR volumes
- Multi-planar 2D slice viewers and interactive 3D rendering (Plotly)
- AI-powered medical analysis using a local LLM (Ollama) with graceful fallbacks
- Clinical prediction models (H5) for tumor type and growth-rate (optional)
- Professional, publication-quality PDF report generation

## Quick links

- Run (web): `streamlit run app.py`
- CLI: `python main.py`
- Example data: `BRATS/` (FLAIR volumes)

## What you get

- Interactive web UI (Streamlit) to upload NIfTI files and run segmentation
- Automatic selection of diagnostic slices and bounding-box ROI analysis
- Downloadable NIfTI masks, AI analysis text reports, and professional PDF reports
- Clinical form and H5-based predictions for tumor type/growth (optional)

## Repository layout (key files)

```
BrainTumorSegmentation/
├── app.py                    # Main Streamlit application
├── main.py                   # CLI entrypoint
├── model.py                  # 3D U-Net model and analysis helpers
├── ai_report.py              # LLM-backed AI analysis (Ollama integration)
├── pdf_report_generator.py   # Professional PDF report generator
├── clinical_form.py          # Clinical form UI and helpers
├── clinical_models.py        # H5 clinical prediction model wrappers
├── theme_manager.py          # UI theme manager
├── Flairbased_2/             # Trained PyTorch model weights (.pt)
├── models/                   # H5 models used for clinical predictions
├── BRATS/                    # Example / sample MRI volumes
├── test_*.py                 # Unit tests
└── assets/                   # (optional) screenshots and report images
```

## Requirements

- Python 3.8+ (3.10/3.11 recommended)
- Optional GPU with CUDA for faster segmentation (PyTorch)

Core Python packages (install with pip):

```powershell
pip install -r requirements.txt
# or, install common dependencies directly
pip install streamlit torch torchvision nibabel numpy matplotlib plotly scikit-image requests pillow fpdf reportlab
```

Notes:

- `tensorflow` is optional and required only if you want to run the H5 clinical prediction models in `models/*.h5`.
- If you plan to use the LLM features, install and run Ollama locally: https://ollama.ai/

## Setup (Ollama for AI analysis)

1. Install Ollama on your machine and run the server (default listens on port 11434).
2. Pull the LLM(s) you want to use (example):

```powershell
ollama pull llama3:latest
ollama pull mistral:latest
```

If Ollama is not available the app will automatically use fallback text analysis included in the code.

## How to run

Web UI (recommended):

```powershell
cd "d:\BrainTumorSegmentation\BrainTumorSegmentation"
streamlit run app.py
```

Then open http://localhost:8501 in your browser (or the port you chose).

CLI:

```powershell
python main.py
```

## Typical workflow

1. Upload a FLAIR NIfTI file (.nii or .nii.gz) via the Streamlit sidebar.
2. Click "Run Segmentation Analysis" to run the 3D U-Net model.
3. Inspect 2D slice viewers (axial/coronal/sagittal) and the interactive 3D rendering.
4. Review automatically selected key slices and AI textual analysis.
5. (Optional) Complete the clinical form to run H5-based predictions for tumor type/growth.
6. Export segmentation mask (.nii), AI report (.txt), or professional PDF report (.pdf).

## Models & Data

- PyTorch segmentation weights are expected at `Flairbased_2/model_epoch_45 (1).pt` (relative path). Update `model.py` if you move the file.
- Clinical H5 models (optional) are in `models/brain_tumor_type.h5` and `models/brain_tumor_growth.h5`.

## Tests

- Unit tests are included (examples: `test_app_startup.py`, `test_model_loading.py`, `test_clinical_workflow.py`). Run them with:

```powershell
python -m pytest -q
```

## Troubleshooting & tips

- If the model fails to load on Windows, check backslash escaping in the path inside `model.py` or set an absolute path.
- If TensorFlow/H5 models are missing, the clinical prediction tab will remain disabled; this is optional.
- Ollama connection errors fall back to a built-in textual analysis — the app remains usable.
- For large volumes, lower the 3D resolution slider in the sidebar to speed up the 3D rendering.

## 📸 Screenshots (images)

#### 🏠 Home Page

![HomePage](./images/Screenshot%202025-10-11%20172617.png)

#### 🧠 Brain Segmentation

![HomePage](./images/Screenshot%202025-10-11%20172653.png)

#### 🩻 Key Slices

![HomePage](./images/Screenshot%202025-10-11%20172719.png)

#### 🧩 3D Visualization

![HomePage](./images/Screenshot%202025-10-11%20172834.png)

#### 🧬 Clinical Analysis

![HomePage](./images/Screenshot%202025-10-11%20172857.png)

#### 📊 Classification Results

![HomePage](./images/Screenshot%202025-10-11%20172953.png)

#### 🤖 AI Report Generation

![HomePage](./images/Screenshot%202025-10-11%20173017.png)

#### 📄 PDF Export for the Report

![HomePage](./images/Screenshot%202025-10-11%20173230.png)

## Contributors & how to contribute

- Fork the repo, create a branch, open a pull request with a clear description.
- Follow the existing code style and add unit tests for new features.

## Medical disclaimer

This project is research software. AI outputs are experimental and must not be used for clinical decision-making. All findings should be reviewed by qualified medical professionals.

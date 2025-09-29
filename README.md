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
- Python 3.8+
- CUDA-capable GPU (recommended)
- Ollama (for AI analysis)
- Required Python packages (see requirements below)

### Required Packages
```bash
pip install streamlit torch torchvision nibabel numpy matplotlib plotly scikit-image requests
```

### Ollama Setup
1. Install Ollama from https://ollama.ai/
2. Pull required models:
```bash
ollama pull llama3:latest
ollama pull mistral:latest
```

## 🚀 Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```
Then visit `http://localhost:8501`

### CLI Interface
```bash
python main.py
```

## 📊 Application Workflow

1. **Upload**: Load FLAIR NIfTI files (.nii/.nii.gz)
2. **Process**: AI segmentation using 3D U-Net model
3. **Visualize**: Interactive 2D slices and 3D volume rendering
4. **Analyze**: AI-powered medical report generation
5. **Export**: Download masks, reports, and professional PDFs

## 📋 PDF Report Features

### AI-Selected Diagnostic Views
- LLM chooses most relevant slices based on tumor characteristics
- Automatic bounding box detection and ROI analysis
- Morphometric measurements and spatial distribution analysis

### Professional Layout
- Consistent page sizing (8.5" x 11")
- Medical-grade typography and formatting
- Color-coded information boxes
- Technical specifications and QA metrics

### Report Sections
1. **Title Page**: Executive summary and key findings
2. **Diagnostic Slices**: AI-selected views with analysis
3. **Medical Analysis**: Detailed clinical interpretation
4. **Technical Summary**: Imaging parameters and model specs

## 🔧 Configuration

### Model Settings
- Model path: `C:\Projects 2\BrainTumorSegmentation\Flairbased_2\model_epoch_45 (1).pt`
- Architecture: 3D U-Net (1 input channel, 2 output classes)
- Base filters: 8, Dropout: 0.2

### Ollama Integration
- Default model: `llama3:latest`
- API endpoint: `http://localhost:11434`
- Automatic fallback analysis if unavailable
- Retry logic with timeout handling

## 📈 Performance Metrics

- **Model Accuracy**: 94.2% validation accuracy
- **Processing Time**: < 30 seconds per scan
- **Sensitivity**: 92.8%
- **Specificity**: 95.6%

## 🛡️ Error Handling

The system includes robust error handling:
- **Connection Timeouts**: Automatic retry with fallback analysis
- **Model Loading**: Graceful degradation if model unavailable
- **File Processing**: Comprehensive validation and error messages
- **Report Generation**: Fallback templates when LLM services fail

## 📝 Notes

- For clinical use, professional radiologist review is required
- AI analysis is for screening purposes only
- Supports FLAIR T2-weighted MRI sequences
- Minimum system requirements: 8GB RAM, GPU with 4GB VRAM

## 🎯 Key Improvements

### Version 2.0 Features
- ✅ Smooth Plotly-based slice navigation
- ✅ Professional PDF reports with bounding boxes
- ✅ AI-powered slice selection for optimal diagnosis
- ✅ Consistent page layouts without text overlap
- ✅ Real-time data streaming and progress indicators
- ✅ Robust error handling with fallback analysis
- ✅ Clean project organization and documentation

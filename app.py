import os
import io
import tempfile
import numpy as np
import streamlit as st
import nibabel as nib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Tuple
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from model import Predict, analyze_tumor_prediction
from ai_report import BrainTumorAnalyzer, get_available_models
from pdf_report_generator import MedicalReportGenerator
from theme_manager import get_theme_manager
from clinical_form import get_clinical_form
from clinical_models import get_clinical_predictor

st.set_page_config(
    page_title="Brain Tumor Segmentation AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state first (before any components that use it)
# Initialize session state for segmentation
if 'segmentation_done' not in st.session_state:
    st.session_state.segmentation_done = False
if 'mri_volume' not in st.session_state:
    st.session_state.mri_volume = None
if 'pred_mask' not in st.session_state:
    st.session_state.pred_mask = None
if 'img_affine' not in st.session_state:
    st.session_state.img_affine = None
if 'voxel_spacing' not in st.session_state:
    st.session_state.voxel_spacing = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Initialize session state for clinical workflow
if 'tumor_analysis' not in st.session_state:
    st.session_state.tumor_analysis = None
if 'show_clinical_form' not in st.session_state:
    st.session_state.show_clinical_form = False
if 'clinical_form_data' not in st.session_state:
    st.session_state.clinical_form_data = {}
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'clinical_predictions' not in st.session_state:
    st.session_state.clinical_predictions = None

# Initialize theme manager and apply styling
theme_manager = get_theme_manager()
theme_manager.apply_theme()

# Render modern header
theme_manager.render_header()

@st.cache_data(show_spinner=False)
def _normalize_slice(img2d: np.ndarray) -> np.ndarray:
    img = img2d.astype(np.float32)
    vmin, vmax = np.percentile(img, [1, 99])
    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    else:
        m, s = img.mean(), img.std() + 1e-6
        img = (img - m) / s
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

def create_plotly_slice_viewer(mri: np.ndarray, pred: np.ndarray, axis: str, container_key: str):
    if axis == "axial":
        max_slice = mri.shape[2] - 1
        get_slice = lambda z: (_normalize_slice(mri[:, :, z].T), pred[:, :, z].T)
        axis_label = "Axial (Z)"
        initial_slice = max_slice // 2
    elif axis == "coronal":
        max_slice = mri.shape[1] - 1
        get_slice = lambda y: (_normalize_slice(mri[:, y, :].T), pred[:, y, :].T)
        axis_label = "Coronal (Y)"
        initial_slice = max_slice // 2
    else:
        max_slice = mri.shape[0] - 1
        get_slice = lambda x: (_normalize_slice(mri[x, :, :].T), pred[x, :, :].T)
        axis_label = "Sagittal (X)"
        initial_slice = max_slice // 2
    
    slice_idx = st.slider(
        axis_label,
        min_value=0,
        max_value=max_slice,
        value=initial_slice,
        key=f"slider_{axis}_{container_key}"
    )
    
    img2d, mask2d = get_slice(slice_idx)
    
    height, width = img2d.shape
    max_dim = max(height, width)
    
    if max_dim > 0:
        img_square = np.zeros((max_dim, max_dim))
        mask_square = np.zeros((max_dim, max_dim))
        
        h_start = (max_dim - height) // 2
        w_start = (max_dim - width) // 2
        
        img_square[h_start:h_start+height, w_start:w_start+width] = img2d
        if mask2d is not None:
            mask_square[h_start:h_start+height, w_start:w_start+width] = mask2d
    else:
        img_square = img2d
        mask_square = mask2d
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=img_square,
        colorscale='gray',
        showscale=False,
        name='Brain',
        hovertemplate='Intensity: %{z:.3f}<extra></extra>'
    ))
    
    if mask2d is not None and mask2d.sum() > 0:
        mask_overlay = np.where(mask_square > 0, mask_square, np.nan)
        fig.add_trace(go.Heatmap(
            z=mask_overlay,
            colorscale=[[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,100,100,0.7)']],
            showscale=False,
            name='Tumor',
            hovertemplate='Tumor detected<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': f"{axis_label} - Slice {slice_idx}/{max_slice}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': '#333333'}
        },
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            constrain="domain"
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        autosize=True
    )
    
    fig.update_xaxes(range=[0, img_square.shape[1]], fixedrange=False)
    fig.update_yaxes(range=[0, img_square.shape[0]], fixedrange=False)
    
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{axis}_{container_key}_{slice_idx}")

def create_interactive_3d(mri: np.ndarray, pred: np.ndarray, scale: float = 0.25):
    from skimage.transform import resize
    
    new_shape = tuple(max(1, int(s * scale)) for s in mri.shape)
    img_small = resize(mri, new_shape, preserve_range=True, anti_aliasing=True)
    pred_small = resize(pred, new_shape, order=0, preserve_range=True, anti_aliasing=False)
    
    img_small = (img_small - img_small.min()) / (img_small.max() - img_small.min() + 1e-8)
    img_small = (img_small * 255).astype(np.uint8)
    pred_bin = (pred_small > 0).astype(np.uint8)
    
    x, y, z = np.mgrid[0:img_small.shape[0], 0:img_small.shape[1], 0:img_small.shape[2]]
    
    fig = go.Figure()
    
    if pred_bin.sum() > 0:
        fig.add_trace(go.Isosurface(
            x=x.flatten(), 
            y=y.flatten(), 
            z=z.flatten(),
            value=pred_bin.flatten(),
            isomin=0.5, 
            isomax=1.0,
            opacity=0.8,
            surface_count=1,
            colorscale='Reds',
            name='Tumor',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
    
    fig.add_trace(go.Volume(
        x=x.flatten(), 
        y=y.flatten(), 
        z=z.flatten(),
        value=img_small.flatten(),
        opacity=0.1,
        surface_count=8,
        colorscale='Gray',
        name='Brain',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(
        title="Interactive 3D Brain Visualization",
        scene=dict(
            xaxis_title="X", 
            yaxis_title="Y", 
            zaxis_title="Z",
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def _compute_metrics(pred: np.ndarray, voxel_spacing: Tuple[float, float, float] | None) -> dict:
    voxels = int((pred > 0).sum())
    total = int(np.prod(pred.shape))
    frac = voxels / max(1, total)
    out = {
        "mask_voxels": voxels,
        "mask_fraction": frac,
    }
    if voxel_spacing is not None:
        vx_vol = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
        out["voxel_volume_mm3"] = vx_vol
        out["mask_volume_mm3"] = voxels * vx_vol
        out["mask_volume_mL"] = out["mask_volume_mm3"] / 1000.0
    return out

@st.cache_data(show_spinner=False)
def _identify_key_slices(mri: np.ndarray, pred_mask: np.ndarray, num_slices: int = 4) -> dict:
    """
    Identify the most diagnostically important slices with highest tumor content.
    Returns dict with key slices for each axis and their tumor content.
    """
    key_slices = {
        'axial': [],
        'coronal': [],
        'sagittal': []
    }

    axes_info = {
        'axial': {'axis': 2, 'shape_idx': 2},
        'coronal': {'axis': 1, 'shape_idx': 1},
        'sagittal': {'axis': 0, 'shape_idx': 0}
    }

    for axis_name, info in axes_info.items():
        max_slice = mri.shape[info['shape_idx']] - 1

        tumor_content = []
        for slice_idx in range(max_slice + 1):
            if axis_name == 'axial':
                slice_mask = pred_mask[:, :, slice_idx]
            elif axis_name == 'coronal':
                slice_mask = pred_mask[:, slice_idx, :]
            else:
                slice_mask = pred_mask[slice_idx, :, :]

            tumor_pixels = int(slice_mask.sum())
            total_pixels = int(slice_mask.size)
            tumor_percentage = (tumor_pixels / max(1, total_pixels)) * 100

            tumor_content.append({
                'slice_idx': slice_idx,
                'tumor_pixels': tumor_pixels,
                'tumor_percentage': tumor_percentage,
                'total_pixels': total_pixels
            })

        tumor_content.sort(key=lambda x: x['tumor_pixels'], reverse=True)

        # Choose slices: prioritize the single best from each axis, then fill remaining by overall tumor content
        if tumor_content and tumor_content[0]['tumor_pixels'] > 0:
            best = tumor_content[0].copy()
            best['axis'] = axis_name
            best['max_slice'] = max_slice
            best['diagnostic_value'] = f"Maximum tumor cross-section ({best['tumor_percentage']:.1f}% of slice)"
            key_slices[axis_name].append(best)

    # Flatten and fill remaining slots by global ranking across all slices if needed
    all_candidates = []
    for axis_name, info in axes_info.items():
        max_slice = mri.shape[info['shape_idx']] - 1
        for slice_idx in range(max_slice + 1):
            if axis_name == 'axial':
                slice_mask = pred_mask[:, :, slice_idx]
            elif axis_name == 'coronal':
                slice_mask = pred_mask[:, slice_idx, :]
            else:
                slice_mask = pred_mask[slice_idx, :, :]
            tumor_pixels = int(slice_mask.sum())
            total_pixels = int(slice_mask.size)
            tumor_percentage = (tumor_pixels / max(1, total_pixels)) * 100
            all_candidates.append({
                'axis': axis_name,
                'slice_idx': slice_idx,
                'tumor_pixels': tumor_pixels,
                'tumor_percentage': tumor_percentage,
                'max_slice': max_slice,
                'diagnostic_value': f"High tumor burden ({tumor_percentage:.1f}% of slice)"
            })

    all_candidates.sort(key=lambda x: x['tumor_pixels'], reverse=True)

    # Start with the best per-axis, then add top remaining unique slices until num_slices reached
    selected = []
    for axis in ['axial', 'coronal', 'sagittal']:
        if key_slices[axis]:
            selected.append(key_slices[axis][0])

    for c in all_candidates:
        if len(selected) >= num_slices:
            break
        if c['tumor_pixels'] == 0:
            continue
        if not any((s['axis'] == c['axis'] and s['slice_idx'] == c['slice_idx']) for s in selected):
            selected.append(c)

    # Organize by axis again for convenience
    out_by_axis = {'axial': [], 'coronal': [], 'sagittal': []}
    for s in selected:
        out_by_axis[s['axis']].append(s)

    return {
        'by_axis': out_by_axis,
        'top_slices': selected,
        'summary': {
            'total_key_slices': len(selected),
            'max_tumor_pixels': selected[0]['tumor_pixels'] if selected else 0,
            'axes_represented': list(sorted(set([s['axis'] for s in selected])))
        }
    }

# Session state already initialized above

with st.sidebar:
    st.header("🔧 Control Panel")
    
    # Theme selector
    theme_manager.get_theme_selector()
    st.markdown("---")
    
    # File upload
    uploaded = st.file_uploader("Upload FLAIR NIfTI (.nii/.nii.gz)", type=["nii", "nii.gz"])
    
    # Model selection
    available_models = get_available_models()
    if available_models:
        selected_model = st.selectbox("AI Analysis Model", available_models, index=0)
    else:
        selected_model = "llama3:latest"
        st.warning("⚠️ Ollama not running or no models found")
    
    # Main action buttons
    run_button = st.button("🚀 Run Segmentation Analysis", type="primary", use_container_width=True)
    
    if st.button("🗑️ Clear All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    scale_3d = st.slider("3D Resolution", 0.1, 0.6, value=0.25, step=0.05)
    auto_generate_report = st.checkbox("Auto-generate AI report", value=True)
    
    # Clinical workflow toggle
    if st.session_state.segmentation_done and st.session_state.get('tumor_analysis', {}).get('tumor_detected'):
        st.markdown("---")
        st.subheader("🎨 Clinical Analysis")
        if st.button("📝 Show Clinical Form", use_container_width=True):
            st.session_state.show_clinical_form = True
            st.rerun()

if uploaded and run_button:
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0, text="Starting analysis...")
        status_text = st.empty()
    
    try:
        status_text.text("📁 Loading NIfTI file...")
        progress_bar.progress(10, text="Loading file...")
        
        suffix = ".nii.gz" if uploaded.name.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_path = tmp.name
        
        img = nib.load(temp_path)
        mri = img.get_fdata()
        
        if mri.ndim == 4:
            mri = mri[..., 0]
        
        mri = np.ascontiguousarray(mri.astype(np.float32))
        
        progress_bar.progress(30, text="Preprocessing complete...")
        status_text.text("🧠 Running AI segmentation model...")
        
        pred = Predict(mri)
        
        progress_bar.progress(70, text="Segmentation complete...")
        status_text.text("📊 Computing metrics...")
        
        if pred.shape != mri.shape:
            from skimage.transform import resize
            pred = resize(pred, mri.shape, order=0, preserve_range=True, anti_aliasing=False).astype(pred.dtype)
        
        st.session_state.mri_volume = mri
        st.session_state.pred_mask = (pred > 0).astype(np.uint8)
        st.session_state.img_affine = img.affine
        st.session_state.segmentation_done = True
        
        # Analyze tumor characteristics
        tumor_analysis = analyze_tumor_prediction(mri, st.session_state.pred_mask)
        st.session_state.tumor_analysis = tumor_analysis
        
        hdr = img.header
        try:
            zooms = hdr.get_zooms()
            st.session_state.voxel_spacing = tuple(float(z) for z in zooms[:3]) if len(zooms) >= 3 else None
        except Exception:
            st.session_state.voxel_spacing = None
        
        progress_bar.progress(100, text="Analysis complete!")
        status_text.text("✅ Ready for visualization and AI report")
        
        try:
            os.remove(temp_path)
        except Exception:
            pass
        
        time.sleep(1)
        progress_container.empty()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")

if st.session_state.segmentation_done and st.session_state.mri_volume is not None:
    mri_volume = st.session_state.mri_volume
    pred_mask = st.session_state.pred_mask
    img_affine = st.session_state.img_affine
    voxel_spacing = st.session_state.voxel_spacing
    
    metrics = _compute_metrics(pred_mask, voxel_spacing)

    # Compute and store key slices (most tumor content) for use across tabs and reports
    key_slices = _identify_key_slices(mri_volume, pred_mask, num_slices=4)
    st.session_state.key_slices = key_slices
    
    tumor_analysis = st.session_state.tumor_analysis
    
    if tumor_analysis and tumor_analysis.get('tumor_detected'):
        st.success(f"🎯 Tumor Detected in {tumor_analysis.get('tumor_location', 'Unknown')} Region")
    else:
        st.info("✅ No Tumor Detected - Brain Appears Normal")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧠 Tumor Voxels", f"{metrics['mask_voxels']:,}")
    with col2:
        st.metric("📊 Brain Coverage", f"{metrics['mask_fraction']*100:.2f}%")
    with col3:
        if "mask_volume_mL" in metrics:
            st.metric("💧 Volume (mL)", f"{metrics['mask_volume_mL']:.2f}")
        else:
            st.metric("💧 Volume", "Unknown")
    with col4:
        if tumor_analysis and tumor_analysis.get('tumor_detected'):
            st.metric("🎯 Tumor Location", tumor_analysis.get('tumor_location', 'Unknown'))
        else:
            st.metric("🎯 Status", "Normal")
    
    st.markdown("---")
    
    # Conditionally show clinical tab based on tumor detection
    if tumor_analysis and tumor_analysis.get('tumor_detected'):
        tab1, tab_key, tab2, tab_clinical, tab3, tab4 = st.tabs([
            "📱 2D Slices", "⭐ Key Slices", "🌐 3D Interactive", 
            "🧬 Clinical Analysis", "📋 AI Report", "💾 Export"
        ])
    else:
        tab1, tab_key, tab2, tab3, tab4 = st.tabs([
            "📱 2D Slices", "⭐ Key Slices", "🌐 3D Interactive", 
            "📋 AI Report", "💾 Export"
        ])
    
    with tab1:
        st.subheader("🔍 Multi-Planar Brain Slice Viewer")
        st.caption("Navigate through brain slices to examine tumor segmentation in different anatomical planes")
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("##### Axial View")
            create_plotly_slice_viewer(mri_volume, pred_mask, "axial", "tab1")
        
        with col2:
            st.markdown("##### Coronal View")
            create_plotly_slice_viewer(mri_volume, pred_mask, "coronal", "tab1")
        
        with col3:
            st.markdown("##### Sagittal View")
            create_plotly_slice_viewer(mri_volume, pred_mask, "sagittal", "tab1")
    
    with tab_key:
        st.subheader("Key Slices (Highest Tumor Burden)")
        st.caption("These slices are automatically selected based on maximum tumor content and will be used in the report")

        if key_slices['top_slices']:
            cols = st.columns(2)
            for i, s in enumerate(key_slices['top_slices']):
                with cols[i % 2]:
                    axis = s['axis']
                    idx = s['slice_idx']
                    if axis == 'axial':
                        img2d = _normalize_slice(mri_volume[:, :, idx].T)
                        mask2d = pred_mask[:, :, idx].T
                        title = f"Axial - Slice {idx}/{s['max_slice']}"
                    elif axis == 'coronal':
                        img2d = _normalize_slice(mri_volume[:, idx, :].T)
                        mask2d = pred_mask[:, idx, :].T
                        title = f"Coronal - Slice {idx}/{s['max_slice']}"
                    else:
                        img2d = _normalize_slice(mri_volume[idx, :, :].T)
                        mask2d = pred_mask[idx, :, :].T
                        title = f"Sagittal - Slice {idx}/{s['max_slice']}"

                    height, width = img2d.shape
                    max_dim = max(height, width)
                    img_square = np.zeros((max_dim, max_dim))
                    mask_square = np.zeros((max_dim, max_dim))
                    h_start = (max_dim - height) // 2
                    w_start = (max_dim - width) // 2
                    img_square[h_start:h_start+height, w_start:w_start+width] = img2d
                    mask_square[h_start:h_start+height, w_start:w_start+width] = mask2d

                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(z=img_square, colorscale='gray', showscale=False))
                    if mask2d is not None and mask2d.sum() > 0:
                        mask_overlay = np.where(mask_square > 0, mask_square, np.nan)
                        fig.add_trace(go.Heatmap(z=mask_overlay,
                                                 colorscale=[[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,100,100,0.7)']],
                                                 showscale=False))
                    fig.update_layout(title=f"{title}<br><sup>{s['diagnostic_value']}</sup>",
                                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                                      margin=dict(l=10, r=10, t=60, b=10), height=350)
                    st.plotly_chart(fig, use_container_width=True, key=f"keyslice_{axis}_{idx}")
        else:
            st.info("No tumor detected; key slices are not available.")

    with tab2:
        st.subheader("3D Volume Rendering")
        if 'plotly_3d_fig' not in st.session_state or st.button("🔄 Regenerate 3D"):
            with st.spinner("Rendering 3D visualization..."):
                st.session_state.plotly_3d_fig = create_interactive_3d(mri_volume, pred_mask, scale=scale_3d)
        
        st.plotly_chart(st.session_state.plotly_3d_fig, use_container_width=True)
    
    # Clinical Analysis Tab (only shown when tumor is detected)
    if tumor_analysis and tumor_analysis.get('tumor_detected'):
        with tab_clinical:
            st.subheader("🧬 Clinical Analysis & Predictions")
            st.caption("Complete the clinical information form to get AI-powered tumor type and growth rate predictions")
            
            # Get clinical form instance
            clinical_form = get_clinical_form()
            
            # Show the clinical form
            form_submitted = clinical_form.render_form(tumor_analysis)
            
            # Show predictions if form was submitted
            if st.session_state.get('form_submitted') or st.session_state.get('clinical_predictions'):
                st.markdown("---")
                clinical_form.render_predictions()
                
                # Reset button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🔄 Reset Clinical Form", use_container_width=True):
                        clinical_form.reset_form()
    
    # AI Report Tab
    with tab3 if not (tumor_analysis and tumor_analysis.get('tumor_detected')) else tab3:
        st.subheader("AI Medical Analysis Report")
        
        if auto_generate_report and not st.session_state.analysis_done:
            analyzer = BrainTumorAnalyzer(model_name=selected_model)
            
            report_container = st.empty()
            report_text = ""
            
            with st.spinner("Generating AI analysis..."):
                for chunk in analyzer.analyze_prediction(metrics, mri_volume.shape, pred_mask, key_slices=st.session_state.get('key_slices', None)):
                    report_text += chunk
                    report_container.markdown(report_text + "▌")
                
                report_container.markdown(report_text)
                st.session_state.analysis_done = True
                st.session_state.ai_report = report_text
        
        elif st.session_state.analysis_done and 'ai_report' in st.session_state:
            st.markdown(st.session_state.ai_report)
        
        if st.button("🔄 Regenerate Report") or not auto_generate_report:
            analyzer = BrainTumorAnalyzer(model_name=selected_model)
            
            report_container = st.empty()
            report_text = ""
            
            with st.spinner("Generating new AI analysis..."):
                for chunk in analyzer.analyze_prediction(metrics, mri_volume.shape, pred_mask, key_slices=st.session_state.get('key_slices', None)):
                    report_text += chunk
                    report_container.markdown(report_text + "▌")
                
                report_container.markdown(report_text)
                st.session_state.ai_report = report_text
    
    # Export Tab
    with tab4 if not (tumor_analysis and tumor_analysis.get('tumor_detected')) else tab4:
        st.subheader("📋 Export Results")
        st.caption("Download segmentation masks, AI reports, and professional PDF reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Standard Downloads")
            
            if img_affine is not None:
                if 'download_data' not in st.session_state:
                    out_img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine=img_affine)
                    
                    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                        temp_path = tmp.name
                    
                    nib.save(out_img, temp_path)
                    with open(temp_path, 'rb') as f:
                        st.session_state.download_data = f.read()
                    
                    try:
                        os.unlink(temp_path)
                    except PermissionError:
                        pass
                
                st.download_button(
                    label="📥 Download Segmentation Mask (.nii)",
                    data=st.session_state.download_data,
                    file_name="brain_tumor_mask.nii",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
                if 'ai_report' in st.session_state:
                    st.download_button(
                        label="📄 Download AI Report (.txt)",
                        data=st.session_state.ai_report,
                        file_name="brain_tumor_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.info("No affine transformation available for NIfTI export")
        
        with col2:
            st.markdown("##### Professional PDF Report")
            
            if st.button("📑 Generate PDF Medical Report", type="secondary", use_container_width=True):
                with st.spinner("🤖 AI is selecting best diagnostic slices..."):
                    report_generator = MedicalReportGenerator(model_name=selected_model)
                    
                    progress_bar = st.progress(0, text="Analyzing tumor characteristics...")
                    progress_bar.progress(25, text="LLM selecting optimal slices...")
                    
                    progress_bar.progress(50, text="Generating medical analysis...")
                    
                    progress_bar.progress(75, text="Creating professional PDF...")
                    
                    try:
                        pdf_path = report_generator.generate_pdf_report(
                            mri_volume, pred_mask, metrics, recommended_slices=st.session_state.get('key_slices', None)
                        )
                        
                        with open(pdf_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        st.session_state.pdf_report_data = pdf_data
                        st.session_state.pdf_filename = os.path.basename(pdf_path)
                        
                        try:
                            os.remove(pdf_path)
                        except:
                            pass
                        
                        progress_bar.progress(100, text="PDF report ready!")
                        st.success("✅ Professional PDF report generated successfully!")
                        
                        time.sleep(1)
                        progress_bar.empty()
                        
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
                        progress_bar.empty()
            
            if 'pdf_report_data' in st.session_state:
                st.download_button(
                    label="📋 Download Professional PDF Report",
                    data=st.session_state.pdf_report_data,
                    file_name=st.session_state.get('pdf_filename', 'brain_tumor_report.pdf'),
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.info("💡 PDF includes AI-selected diagnostic slices, detailed medical analysis, and technical summary")

else:
    st.info("🚀 Upload a FLAIR NIfTI file and click 'Run Segmentation Analysis' to begin AI-powered brain tumor analysis.")

st.markdown("---")
st.caption("🧠 Powered by 3D U-Net AI • Made with ❤️ and Streamlit")

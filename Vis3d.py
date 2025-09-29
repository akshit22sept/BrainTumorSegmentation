import numpy as np
import plotly.graph_objects as go
from skimage.transform import resize

def visualize_3d_scan(flair_image, prediction_mask, scale=0.25, title="3D Scan with Prediction"):
    new_shape = tuple(int(s * scale) for s in flair_image.shape)
    
    img_small = resize(flair_image, new_shape, preserve_range=True, anti_aliasing=True)
    pred_small = resize(prediction_mask, new_shape, order=0, preserve_range=True, anti_aliasing=False)

    img_small = (img_small - img_small.min()) / (img_small.max() - img_small.min() + 1e-8)
    img_small = (img_small * 255).astype(np.uint8)
    
    pred_bin = (pred_small > 0).astype(np.uint8)

    x, y, z = np.mgrid[0:img_small.shape[0], 0:img_small.shape[1], 0:img_small.shape[2]]
    
    fig = go.Figure()

    fig.add_trace(go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=img_small.flatten(),
        opacity=0.1, 
        surface_count=12, 
        colorscale="Gray",
        name="FLAIR Scan"
    ))

    fig.add_trace(go.Isosurface(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=pred_bin.flatten(),
        isomin=0.5, 
        isomax=1.0,
        opacity=0.8, 
        surface_count=1, 
        colorscale="Blues",
        name="Prediction"
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800, 
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import find_contours

def visualize_slices(mri_volume, prediction_mask, title="Slice Viewer"):
    num_slices = mri_volume.shape[2]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    
    initial_slice_idx = num_slices // 2
    
    mri_plot = ax.imshow(mri_volume[:, :, initial_slice_idx], cmap='gray')
    ax.set_title(f"{title}\nSlice: {initial_slice_idx}/{num_slices-1}")
    ax.axis('off')
    
    dynamic_plots = []

    def draw_overlay_and_contours(slice_idx):
        for plot in dynamic_plots:
            plot.remove()
        dynamic_plots.clear()

        mask_slice = prediction_mask[:, :, slice_idx] > 0
        
        h, w = mask_slice.shape
        rgba_overlay = np.zeros((h, w, 4), dtype=np.float32)
        rgba_overlay[mask_slice, :] = [0.1, 0.1, 0.8, 0.35]
        
        overlay_plot = ax.imshow(rgba_overlay)
        dynamic_plots.append(overlay_plot)
        
        contours = find_contours(mask_slice, 0.5)
        for contour in contours:
            contour_plot = ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='cyan')
            dynamic_plots.extend(contour_plot)
            
    draw_overlay_and_contours(initial_slice_idx)
    
    slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    
    slice_slider = Slider(
        ax=slider_ax,
        label='Slice',
        valmin=0,
        valmax=num_slices - 1,
        valinit=initial_slice_idx,
        valstep=1
    )

    def update(val):
        slice_idx = int(slice_slider.val)
        mri_plot.set_data(mri_volume[:, :, slice_idx])
        draw_overlay_and_contours(slice_idx)
        ax.set_title(f"{title}\nSlice: {slice_idx}/{num_slices-1}")
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def show_interactive_orthogonal_views(mri_volume, prediction_mask):
    W, H, D = mri_volume.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(bottom=0.25)
    
    x, y, z = W // 2, H // 2, D // 2
    ax_plot = axes[0].imshow(mri_volume[:, :, z].T, cmap='gray', origin='lower')
    sag_plot = axes[1].imshow(mri_volume[x, :, :].T, cmap='gray', origin='lower')
    cor_plot = axes[2].imshow(mri_volume[:, y, :].T, cmap='gray', origin='lower')
    plots = [ax_plot, sag_plot, cor_plot]
    ax_contour = axes[0].contour(prediction_mask[:, :, z].T, colors='cyan', linewidths=1.5, levels=[0.5])
    sag_contour = axes[1].contour(prediction_mask[x, :, :].T, colors='cyan', linewidths=1.5, levels=[0.5])
    cor_contour = axes[2].contour(prediction_mask[:, y, :].T, colors='cyan', linewidths=1.5, levels=[0.5])
    contours = [ax_contour, sag_contour, cor_contour]

    axes[0].set_title('Axial View')
    axes[1].set_title('Sagittal View')
    axes[2].set_title('Coronal View')
    ax_slider_z = fig.add_axes([0.15, 0.15, 0.65, 0.03])
    ax_slider_y = fig.add_axes([0.15, 0.1, 0.65, 0.03])
    ax_slider_x = fig.add_axes([0.15, 0.05, 0.65, 0.03])
    slider_z = Slider(ax_slider_z, 'Axial (Z)', 0, D - 1, valinit=z, valstep=1)
    slider_y = Slider(ax_slider_y, 'Coronal (Y)', 0, H - 1, valinit=y, valstep=1)
    slider_x = Slider(ax_slider_x, 'Sagittal (X)', 0, W - 1, valinit=x, valstep=1)

    def update(val):
        for c_set in contours:
            for c in c_set.collections: c.remove()
        
        x_new, y_new, z_new = int(slider_x.val), int(slider_y.val), int(slider_z.val)
        
        plots[0].set_data(mri_volume[:, :, z_new].T)
        plots[1].set_data(mri_volume[x_new, :, :].T)
        plots[2].set_data(mri_volume[:, y_new, :].T)
        
        contours[0] = axes[0].contour(prediction_mask[:, :, z_new].T, colors='cyan', levels=[0.5])
        contours[1] = axes[1].contour(prediction_mask[x_new, :, :].T, colors='cyan', levels=[0.5])
        contours[2] = axes[2].contour(prediction_mask[:, y_new, :].T, colors='cyan', levels=[0.5])
        
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    plt.show()
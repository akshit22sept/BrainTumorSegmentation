from Vis3d import visualize_3d_scan
from VisSlices import visualize_slices,show_interactive_orthogonal_views
from model import Predict
import nibabel as nib



flair_path="BRATS/BraTS20_Training_030_flair.nii"
flair_img = nib.load(flair_path).get_fdata()
visualize_3d_scan(
    flair_image=flair_img, 
    prediction_mask=Predict(flair_img),
    title="Mask Visualization 3D"
)

flair_path="BRATS/BraTS20_Training_030_flair.nii"
flair_img = nib.load(flair_path).get_fdata()
pred=Predict(flair_img)
visualize_slices(flair_img,pred,
    title="Mask Visualization Slices"
)
show_interactive_orthogonal_views(flair_img,pred)
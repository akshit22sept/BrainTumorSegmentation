import os, glob, random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from collections import OrderedDict
import random
random.seed(42)
torch.cuda.empty_cache()






def crop_or_pad(enc_feat, dec_feat):
    _, _, D_enc, H_enc, W_enc = enc_feat.shape
    _, _, D_dec, H_dec, W_dec = dec_feat.shape

    dec_feat = F.pad(
        dec_feat,
        [max(0, W_enc - W_dec), 0,  
         max(0, H_enc - H_dec), 0,  
         max(0, D_enc - D_dec), 0]   
    )

    _, _, D_dec, H_dec, W_dec = dec_feat.shape

    d_start = (D_dec - D_enc) // 2
    h_start = (H_dec - H_enc) // 2
    w_start = (W_dec - W_enc) // 2

    dec_feat = dec_feat[:, :,
                        d_start:d_start + D_enc,
                        h_start:h_start + H_enc,
                        w_start:w_start + W_enc]

    return dec_feat




class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x








class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base_filters=16, dropout=0.3):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_filters, dropout)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_filters, base_filters*2, dropout)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(base_filters*2, base_filters*4, dropout)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ConvBlock3D(base_filters*4, base_filters*8, dropout)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base_filters*8, base_filters*16, dropout)

        self.up4 = nn.ConvTranspose3d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_filters*16, base_filters*8, dropout)

        self.up3 = nn.ConvTranspose3d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_filters*8, base_filters*4, dropout)

        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_filters*4, base_filters*2, dropout)

        self.up1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_filters*2, base_filters, dropout)

        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = crop_or_pad(e4, d4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = crop_or_pad(e3, d3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = crop_or_pad(e2, d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = crop_or_pad(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
    















device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet3D(in_channels=1, out_channels=2, base_filters=8, dropout=0.2).to(device)

checkpoint = torch.load("Flairbased_2\\model_epoch_45 (1).pt", map_location=device)
state_dict = checkpoint["model_state_dict"]

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

def extract_tumor_location_and_size(pred_mask):
    """
    Extract tumor location and size from prediction mask.
    Returns location (brain region) and size metrics.
    """
    if pred_mask.sum() == 0:
        return None, 0
    
    # Calculate tumor center of mass
    tumor_indices = np.where(pred_mask > 0)
    if len(tumor_indices[0]) == 0:
        return None, 0
    
    center_x = np.mean(tumor_indices[0])
    center_y = np.mean(tumor_indices[1])
    center_z = np.mean(tumor_indices[2])
    
    # Map anatomical location based on center coordinates
    # These are approximate brain region mappings
    height, width, depth = pred_mask.shape
    
    # Normalize coordinates to 0-1 range
    norm_x = center_x / height
    norm_y = center_y / width
    norm_z = center_z / depth
    
    # Simple anatomical region mapping (this is a basic approximation)
    # In a real application, you'd use anatomical atlases
    if norm_z < 0.3:  # Lower sections
        if norm_x < 0.4:
            location = "Temporal"
        else:
            location = "Cerebellum"
    elif norm_z < 0.7:  # Middle sections
        if norm_y < 0.4:
            location = "Frontal"
        elif norm_y > 0.6:
            location = "Occipital"
        else:
            location = "Parietal"
    else:  # Upper sections
        location = "Parietal"
    
    # Calculate tumor size (volume in voxels)
    tumor_size = int(pred_mask.sum())
    
    return location, tumor_size

def Predict(image):
    flair_norm = (image - image.mean()) / (image.std() + 1e-8)
    input_tensor = torch.from_numpy(flair_norm[None, None].astype(np.float32)).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return pred

def analyze_tumor_prediction(image, pred_mask):
    """
    Comprehensive tumor analysis including detection, location, and size.
    """
    tumor_detected = pred_mask.sum() > 0
    
    if not tumor_detected:
        return {
            'tumor_detected': False,
            'tumor_location': None,
            'tumor_size': 0,
            'brain_tumor_present': 0 
        }
    
    location, size = extract_tumor_location_and_size(pred_mask)
    
    return {
        'tumor_detected': True,
        'tumor_location': location,
        'tumor_size': size,
        'brain_tumor_present': 1  # For H5 model input
    }


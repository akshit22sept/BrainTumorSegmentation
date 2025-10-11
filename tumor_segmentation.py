"""
Tumor Segmentation Module
Handles 3D U-Net model loading and inference for brain tumor segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress KMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def crop_or_pad(enc_feat, dec_feat):
    """Crop or pad decoder features to match encoder features for skip connections."""
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
    """3D Convolutional block with BatchNorm, ReLU, and Dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
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
    """3D U-Net architecture for brain tumor segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2, 
                 base_filters: int = 8, dropout: float = 0.2):
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


class TumorSegmentationModel:
    """Main class for tumor segmentation inference."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the tumor segmentation model.
        
        Args:
            model_path: Path to the model checkpoint. If None, uses default path.
            device: Device to run inference on. If None, auto-detects CUDA availability.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or r"C:\Projects 2\BrainTumorSegmentation\Flairbased_2\model_epoch_45 (1).pt"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the 3D U-Net model from checkpoint."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Initialize model architecture
            self.model = UNet3D(in_channels=1, out_channels=2, base_filters=8, dropout=0.2)
            
            # Load checkpoint
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"]
            
            # Remove 'module.' prefix if present (for DataParallel compatibility)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess MRI image for model inference.
        
        Args:
            image: 3D numpy array representing FLAIR MRI scan
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Normalize to zero mean and unit variance
        flair_norm = (image - image.mean()) / (image.std() + 1e-8)
        
        # Convert to tensor and add batch and channel dimensions
        input_tensor = torch.from_numpy(flair_norm[None, None].astype(np.float32))
        
        return input_tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Perform tumor segmentation on input MRI image.
        
        Args:
            image: 3D numpy array representing FLAIR MRI scan
            
        Returns:
            Binary segmentation mask (0=background, 1=tumor)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check model initialization.")
        
        try:
            # Preprocess input
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            return pred.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "Loaded",
            "device": self.device,
            "model_path": self.model_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "3D U-Net",
            "input_channels": 1,
            "output_channels": 2,
            "base_filters": 8
        }


# Global model instance for backward compatibility
_global_model = None

def get_global_model() -> TumorSegmentationModel:
    """Get the global model instance, creating it if necessary."""
    global _global_model
    if _global_model is None:
        _global_model = TumorSegmentationModel()
    return _global_model

def Predict(image: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Perform tumor segmentation using the global model instance.
    """
    model = get_global_model()
    return model.predict(image)
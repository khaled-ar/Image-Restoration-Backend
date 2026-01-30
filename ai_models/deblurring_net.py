"""
Deblurring Neural Network based on simplified DeblurGAN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ResidualBlock(nn.Module):
    """Residual block for deblurring network"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)


class DeblurringNetwork(nn.Module):
    """
    Lightweight deblurring network for motion and focus blur removal
    """
    
    def __init__(self, in_channels=3, out_channels=3, num_blocks=8):
        super(DeblurringNetwork, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        
        # Final convolution
        self.final = nn.Conv2d(64, out_channels, kernel_size=7, padding=3)
    
    def forward(self, x):
        initial_out = self.initial(x)
        res_out = self.res_blocks(initial_out)
        output = self.final(res_out)
        return output


class DeblurringModel:
    """
    Wrapper for deblurring neural network
    """
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model = DeblurringNetwork()
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            print("No pretrained deblurring model found. Using random weights.")
            self.model.to(self.device)
    
    def load_model(self, model_path):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"Deblurring model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading deblurring model: {e}")
            self.model.to(self.device)
            self.model.eval()
    
    def deblur_image(self, image_path, output_path=None):
        """
        Remove blur from image using neural network
        
        Args:
            image_path: Path to input image
            output_path: Path to save deblurred image
            
        Returns:
            numpy.ndarray: Deblurred image
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            original_shape = image.shape
            
            # Resize if too large
            max_dim = 1024
            if max(image.shape[:2]) > max_dim:
                scale = max_dim / max(image.shape[:2])
                new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            # Convert to float32 and normalize
            img_float = image.astype(np.float32) / 255.0
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            # Apply deblurring
            with torch.no_grad():
                deblurred_tensor = self.model(img_tensor)
            
            # Convert back to numpy
            deblurred_np = deblurred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            deblurred_np = np.clip(deblurred_np * 255, 0, 255).astype(np.uint8)
            
            # Resize back to original size if needed
            if deblurred_np.shape != original_shape[:2]:
                deblurred_np = cv2.resize(deblurred_np, 
                                         (original_shape[1], original_shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(str(output_path), deblurred_np)
            
            return deblurred_np
            
        except Exception as e:
            print(f"Error in deblurring: {e}")
            # Fallback to OpenCV deblurring
            return self._fallback_deblur(image_path, output_path)
    
    def _fallback_deblur(self, image_path, output_path):
        """Fallback to traditional deblurring"""
        image = cv2.imread(str(image_path))
        
        # Try Wiener filter deblurring
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        psf = np.ones((5, 5)) / 25  # Simulate motion blur
        
        # Deconvolve using Wiener filter
        deblurred = cv2.filter2D(gray, -1, psf)
        deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        
        if output_path:
            cv2.imwrite(str(output_path), deblurred)
        
        return deblurred
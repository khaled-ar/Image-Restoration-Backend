"""
Lightweight Super-Resolution Neural Network (2x upscaling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class LightweightSRNet(nn.Module):
    """
    Efficient super-resolution network for 2x upscaling
    Based on ESPCN (Efficient Sub-Pixel CNN) architecture
    """
    
    def __init__(self, scale_factor=2):
        super(LightweightSRNet, self).__init__()
        self.scale = scale_factor
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Sub-pixel convolution for upscaling
        self.subpixel = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Final refinement
        self.refinement = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Extract features
        features = self.feature_extraction(x)
        
        # Sub-pixel convolution
        subpixel_out = self.subpixel(features)
        upscaled = self.pixel_shuffle(subpixel_out)
        
        # Final refinement
        output = self.refinement(upscaled)
        
        return output


class SuperResolutionModel:
    """
    Super-resolution model for 2x image upscaling
    """
    
    def __init__(self, model_path=None, device='cpu', scale_factor=2):
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.model = LightweightSRNet(scale_factor=scale_factor)
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            print("No pretrained SR model found. Using random weights.")
            self.model.to(self.device)
    
    def load_model(self, model_path):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"Super-resolution model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading SR model: {e}")
            self.model.to(self.device)
            self.model.eval()
    
    def upscale_image(self, image_path, output_path=None):
        """
        Upscale image by 2x using neural network
        
        Args:
            image_path: Path to input image
            output_path: Path to save upscaled image
            
        Returns:
            numpy.ndarray: Upscaled image (2x resolution)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            original_shape = image.shape
            print(f"Upscaling image from {original_shape[1]}x{original_shape[0]} to "
                  f"{original_shape[1]*self.scale_factor}x{original_shape[0]*self.scale_factor}")
            
            # Resize if too large for processing
            max_input_dim = 512
            if max(image.shape[:2]) > max_input_dim:
                scale = max_input_dim / max(image.shape[:2])
                new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            # Convert to float32 and normalize
            img_float = image.astype(np.float32) / 255.0
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            # Apply super-resolution
            with torch.no_grad():
                upscaled_tensor = self.model(img_tensor)
            
            # Convert back to numpy
            upscaled_np = upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            upscaled_np = np.clip(upscaled_np * 255, 0, 255).astype(np.uint8)
            
            # Calculate target size based on original
            target_width = original_shape[1] * self.scale_factor
            target_height = original_shape[0] * self.scale_factor
            
            # Resize to exact target size
            if upscaled_np.shape[1] != target_width or upscaled_np.shape[0] != target_height:
                upscaled_np = cv2.resize(upscaled_np, 
                                        (target_width, target_height),
                                        interpolation=cv2.INTER_CUBIC)
            
            # Apply gentle sharpening
            kernel = np.array([[0, -0.25, 0],
                               [-0.25, 2, -0.25],
                               [0, -0.25, 0]])
            upscaled_np = cv2.filter2D(upscaled_np, -1, kernel)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(str(output_path), upscaled_np)
            
            return upscaled_np
            
        except Exception as e:
            print(f"Error in super-resolution: {e}")
            # Fallback to traditional upscaling
            return self._fallback_upscale(image_path, output_path)
    
    def _fallback_upscale(self, image_path, output_path):
        """Traditional upscaling fallback"""
        image = cv2.imread(str(image_path))
        
        # Use Lanczos interpolation for good quality
        upscaled = cv2.resize(image, 
                             (image.shape[1] * self.scale_factor, 
                              image.shape[0] * self.scale_factor),
                             interpolation=cv2.INTER_LANCZOS4)
        
        if output_path:
            cv2.imwrite(str(output_path), upscaled)
        
        return upscaled
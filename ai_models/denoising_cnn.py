"""
Deep Convolutional Neural Network for Image Denoising (DnCNN architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DnCNN(nn.Module):
    """
    Deep Convolutional Neural Network for image denoising
    Based on: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
    """
    
    def __init__(self, in_channels=3, out_channels=3, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        
        # First convolutional layer
        layers = [nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        
        # Middle convolutional layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last convolutional layer
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        # Residual learning: output = input - noise
        noise = self.dncnn(x)
        return x - noise


class DenoisingModel:
    """
    Wrapper for denoising CNN with preprocessing and postprocessing
    """
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model = DnCNN()
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            print("⚠️ No pretrained model found. Using randomly initialized model.")
            self.model.to(self.device)
    
    def load_model(self, model_path):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Denoising model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading denoising model: {e}")
            self.model.to(self.device)
            self.model.eval()
    
    def denoise_image(self, image_path, output_path=None):
        """
        Apply AI-based denoising to an image
        
        Args:
            image_path: Path to input image
            output_path: Path to save denoised image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        try:
            # Read and preprocess image
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
            
            # Apply denoising
            with torch.no_grad():
                denoised_tensor = self.model(img_tensor)
            
            # Convert back to numpy
            denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            denoised_np = np.clip(denoised_np * 255, 0, 255).astype(np.uint8)
            
            # Resize back to original size if needed
            if denoised_np.shape != original_shape[:2]:
                denoised_np = cv2.resize(denoised_np, 
                                        (original_shape[1], original_shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(str(output_path), denoised_np)
            
            return denoised_np
            
        except Exception as e:
            print(f"❌ Error in denoising: {e}")
            # Fallback to OpenCV denoising
            return self._fallback_denoise(image_path, output_path)
    
    def _fallback_denoise(self, image_path, output_path):
        """Fallback to traditional denoising if AI model fails"""
        image = cv2.imread(str(image_path))
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        if output_path:
            cv2.imwrite(str(output_path), denoised)
        
        return denoised
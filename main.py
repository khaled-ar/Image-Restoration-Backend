"""
AI Image Restoration Server - ENHANCED VERSION
Intelligent restoration, colorization, and enhancement for old photos
With improved sharpness and detail preservation
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import uuid
import os
import time
import io
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import warnings
import config

warnings.filterwarnings('ignore')

try:
    import cv2.ximgproc
    GUIDED_FILTER_AVAILABLE = True
except ImportError:
    GUIDED_FILTER_AVAILABLE = False
    print("Guided filter not available, using alternative methods")

# Create directories
Path("storage/original").mkdir(parents=True, exist_ok=True)
Path("storage/enhanced").mkdir(parents=True, exist_ok=True)
Path("pretrained_models").mkdir(exist_ok=True)
Path("storage/samples").mkdir(parents=True, exist_ok=True)
Path("checkpoints").mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced AI Image Restoration API",
    description="Restore, colorize, and enhance old/damaged images using advanced AI techniques",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENHANCED MODELS ====================

class EnhancedRestorationNet(nn.Module):
    """Enhanced CNN for image restoration with attention"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        
        # Attention blocks
        self.attn1 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1)
        )
        
        self.attn2 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1)
        )
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Conv2d(32, out_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        
        # Attention with residual
        a1 = self.attn1(e3) + e3
        a2 = self.attn2(a1) + a1
        
        # Decoder
        d3 = self.relu(self.dec3(a2))
        d2 = self.relu(self.dec2(d3))
        d1 = self.dec1(d2)
        
        return self.sigmoid(d1)

class ColorizationNet(nn.Module):
    """Neural network for realistic black & white image colorization"""
    
    def __init__(self):
        super().__init__()
        
        # Process luminance channel
        self.lum_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Color prediction
        self.color_predictor = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, luminance):
        features = self.lum_encoder(luminance)
        ab_channels = self.color_predictor(features)
        return ab_channels

class SuperResolutionNet(nn.Module):
    """2x Super Resolution with residual learning"""
    
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale = scale_factor
        
        self.initial = nn.Conv2d(3, 64, 9, padding=4)
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(8)]
        )
        
        self.mid = nn.Conv2d(64, 64, 3, padding=1)
        
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, 3, 9, padding=4)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        initial = self.relu(self.initial(x))
        res_out = self.res_blocks(initial)
        mid_out = self.mid(res_out) + initial
        upscaled = self.upscale(mid_out)
        return torch.clamp(upscaled, 0, 1)

class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

# ==================== ENHANCED IMAGE PROCESSOR WITH IMPROVED SHARPNESS ====================

class EnhancedImageProcessor:
    """Advanced image processor with multiple AI enhancements and improved sharpness"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Configuration
        self.config = {
            'max_dimension': 2048,
            'min_dimension': 256,
            'sharpening_strength': 0.3,
            'detail_enhancement': True,
            'quality_threshold': 0.8,
            'interpolation': cv2.INTER_LANCZOS4,
            'use_guided_filter': GUIDED_FILTER_AVAILABLE
        }
        
        # Initialize models with improved loading
        self.restoration_model = self._init_model(EnhancedRestorationNet(), "enhanced_restoration.pth")
        self.colorization_model = self._init_model(ColorizationNet(), "colorization.pth")
        self.sr_model = self._init_model(SuperResolutionNet(scale_factor=2), "super_resolution.pth")
        
        # Traditional enhancement flag
        self.use_traditional_fallback = False
        
        print("All AI models initialized")
    
    def _init_model(self, model, filename):
        """Initialize model with improved error handling and partial weight loading"""
        model_path = Path("pretrained_models") / filename
        
        try:
            if model_path.exists():
                print(f"Loading model: {filename}")
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Try different loading strategies
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print(f"Loaded {filename} (strict mode)")
                except RuntimeError as e:
                    print(f"Strict loading failed: {e}")
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded {filename} (non-strict mode)")
                    except Exception as e2:
                        print(f"Non-strict loading failed: {e2}")
                        # Manual partial loading
                        model_dict = model.state_dict()
                        pretrained_dict = {}
                        
                        for k, v in state_dict.items():
                            if k in model_dict and v.shape == model_dict[k].shape:
                                pretrained_dict[k] = v
                        
                        if pretrained_dict:
                            model_dict.update(pretrained_dict)
                            model.load_state_dict(model_dict)
                            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from {filename}")
                        else:
                            print(f"No compatible weights found in {filename}")
                            self.use_traditional_fallback = True
            else:
                print(f"Model file not found: {filename}")
                print(f"Using initialized weights")
                self.use_traditional_fallback = True
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            print(f"Using traditional enhancement methods")
            self.use_traditional_fallback = True
        
        model.to(self.device)
        model.eval()
        return model
    
    def _validate_enhancement(self, original: np.ndarray, enhanced: np.ndarray) -> bool:
        """Validate that enhancement improved image quality"""
        try:
            # Calculate sharpness metrics
            orig_sharpness = self._calculate_sharpness(original)
            enh_sharpness = self._calculate_sharpness(enhanced)
            
            # Calculate PSNR
            mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
            
            # Quality criteria
            sharpness_improved = enh_sharpness >= orig_sharpness * 0.9
            good_psnr = psnr > 25
            
            return sharpness_improved and good_psnr
            
        except Exception as e:
            print(f"Validation error: {e}")
            return True
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _apply_traditional_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Traditional enhancement fallback with improved sharpness"""
        result = img.copy()
        
        # CLAHE for contrast enhancement
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Adaptive sharpening
        result = self._adaptive_sharpening(result)
        
        # Denoise if needed
        if self._estimate_noise_level(img) > 20:
            result = cv2.fastNlMeansDenoisingColored(result, None, 7, 7, 7, 21)
        
        return result
    
    def _adaptive_sharpening(self, img: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Apply adaptive sharpening based on image content"""
        # Create sharpening kernel
        kernel = np.array([
            [0, -strength, 0],
            [-strength, 1 + 4 * strength, -strength],
            [0, -strength, 0]
        ])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Blend with original based on local contrast
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate local contrast
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        contrast = cv2.absdiff(gray, blur)
        
        # Normalize contrast mask
        contrast_normalized = contrast / 255.0
        
        # Blend based on contrast (more sharpening in high-contrast areas)
        result = np.zeros_like(img, dtype=np.float32)
        
        for c in range(img.shape[2] if len(img.shape) == 3 else 1):
            if len(img.shape) == 3:
                orig_channel = img[:, :, c].astype(np.float32)
                sharp_channel = sharpened[:, :, c].astype(np.float32)
            else:
                orig_channel = img.astype(np.float32)
                sharp_channel = sharpened.astype(np.float32)
            
            # Adaptive blending
            blended = orig_channel * (1 - contrast_normalized * 0.7) + \
                     sharp_channel * (contrast_normalized * 0.7)
            
            if len(img.shape) == 3:
                result[:, :, c] = blended
            else:
                result = blended
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _estimate_noise_level(self, img: np.ndarray) -> float:
        """Estimate noise level in image"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate noise as standard deviation in smooth areas
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        
        return np.std(diff)
    
    def enhance_details(self, img: np.ndarray) -> np.ndarray:
        """Enhance image details while preserving edges"""
        if not self.config['detail_enhancement']:
            return img
        
        try:
            if self.config['use_guided_filter'] and len(img.shape) == 3:
                # Use guided filter for edge-preserving smoothing
                guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                guided = cv2.ximgproc.guidedFilter(
                    guide=guide, src=img, radius=2, eps=0.01
                )
            else:
                # Alternative: bilateral filter
                guided = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Extract details
            details = img.astype(np.float32) - guided.astype(np.float32)
            
            # Enhance details moderately
            enhanced_details = details * 1.3
            
            # Recombine
            result = guided.astype(np.float32) + enhanced_details
            
            # Apply gentle sharpening
            result = self._adaptive_sharpening(result, strength=0.2)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Detail enhancement error: {e}")
            return img
    
    def analyze_image(self, img):
        """Comprehensive image analysis"""
        if img is None:
            return {"error": "Invalid image"}
        
        analysis = {
            "dimensions": f"{img.shape[1]}x{img.shape[0]}",
            "channels": img.shape[2] if len(img.shape) == 3 else 1,
            "dpi_estimate": self._estimate_dpi(img),
            "quality_score": self._calculate_quality_score(img),
            "sharpness_score": round(self._calculate_sharpness(img), 2),
            "noise_level": round(self._estimate_noise_level(img), 2),
            "needs_restoration": self._needs_restoration(img)
        }
        
        # Detect image type
        analysis["is_black_white"] = self._is_truly_black_white(img)
        analysis["color_variance"] = self._calculate_color_variance(img)
        
        return analysis
    
    def _is_truly_black_white(self, img):
        """Advanced black & white detection"""
        if len(img.shape) == 2:
            return True
        
        # Method 1: Check if all channels are identical
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            diff = np.mean(np.abs(b - g)) + np.mean(np.abs(b - r)) + np.mean(np.abs(g - r))
            if diff < 5:
                return True
        
        # Method 2: Check saturation in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        if np.mean(saturation) < 8:
            return True
        
        # Method 3: Check histogram similarity
        if img.shape[2] == 3:
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            corr_bg = np.corrcoef(hist_b.flatten(), hist_g.flatten())[0, 1]
            corr_br = np.corrcoef(hist_b.flatten(), hist_r.flatten())[0, 1]
            
            if corr_bg > 0.98 and corr_br > 0.98:
                return True
        
        return False
    
    def _estimate_dpi(self, img):
        """Estimate image DPI based on content"""
        h, w = img.shape[:2]
        
        # Analyze edges to estimate quality
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (h * w)
        
        if edge_density > 0.1 and max(h, w) > 1000:
            return "High (300+ DPI)"
        elif edge_density > 0.05:
            return "Medium (150-300 DPI)"
        else:
            return "Low (<150 DPI)"
    
    def _calculate_quality_score(self, img):
        """Calculate image quality score (0-100)"""
        try:
            if len(img.shape) == 2:
                gray = img
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast (standard deviation)
            contrast = gray.std()
            
            # Noise level (high frequency content)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            high_freq = np.mean(magnitude_spectrum[:10, :10])
            
            # Combine scores
            sharpness_score = min(sharpness / 100, 1) * 40
            contrast_score = min(contrast / 50, 1) * 40
            noise_score = max(0, 1 - high_freq / 100) * 20
            
            total_score = sharpness_score + contrast_score + noise_score
            
            return int(total_score)
        except:
            return 50
    
    def _needs_restoration(self, img):
        """Determine if image needs restoration"""
        quality = self._calculate_quality_score(img)
        return quality < 70
    
    def _calculate_color_variance(self, img):
        """Calculate color variance"""
        if len(img.shape) == 2:
            return 0
        
        b, g, r = cv2.split(img)
        return np.var([b.std(), g.std(), r.std()])
    
    def remove_noise_and_artifacts(self, img, strength='medium'):
        """Advanced noise and artifact removal"""
        if strength == 'strong':
            h = 15
            h_color = 15
        elif strength == 'medium':
            h = 10
            h_color = 10
        else:
            h = 7
            h_color = 7
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None,
            h=h,
            hColor=h_color,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Apply bilateral filter for edge preservation
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Remove JPEG artifacts
        _, buffer = cv2.imencode('.jpg', denoised, [cv2.IMWRITE_JPEG_QUALITY, 95])
        denoised = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        return denoised
    
    def apply_ai_restoration(self, img):
        """Apply AI-based restoration"""
        try:
            if self.use_traditional_fallback:
                print("Using traditional restoration (AI model not available)")
                return self._apply_traditional_enhancement(img)
            
            # Ensure proper dimensions
            h, w = img.shape[:2]
            
            # Pad to multiple of 8
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            
            if pad_h > 0 or pad_w > 0:
                img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            else:
                img_padded = img
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Apply restoration
            with torch.no_grad():
                restored_tensor = self.restoration_model(img_tensor)
            
            # Convert back
            restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                restored = restored[:h, :w]
            
            # Validate enhancement
            if not self._validate_enhancement(img, restored):
                print("AI restoration reduced quality, using traditional method")
                restored = self._apply_traditional_enhancement(img)
            
            return restored
            
        except Exception as e:
            print(f"AI restoration failed: {e}")
            return self._apply_traditional_enhancement(img)
    
    def apply_ai_colorization(self, bw_img):
        """Apply AI colorization to black and white images"""
        try:
            if len(bw_img.shape) == 2:
                bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
            
            # Model expects [0, 1] range
            lab = cv2.cvtColor(bw_img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Enhance L channel
            l_channel = cv2.equalizeHist(l_channel)
            
            # Normalize to [0, 1]
            l_tensor = torch.from_numpy(l_channel).float() / 255.0
            l_tensor = l_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                ab_tensor = self.colorization_model(l_tensor)
            
            # Model outputs [-1, 1], convert to LAB range
            ab_tensor = ab_tensor * 128
            ab_channels = ab_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # Merge channels
            lab_colorized = np.zeros((l_channel.shape[0], l_channel.shape[1], 3), dtype=np.float32)
            lab_colorized[:, :, 0] = l_channel
            lab_colorized[:, :, 1:] = ab_channels
            
            colorized = cv2.cvtColor(lab_colorized.astype(np.uint8), cv2.COLOR_LAB2BGR)
            blended = cv2.addWeighted(bw_img, 0.3, colorized, 0.7, 0)
            
            # Color enhancement
            hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
            blended = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return blended
        except Exception as e:
            print(f"AI colorization failed: {e}")
            return self.apply_traditional_colorization(bw_img)
    
    def apply_traditional_colorization(self, bw_img):
        """Traditional colorization as fallback"""
        # Convert to LAB
        lab = cv2.cvtColor(bw_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Create color hints based on luminance
        a_channel = np.ones_like(l_channel) * 128
        b_channel = np.ones_like(l_channel) * 128
        
        # Different color zones based on luminance
        zones = [
            (l_channel > 200, 140, 150),  # Bright: warm
            (l_channel > 150, 135, 140),  # Light: warm
            (l_channel > 100, 130, 135),  # Medium: natural
            (l_channel > 50, 125, 130),   # Dark: cool
            (True, 120, 125)              # Very dark: cooler
        ]
        
        for mask, a_val, b_val in zones:
            a_channel[mask] = a_val
            b_channel[mask] = b_val
        
        # Merge and convert
        lab_colorized = cv2.merge([l_channel, a_channel, b_channel])
        colorized = cv2.cvtColor(lab_colorized, cv2.COLOR_LAB2BGR)
        
        return cv2.addWeighted(bw_img, 0.4, colorized, 0.6, 0)
    
    def apply_super_resolution(self, img, scale_factor=2):
        """Apply AI super resolution with color preservation"""
        try:
            # Use LAB color space to preserve colors
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Upscale luminance channel with AI
            l_tensor = torch.from_numpy(l).float() / 255.0
            l_tensor = l_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                l_upscaled_tensor = self.sr_model(l_tensor)
            
            l_upscaled = l_upscaled_tensor.squeeze().cpu().numpy()
            l_upscaled = np.clip(l_upscaled * 255, 0, 255).astype(np.uint8)
            
            # Upscale chrominance channels with traditional method
            h, w = img.shape[:2]
            a_upscaled = cv2.resize(a, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
            b_upscaled = cv2.resize(b, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
            
            # Merge channels
            lab_upscaled = cv2.merge([l_upscaled, a_upscaled, b_upscaled])
            result = cv2.cvtColor(lab_upscaled, cv2.COLOR_LAB2BGR)
            
            return result
        except:
            # Fallback
            h, w = img.shape[:2]
            return cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)
    
    def enhance_quality(self, img):
        """Comprehensive quality enhancement"""
        # Convert to LAB for better color processing
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Moderate color saturation enhancement
        a = cv2.addWeighted(a, 1.15, np.zeros_like(a), 0, -15)
        b = cv2.addWeighted(b, 1.15, np.zeros_like(b), 0, -15)
        
        # Merge
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply adaptive sharpening
        enhanced = self._adaptive_sharpening(enhanced, strength=0.2)
        
        # Blend for natural look
        result = cv2.addWeighted(img, 0.3, enhanced, 0.7, 0)
        
        return result
    
    def process_image(self, input_path, output_path, options=None):
        """Complete image processing pipeline with improved sharpness"""
        start_time = time.time()
        
        if options is None:
            options = {
                'colorize': True,
                'enhance': True,
                'upscale': 1,
                'strength': 'medium',
                'enhance_details': True
            }
        
        try:
            print(f"Processing: {input_path}")
            
            # 1. Read image
            img = cv2.imread(input_path)
            if img is None:
                return False, "Cannot read image", {}
            
            original_h, original_w = img.shape[:2]
            print(f"Original: {original_w}x{original_h}")
            
            # 2. Resize if needed
            max_dim = self.config['max_dimension']
            if max(img.shape[:2]) > max_dim:
                print("Resizing large image...")
                scale = max_dim / max(img.shape[:2])
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=self.config['interpolation'])
                print(f"Resized to: {new_w}x{new_h}")
            
            # 3. Analyze image
            analysis = self.analyze_image(img)
            print(f"Analysis: {analysis['dimensions']}, Quality: {analysis['quality_score']}/100, "
                  f"Type: {'B/W' if analysis['is_black_white'] else 'Color'}")
            
            # 4. Remove noise and artifacts
            print("Step 1: Removing noise and artifacts...")
            denoised = self.remove_noise_and_artifacts(img, options['strength'])
            
            # 5. Apply AI restoration if needed
            if analysis['needs_restoration']:
                print("Step 2: Applying AI restoration...")
                restored = self.apply_ai_restoration(denoised)
            else:
                restored = denoised
            
            # 6. Colorize black & white images
            if analysis['is_black_white'] and options['colorize']:
                print("Step 3: Colorizing black & white image...")
                colorized = self.apply_ai_colorization(restored)
            else:
                colorized = restored
            
            # 7. Enhance quality
            if options['enhance']:
                print("Step 4: Enhancing quality...")
                enhanced = self.enhance_quality(colorized)
            else:
                enhanced = colorized
            
            # 8. Apply super resolution
            final_image = enhanced
            if options['upscale'] > 1:
                print(f"Step 5: Upscaling {options['upscale']}x...")
                final_image = self.apply_super_resolution(enhanced, options['upscale'])
            
            # 9. Enhance details if enabled
            if options.get('enhance_details', True):
                print("Step 6: Enhancing details...")
                final_image = self.enhance_details(final_image)
            
            # 10. Final sharpening
            print("Step 7: Final adjustments...")
            final_image = self._adaptive_sharpening(final_image, strength=0.15)
            
            # 11. Ensure minimum quality
            if final_image.shape[0] < 300 or final_image.shape[1] < 300:
                scale = max(300 / final_image.shape[0], 300 / final_image.shape[1])
                new_size = (int(final_image.shape[1] * scale), int(final_image.shape[0] * scale))
                final_image = cv2.resize(final_image, new_size, interpolation=cv2.INTER_CUBIC)
            
            # 12. Save result
            print(f"Saving to: {output_path}")
            success = cv2.imwrite(output_path, final_image, [
                cv2.IMWRITE_JPEG_QUALITY, 95,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            
            if not success:
                return False, "Failed to save image", {}
            
            # Verify saved image
            saved_img = cv2.imread(output_path)
            if saved_img is None:
                return False, "Saved file is not a valid image", {}
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Calculate improvements
            improvements = self._calculate_improvements(img, final_image)
            
            metrics = {
                **analysis,
                **improvements,
                "processing_time": round(processing_time, 2),
                "final_dimensions": f"{final_image.shape[1]}x{final_image.shape[0]}",
                "upscale_factor": options['upscale'],
                "colorized": analysis['is_black_white'] and options['colorize'],
                "method_used": "AI" if not self.use_traditional_fallback else "Traditional",
                "detail_enhancement": options.get('enhance_details', True)
            }
            
            print(f"Processing complete in {processing_time:.2f}s")
            print(f"Improvements: Sharpness +{improvements.get('sharpness_improvement', 0)}%, "
                  f"Contrast +{improvements.get('contrast_improvement', 0)}%")
            print(f"Method: {metrics['method_used']}")
            
            return True, output_path, metrics
            
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e), {}
    
    def _calculate_improvements(self, original, enhanced):
        """Calculate quantitative improvements"""
        try:
            # Convert to grayscale for some metrics
            if len(original.shape) == 3:
                gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                gray_orig = original
                gray_enh = enhanced
            
            # Sharpness improvement
            sharpness_orig = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
            sharpness_enh = cv2.Laplacian(gray_enh, cv2.CV_64F).var()
            sharpness_improvement = ((sharpness_enh - sharpness_orig) / sharpness_orig * 100 
                                    if sharpness_orig > 0 else 0)
            
            # Contrast improvement
            contrast_orig = gray_orig.std()
            contrast_enh = gray_enh.std()
            contrast_improvement = ((contrast_enh - contrast_orig) / contrast_orig * 100 
                                  if contrast_orig > 0 else 0)
            
            # Colorfulness improvement (for color images)
            if len(original.shape) == 3 and len(enhanced.shape) == 3:
                def colorfulness(image):
                    b, g, r = cv2.split(image.astype(np.float32))
                    rg = np.abs(r - g)
                    yb = np.abs(0.5 * (r + g) - b)
                    std_rg = np.std(rg)
                    std_yb = np.std(yb)
                    mean_rg = np.mean(rg)
                    mean_yb = np.mean(yb)
                    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
                
                color_orig = colorfulness(original)
                color_enh = colorfulness(enhanced)
                color_improvement = ((color_enh - color_orig) / color_orig * 100 
                                   if color_orig > 0 else 0)
            else:
                color_improvement = 0
            
            # PSNR calculation
            mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
            
            return {
                "sharpness_improvement": round(max(sharpness_improvement, 0), 1),
                "contrast_improvement": round(max(contrast_improvement, 0), 1),
                "colorfulness_improvement": round(max(color_improvement, 0), 1),
                "psnr": round(psnr, 2),
                "original_sharpness": round(sharpness_orig, 2),
                "enhanced_sharpness": round(sharpness_enh, 2)
            }
            
        except Exception as e:
            print(f"Error calculating improvements: {e}")
            return {}

# Initialize processor
processor = EnhancedImageProcessor()

# ==================== HELPER FUNCTIONS ====================

def get_recommendations(analysis):
    """Get processing recommendations based on analysis"""
    recs = []
    
    if analysis.get('is_black_white', False):
        recs.append("Enable colorization for realistic colors")
    
    if analysis.get('quality_score', 0) < 60:
        recs.append("Enable restoration for noise/artifact removal")
    
    if analysis.get('dpi_estimate', '').startswith('Low'):
        recs.append("Enable upscaling (2x) for better resolution")
    
    if analysis.get('quality_score', 0) < 70:
        recs.append("Enable quality enhancement")
    
    if analysis.get('sharpness_score', 0) < 50:
        recs.append("Enable detail enhancement for better sharpness")
    
    return recs

def create_comparison(original_path, enhanced_path, output_path):
    """Create side-by-side comparison image"""
    try:
        original = cv2.imread(original_path)
        enhanced = cv2.imread(enhanced_path)
        
        if original is None or enhanced is None:
            return False
        
        # Resize to same height
        height = 600
        orig_width = int(height * (original.shape[1] / original.shape[0]))
        enh_width = int(height * (enhanced.shape[1] / enhanced.shape[0]))
        
        orig_resized = cv2.resize(original, (orig_width, height), interpolation=cv2.INTER_LANCZOS4)
        enh_resized = cv2.resize(enhanced, (enh_width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create comparison
        comparison = np.hstack([orig_resized, enh_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add separator
        cv2.line(comparison, (orig_width, 0), (orig_width, height), (255, 255, 255), 3)
        
        # Save
        success = cv2.imwrite(output_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return success
        
    except Exception as e:
        print(f"Comparison creation failed: {e}")
        return False

def cleanup_directory(directory, max_files):
    """Clean up old files"""
    try:
        if not os.path.exists(directory):
            return
        
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        files = [f for f in files if os.path.isfile(f)]
        
        if len(files) > max_files:
            files.sort(key=os.path.getmtime)
            files_to_delete = files[:len(files) - max_files]
            
            deleted_count = 0
            for f in files_to_delete:
                try:
                    os.remove(f)
                    deleted_count += 1
                except:
                    pass
            
            if deleted_count > 0:
                print(f"Cleaned {deleted_count} files from {directory}")
                
    except Exception as e:
        print(f"Cleanup error: {e}")

# ==================== API ENDPOINTS ====================

@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image to restore"),
):
    """Upload and process image with advanced AI"""
    try:
        print(f"Received: {file.filename}")
        print(f"Options: colorize=false, enhance=false, upscale=4x")
        
        # Validate
        allowed_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_ext):
            raise HTTPException(400, "Only image files are allowed")
        
        # Generate ID
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1] or ".jpg" 
        
        # Save original
        original_path = f"storage/original/{file_id}{file_ext}"
        with open(original_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"Saved: {original_path}")
        
        # Verify file
        test_img = cv2.imread(original_path)
        if test_img is None:
            os.remove(original_path)
            raise HTTPException(400, "Uploaded file is not a valid image")
        
        # Process options
        options = {
            'colorize': False,
            'enhance': False,
            'upscale': 4,
            'strength': "high",
            'enhance_details': True
        }
        
        # Process image
        enhanced_path = f"storage/enhanced/{file_id}_enhanced.jpg"
        success, result, metrics = processor.process_image(
            original_path, enhanced_path, options
        )
        
        if not success:
            # Clean up
            for path in [original_path, enhanced_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise HTTPException(500, f"Processing failed: {result}")
        
        # Verify enhanced file
        enhanced_img = cv2.imread(enhanced_path)
        if enhanced_img is None:
            for path in [original_path, enhanced_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise HTTPException(500, "Enhanced image creation failed")
        
        # Create comparison
        comparison_path = f"storage/samples/comparison_{file_id}.jpg"
        comparison_created = create_comparison(original_path, enhanced_path, comparison_path)
        
        if comparison_created:
            print(f"Comparison: {comparison_path}")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_directory, "storage/original", 100)
        background_tasks.add_task(cleanup_directory, "storage/enhanced", 100)
        background_tasks.add_task(cleanup_directory, "storage/samples", 50)
        
        base_url = config.API_CONFIG["base_url"]
        
        # Response
        response = {
            "success": True,
            "message": "Image processed successfully",
            "image_id": file_id,
            "comparison_url": f"{base_url}/compare/{file_id}",
            "download_url": f"{base_url}/download/{file_id}",
            "metrics": metrics
        }
        
        return JSONResponse(content=response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Server error: {str(e)}")

@app.get("/download/{image_id}")
async def download_image(image_id: str):
    """Download enhanced image"""
    enhanced_path = f"storage/enhanced/{image_id}_enhanced.jpg"
    
    if os.path.exists(enhanced_path):
        test_img = cv2.imread(enhanced_path)
        if test_img is not None:
            return FileResponse(
                enhanced_path,
                media_type="image/jpeg",
                filename=f"enhanced_{image_id}.jpg"
            )
    
    raise HTTPException(404, "Image not found")

@app.get("/compare/{image_id}")
async def compare_images(image_id: str):
    """Get comparison image"""
    comparison_path = f"storage/samples/comparison_{image_id}.jpg"
    
    if os.path.exists(comparison_path):
        test_img = cv2.imread(comparison_path)
        if test_img is not None:
            return FileResponse(
                comparison_path,
                media_type="image/jpeg",
                filename=f"comparison_{image_id}.jpg"
            )
    
    raise HTTPException(404, "Comparison not found")

# ==================== START SERVER ====================

def start_server():
    """Start the server"""
    print("API Docs: http://127.0.0.1:8000/docs")
    print("Server started on http://127.0.0.1:8000")
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    start_server()
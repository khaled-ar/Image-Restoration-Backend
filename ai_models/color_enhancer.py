"""
Intelligent Color Enhancement Neural Network
Preserves original colors while enhancing realism and quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage import color

class ColorEnhancementNet(nn.Module):
    """
    CNN for intelligent color enhancement and correction
    - Preserves original colors
    - Realistic colorization for B/W images
    - Gentle color enhancement for colored images
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super(ColorEnhancementNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature processing
        self.features = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with color attention
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Color preservation gate
        self.color_gate = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Process features
        features = self.features(encoded)
        
        # Generate color preservation mask
        color_mask = self.color_gate(features)
        
        # Decode
        color_adjustment = self.decoder(features)
        
        # Apply adjustment with preservation mask
        # Original colors are preserved where mask is high
        output = x + color_adjustment * (1 - color_mask)
        
        return output


class ColorEnhancer:
    """
    Intelligent color enhancement with B/W detection and realistic colorization
    """
    
    def __init__(self, model_path=None, device='cpu', realistic_mode=True):
        self.device = torch.device(device)
        self.realistic_mode = realistic_mode
        self.model = ColorEnhancementNet()
        
        # إضافة هذا لتتبع حالة النموذج
        self.model_loaded = False
        self.model_uses_tanh = True  # النموذج الأصلي يستخدم Tanh
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            print("⚠️ No pretrained color enhancer found. Using random weights.")
            self.model.to(self.device)
    
    def load_model(self, model_path):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Color enhancer model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading color enhancer: {e}")
            self.model.to(self.device)
            self.model.eval()
    
    def is_black_white(self, image):
        """Detect if image is black and white using advanced analysis"""
        if len(image.shape) == 2:
            return True
        
        # Method 1: Check channel similarity
        b, g, r = cv2.split(image)
        diff_bg = np.mean(np.abs(b - g))
        diff_br = np.mean(np.abs(b - r))
        diff_gr = np.mean(np.abs(g - r))
        
        if diff_bg < 5 and diff_br < 5 and diff_gr < 5:
            return True
        
        # Method 2: Check color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        
        if avg_saturation < 15:  # Very low saturation
            return True
        
        # Method 3: Check histogram
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        correlation_bg = np.corrcoef(hist_b.flatten(), hist_g.flatten())[0, 1]
        correlation_br = np.corrcoef(hist_b.flatten(), hist_r.flatten())[0, 1]
        
        if correlation_bg > 0.98 and correlation_br > 0.98:
            return True
        
        return False
    
    def enhance_colors(self, image_path, output_path=None, preserve_original=True):
        """
        Enhance image colors intelligently
        - Preserves original colors for colored images
        - Realistic colorization for B/W images
        - Gentle enhancement for all images
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            original_shape = image.shape
            is_bw = self.is_black_white(image)
            
            print(f"🎨 Image analysis: {'Black-White' if is_bw else 'Color'} image detected")
            
            # Resize if too large
            max_dim = 1024
            if max(image.shape[:2]) > max_dim:
                scale = max_dim / max(image.shape[:2])
                new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            if is_bw and self.realistic_mode:
                print("🖤 Applying realistic colorization to B/W image...")
                enhanced = self._realistic_colorization(image)
            else:
                print("🌈 Enhancing colors while preserving original...")
                enhanced = self._enhance_with_ai(image, preserve_original)
            
            # Resize back to original size if needed
            if enhanced.shape != original_shape[:2]:
                enhanced = cv2.resize(enhanced, 
                                    (original_shape[1], original_shape[0]),
                                    interpolation=cv2.INTER_CUBIC)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(str(output_path), enhanced)
            
            return enhanced
            
        except Exception as e:
            print(f"❌ Error in color enhancement: {e}")
            # Fallback to traditional enhancement
            return self._fallback_enhance(image_path, output_path, is_bw)
    
    def _enhance_with_ai(self, image, preserve_original=True):
        """Enhance colors using neural network - FIXED VERSION"""
        
        # 1. تحميل النموذج أولاً لمعرفة نوعه
        model_type = self._get_model_type()  # تحتاج لإنشاء هذه الدالة
        
        # 2. تطبيع المدخلات حسب نوع النموذج
        if model_type == "sigmoid":
            # النماذج المدرَّسة (Sigmoid) تتوقع [0, 1]
            img_float = image.astype(np.float32) / 255.0
        else:  # tanh أو unknown
            # النماذج الأصلية (Tanh) تتوقع [-1, 1]
            img_float = image.astype(np.float32) / 127.5 - 1.0
        
        # 3. التحويل إلى Tensor
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)
        
        # 4. تطبيق النموذج
        with torch.no_grad():
            enhanced_tensor = self.model(img_tensor)
        
        # 5. تحويل المخرجات حسب نوع النموذج
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        if model_type == "sigmoid":
            # تحويل [0, 1] → [0, 255]
            enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
        else:  # tanh
            # تحويل [-1, 1] → [0, 255]
            enhanced_np = np.clip((enhanced_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        # 6. الدمج مع الأصل
        if preserve_original:
            alpha = 0.3
            enhanced_np = cv2.addWeighted(image, 1 - alpha, enhanced_np, alpha, 0)
        
        return enhanced_np

    def _get_model_type(self):
        """Detect if model uses Tanh or Sigmoid"""
        if not hasattr(self, '_model_type_detected'):
            # اختبار بسيط
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
            
            output_min = test_output.min().item()
            output_max = test_output.max().item()
            
            if output_min >= -1.1 and output_max <= 1.1:
                self._model_type_detected = "tanh"
                print(f"   🔍 Model uses TANH (output: [{output_min:.3f}, {output_max:.3f}])")
            elif output_min >= 0 and output_max <= 1:
                self._model_type_detected = "sigmoid"
                print(f"   🔍 Model uses SIGMOID (output: [{output_min:.3f}, {output_max:.3f}])")
            else:
                self._model_type_detected = "unknown"
                print(f"   ⚠️ Unknown output range: [{output_min:.3f}, {output_max:.3f}]")
        
        return self._model_type_detected
    
    def _realistic_colorization(self, image):
        """
        Apply realistic colorization to B/W images
        Uses color hints from similar regions
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply mild colorization
        # Create color hints based on luminance
        color_hints = np.zeros_like(lab)
        color_hints[:, :, 0] = l_channel
        
        # Generate realistic A and B channels based on luminance
        # Darker areas -> cooler colors (blues)
        # Lighter areas -> warmer colors (yellows, reds)
        normalized_l = l_channel / 255.0
        
        # A channel (green-red)
        color_hints[:, :, 1] = 128 + (normalized_l - 0.5) * 50
        
        # B channel (blue-yellow)
        color_hints[:, :, 2] = 128 + (0.5 - normalized_l) * 30
        
        # Convert back to BGR
        colorized = cv2.cvtColor(color_hints.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Blend with original for subtle effect
        colorized = cv2.addWeighted(image, 0.7, colorized, 0.3, 0)
        
        return colorized
    
    def _fallback_enhance(self, image_path, output_path, is_bw):
        """Traditional color enhancement fallback"""
        image = cv2.imread(str(image_path))
        
        if is_bw:
            # Simple colorization for B/W
            colorized = self._realistic_colorization(image)
        else:
            # Color enhancement using CLAHE in LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l, a, b])
            colorized = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        if output_path:
            cv2.imwrite(str(output_path), colorized)
        
        return colorized
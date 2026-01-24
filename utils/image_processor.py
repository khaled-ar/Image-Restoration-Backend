"""
Enhanced Image Processor with AI capabilities
"""

import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import config

class AIImageProcessor:
    """Advanced image processor with AI capabilities"""
    
    def validate_image(self, file):
        """Validate uploaded image file with AI analysis"""
        try:
            # Save temporarily for analysis
            temp_path = Path("temp_validation.jpg")
            with open(temp_path, "wb") as f:
                f.write(file.file.read())
            
            # Analyze image
            image = cv2.imread(str(temp_path))
            if image is None:
                os.remove(temp_path)
                return False, "Cannot read image", {}
            
            # Basic validation
            allowed_extensions = config.API_CONFIG["allowed_extensions"]
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext not in allowed_extensions:
                os.remove(temp_path)
                return False, f"File type '{file_ext}' not allowed", {}
            
            # File size validation
            max_size = config.API_CONFIG["max_file_size"]
            file_size = os.path.getsize(temp_path)
            
            if file_size > max_size:
                os.remove(temp_path)
                size_mb = max_size / (1024 * 1024)
                return False, f"File too large. Max: {size_mb}MB", {}
            
            # AI-based analysis
            metadata = self._analyze_image_ai(image)
            
            # Clean up
            os.remove(temp_path)
            
            return True, "Image is valid", metadata
            
        except Exception as e:
            if Path("temp_validation.jpg").exists():
                os.remove("temp_validation.jpg")
            return False, f"Validation error: {str(e)}", {}
    
    def _analyze_image_ai(self, image):
        """Analyze image using AI and computer vision"""
        try:
            # Basic image info
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            
            # Detect if black and white
            is_bw = False
            if channels == 3:
                b, g, r = cv2.split(image)
                diff = np.mean(np.abs(b - g)) + np.mean(np.abs(b - r)) + np.mean(np.abs(g - r))
                is_bw = diff < 15
            
            # Estimate quality metrics
            blurriness = cv2.Laplacian(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if channels == 3 else image,
                cv2.CV_64F
            ).var()
            
            # Estimate noise level
            noise_level = 0
            if channels == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                noise_level = np.std(gray)
            
            return {
                "dimensions": f"{width}x{height}",
                "channels": channels,
                "is_black_white": is_bw,
                "estimated_blurriness": round(blurriness, 2),
                "estimated_noise_level": round(noise_level, 2),
                "aspect_ratio": round(width / height, 2) if height > 0 else 0,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return {}
    
    def save_image_with_metadata(self, file, directory):
        """Save image with AI-generated metadata"""
        try:
            # Generate unique filename
            file_ext = Path(file.filename).suffix.lower()
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}{file_ext}"
            file_path = Path(directory) / filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            
            # Analyze saved image
            image = cv2.imread(str(file_path))
            metadata = self._analyze_image_ai(image) if image is not None else {}
            
            # Add file info
            metadata.update({
                "original_filename": file.filename,
                "saved_filename": filename,
                "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
                "saved_timestamp": datetime.now().isoformat()
            })
            
            return True, file_path, metadata
            
        except Exception as e:
            return False, None, {"error": str(e)}
    
    def create_ai_comparison(self, original_path, enhanced_path, output_path):
        """Create AI-enhanced comparison image with metrics"""
        try:
            # Read images
            original = cv2.imread(str(original_path))
            enhanced = cv2.imread(str(enhanced_path))
            
            if original is None or enhanced is None:
                return False, "Cannot read one or both images", {}
            
            # Resize to same height
            target_height = max(original.shape[0], enhanced.shape[0])
            
            # Calculate new widths maintaining aspect ratio
            orig_aspect = original.shape[1] / original.shape[0]
            enh_aspect = enhanced.shape[1] / enhanced.shape[0]
            
            orig_width = int(target_height * orig_aspect)
            enh_width = int(target_height * enh_aspect)
            
            # Resize images
            orig_resized = cv2.resize(original, (orig_width, target_height))
            enh_resized = cv2.resize(enhanced, (enh_width, target_height))
            
            # Create comparison image
            comparison = np.hstack((orig_resized, enh_resized))
            
            # Add AI-generated metrics
            metrics = self._calculate_comparison_metrics(original, enhanced)
            
            # Add visual labels with metrics
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 0)
            
            # Original label
            cv2.putText(comparison, 'ORIGINAL', (10, 30), font, font_scale, color, thickness)
            
            # Enhanced label
            cv2.putText(comparison, 'AI ENHANCED', (orig_width + 10, 30), font, font_scale, color, thickness)
            
            # Add metrics text
            metrics_text = [
                f"PSNR: {metrics.get('psnr', 'N/A')} dB",
                f"Sharpness +{metrics.get('sharpness_improvement', 0):.1f}%",
                f"Colors +{metrics.get('colorfulness_improvement', 0):.1f}%"
            ]
            
            y_offset = 60
            for text in metrics_text:
                cv2.putText(comparison, text, (orig_width + 10, y_offset), 
                           font, font_scale * 0.8, (255, 255, 255), 1)
                y_offset += 25
            
            # Add separator line
            cv2.line(comparison, (orig_width, 0), (orig_width, target_height), (255, 255, 255), 2)
            
            # Save comparison
            cv2.imwrite(str(output_path), comparison)
            
            return True, output_path, metrics
            
        except Exception as e:
            return False, f"Comparison error: {str(e)}", {}
    
    def _calculate_comparison_metrics(self, original, enhanced):
        """Calculate AI-based comparison metrics"""
        try:
            # Ensure same dimensions
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # Calculate PSNR
            mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
            
            # Calculate sharpness improvement
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            orig_sharp = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
            enh_sharp = cv2.Laplacian(gray_enh, cv2.CV_64F).var()
            sharp_improvement = ((enh_sharp - orig_sharp) / orig_sharp) * 100 if orig_sharp > 0 else 0
            
            # Calculate colorfulness improvement
            def colorfulness(img):
                b, g, r = cv2.split(img)
                rg = np.abs(r.astype(np.float32) - g.astype(np.float32))
                yb = np.abs(0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32))
                std_rg = np.std(rg)
                std_yb = np.std(yb)
                mean_rg = np.mean(rg)
                mean_yb = np.mean(yb)
                return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
            
            orig_color = colorfulness(original)
            enh_color = colorfulness(enhanced)
            color_improvement = ((enh_color - orig_color) / orig_color) * 100 if orig_color > 0 else 0
            
            return {
                "psnr": round(psnr, 2),
                "sharpness_improvement": round(sharp_improvement, 2),
                "colorfulness_improvement": round(color_improvement, 2),
                "original_sharpness": round(orig_sharp, 2),
                "enhanced_sharpness": round(enh_sharp, 2),
                "original_colorfulness": round(orig_color, 2),
                "enhanced_colorfulness": round(enh_color, 2)
            }
            
        except Exception as e:
            print(f"Error calculating comparison metrics: {e}")
            return {}
"""
Configuration for AI-Powered Image Restoration System - ENHANCED VERSION
"""

import os
from pathlib import Path
import torch

# Base directory
BASE_DIR = Path(__file__).parent

# File storage paths
UPLOAD_DIR = BASE_DIR / "storage" / "original"
ENHANCED_DIR = BASE_DIR / "storage" / "enhanced"
SAMPLES_DIR = BASE_DIR / "storage" / "samples"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "pretrained_models"

# Create directories
for dir_path in [UPLOAD_DIR, ENHANCED_DIR, SAMPLES_DIR, LOGS_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Check GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# AI Models Configuration
AI_MODELS = {
    "restoration": {
        "model": "EnhancedRestorationNet",
        "path": MODELS_DIR / "enhanced_restoration.pth",
        "enabled": True,
        "description": "Enhanced CNN for image restoration with attention"
    },
    "colorization": {
        "model": "ColorizationNet",
        "path": MODELS_DIR / "colorization.pth",
        "enabled": True,
        "description": "Neural network for realistic black & white image colorization"
    },
    "super_resolution": {
        "model": "SuperResolutionNet",
        "path": MODELS_DIR / "super_resolution.pth",
        "enabled": True,
        "scale_factor": 2,
        "description": "2x Super Resolution with residual learning"
    }
}

# Processing Configuration
PROCESSING_CONFIG = {
    "pipeline": ["denoise", "restore", "color_enhance", "super_resolution", "enhance_details"],
    "preserve_original_colors": True,
    "realistic_bw_colorization": True,
    "max_image_dimension": 2048,
    "min_image_dimension": 256,
    "batch_size": 1,
    "output_quality": 95,
    "sharpening_strength": 0.3,
    "detail_enhancement": True,
    "quality_threshold": 0.8,
    "interpolation_method": "LANCZOS4"
}

# API Configuration
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": True,
    "max_file_size": 15 * 1024 * 1024,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    "base_url": "https://image-restoration-backend.onrender.com"
}

# Model loading configuration
MODEL_LOADING = {
    "strict_loading": False,
    "fallback_enabled": True,
    "validate_after_load": True
}
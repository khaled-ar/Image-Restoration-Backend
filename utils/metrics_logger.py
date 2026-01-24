"""
Metrics Logger - Saves training and testing results to Excel files
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import config

class MetricsLogger:
    """Handles logging of training and testing metrics to Excel files"""
    
    def __init__(self):
        """Initialize the MetricsLogger and create Excel files if they don't exist"""
        self.training_file = config.RESULTS_DIR / "training_results.csv"
        self.testing_file = config.RESULTS_DIR / "testing_results.csv"
        
        # Initialize Excel files with headers
        self._initialize_files()
        print("MetricsLogger initialized")
    
    def _initialize_files(self):
        """Create Excel files with proper headers if they don't exist"""
        # Training results file structure
        training_headers = [
            "timestamp", "model_name", "dataset", "epochs", 
            "batch_size", "learning_rate", "loss", "psnr", 
            "ssim", "training_time_minutes", "notes"
        ]
        
        if not os.path.exists(self.training_file):
            df = pd.DataFrame(columns=training_headers)
            df.to_excel(self.training_file, index=False)
            print(f"Created training results file: {self.training_file}")
        
        # Testing results file structure
        testing_headers = [
            "timestamp", "image_id", "original_filename", 
            "original_size_kb", "enhanced_size_kb", 
            "brightness_improvement_percent", "contrast_improvement_percent",
            "sharpness_improvement_percent", "processing_time_seconds",
            "model_used", "mode", "scale_factor",
            "original_width", "original_height",
            "enhanced_width", "enhanced_height", "notes"
        ]
        
        if not os.path.exists(self.testing_file):
            df = pd.DataFrame(columns=testing_headers)
            df.to_excel(self.testing_file, index=False)
            print(f"Created testing results file: {self.testing_file}")
    
    def log_training_result(self, data):
        """Log training results to Excel file"""
        try:
            # Read existing data
            df = pd.read_excel(self.training_file)
            
            # Add timestamp if not provided
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Append new row
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.training_file, index=False)
            print(f"Logged training result: {data.get('model_name', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"Error logging training result: {e}")
            return False
    
    def log_testing_result(self, image_id, metrics, original_info, enhanced_info, user_settings=None):
        """Log testing results for a single image with user settings"""
        try:
            # Read existing data
            df = pd.read_excel(self.testing_file)
            
            # Prepare data row
            data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_id": image_id,
                "original_filename": original_info.get("filename", "unknown"),
                "original_size_kb": original_info.get("size_kb", 0),
                "enhanced_size_kb": enhanced_info.get("size_kb", 0),
                "brightness_improvement_percent": metrics.get("brightness_improvement_percent", 0),
                "contrast_improvement_percent": metrics.get("contrast_improvement_percent", 0),
                "sharpness_improvement_percent": metrics.get("sharpness_improvement_percent", 0),
                "colorfulness_improvement_percent": metrics.get("colorfulness_improvement_percent", 0),
                "processing_time_seconds": metrics.get("processing_time", 0),
                "model_used": "EnhancedRestorationNet",
                "enhancement_mode": user_settings.get("enhancement_mode", "balanced") if user_settings else "balanced",
                "preserve_colors": user_settings.get("preserve_colors", True) if user_settings else True,
                "scale_factor": user_settings.get("scale_factor", 2) if user_settings else 2,
                "image_type": metrics.get("image_type", "unknown"),
                "original_width": original_info.get("width", 0),
                "original_height": original_info.get("height", 0),
                "enhanced_width": enhanced_info.get("width", 0),
                "enhanced_height": enhanced_info.get("height", 0),
                "notes": f"Processed with EnhancedRestorationNet"
            }
            
            # Append new row
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.testing_file, index=False)
            print(f"Logged testing result for image: {image_id}")
            return True
            
        except Exception as e:
            print(f"Error logging testing result: {e}")
            return False
        
    def get_training_statistics(self):
        """Get training statistics from logged data"""
        try:
            df = pd.read_excel(self.training_file)
            if df.empty:
                return {"message": "No training data available"}
            
            stats = {
                "total_training_runs": len(df),
                "models_used": df["model_name"].unique().tolist(),
                "total_training_time_hours": round(df["training_time_minutes"].sum() / 60, 2),
                "average_training_time_minutes": round(df["training_time_minutes"].mean(), 2)
            }
            
            # Add PSNR and SSIM stats if available
            if "psnr" in df.columns:
                stats["best_psnr"] = round(df["psnr"].max(), 2)
                stats["average_psnr"] = round(df["psnr"].mean(), 2)
            
            if "ssim" in df.columns:
                stats["best_ssim"] = round(df["ssim"].max(), 4)
                stats["average_ssim"] = round(df["ssim"].mean(), 4)
            
            return stats
            
        except Exception as e:
            print(f"Error getting training stats: {e}")
            return {"error": str(e)}
    
    def get_testing_statistics(self):
        """Get testing statistics from logged data"""
        try:
            df = pd.read_excel(self.testing_file)
            if df.empty:
                return {
                    "total_images_processed": 0,
                    "message": "No testing data available"
                }
            
            stats = {
                "total_images_processed": len(df),
                "average_brightness_improvement": f"{round(df['brightness_improvement_percent'].mean(), 2)}%",
                "average_contrast_improvement": f"{round(df['contrast_improvement_percent'].mean(), 2)}%",
                "average_sharpness_improvement": f"{round(df['sharpness_improvement_percent'].mean(), 2)}%",
                "average_processing_time_seconds": round(df["processing_time_seconds"].mean(), 2),
                "total_processing_time_hours": round(df["processing_time_seconds"].sum() / 3600, 3),
                "models_used": df["model_used"].unique().tolist()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting testing stats: {e}")
            return {"error": str(e)}
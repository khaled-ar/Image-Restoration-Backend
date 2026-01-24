"""
File Handler - Manages file operations and storage
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import config

class FileHandler:
    """Handles file operations including storage management and cleanup"""
    
    @staticmethod
    def save_sample_images(original_path, enhanced_path):
        """Save sample images for demonstration purposes"""
        try:
            # Create sample directories
            original_samples_dir = config.SAMPLES_DIR / "original"
            enhanced_samples_dir = config.SAMPLES_DIR / "enhanced"
            os.makedirs(original_samples_dir, exist_ok=True)
            os.makedirs(enhanced_samples_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create sample filenames
            original_sample_name = f"sample_{timestamp}_original.jpg"
            enhanced_sample_name = f"sample_{timestamp}_enhanced.jpg"
            
            original_sample_path = original_samples_dir / original_sample_name
            enhanced_sample_path = enhanced_samples_dir / enhanced_sample_name
            
            # Copy files to samples directory
            shutil.copy2(original_path, original_sample_path)
            shutil.copy2(enhanced_path, enhanced_sample_path)
            
            print(f"Saved sample images: {original_sample_name}, {enhanced_sample_name}")
            
            return {
                "original_sample": str(original_sample_path),
                "enhanced_sample": str(enhanced_sample_path),
                "timestamp": timestamp
            }
            
        except Exception as e:
            print(f"Error saving sample images: {e}")
            return {}
    
    @staticmethod
    def cleanup_old_files(directory, max_files=100, max_age_days=30):
        """Clean up old files to prevent storage overflow"""
        try:
            if not os.path.exists(directory):
                return True, f"Directory does not exist: {directory}"
            
            # Get all files in directory
            files = [os.path.join(directory, f) for f in os.listdir(directory)]
            files = [f for f in files if os.path.isfile(f)]
            
            if not files:
                return True, "No files to clean"
            
            # Sort files by modification time (oldest first)
            files.sort(key=os.path.getmtime)
            
            deleted_count = 0
            current_time = datetime.now()
            
            # Delete files based on age
            for file_path in files:
                file_age = current_time - datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_age > timedelta(days=max_age_days):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"Deleted old file (age: {file_age.days} days): {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Could not delete file {file_path}: {e}")
            
            # Delete excess files if still over limit
            files_remaining = [f for f in files if os.path.exists(f)]
            if len(files_remaining) > max_files:
                files_remaining.sort(key=os.path.getmtime)
                excess = len(files_remaining) - max_files
                
                for i in range(excess):
                    try:
                        os.remove(files_remaining[i])
                        deleted_count += 1
                        print(f"Deleted excess file: {os.path.basename(files_remaining[i])}")
                    except Exception as e:
                        print(f"Could not delete file: {e}")
            
            if deleted_count > 0:
                return True, f"Cleaned up {deleted_count} old files from {directory}"
            else:
                return True, f"No cleanup needed in {directory}"
            
        except Exception as e:
            return False, f"Error during cleanup: {str(e)}"
    
    @staticmethod
    def get_storage_statistics():
        """Get storage usage statistics"""
        try:
            stats = {}
            directories = {
                "original_uploads": config.UPLOAD_DIR,
                "enhanced_images": config.ENHANCED_DIR,
                "logs": config.LOGS_DIR,
                "results": config.RESULTS_DIR
            }
            
            total_files = 0
            total_size_mb = 0
            
            for name, directory in directories.items():
                if os.path.exists(directory):
                    # Count files
                    files = [f for f in os.listdir(directory) 
                            if os.path.isfile(os.path.join(directory, f))]
                    
                    # Calculate total size
                    size_bytes = sum(
                        os.path.getsize(os.path.join(directory, f)) 
                        for f in files
                    )
                    
                    size_mb = size_bytes / (1024 * 1024)
                    
                    stats[name] = {
                        "file_count": len(files),
                        "size_mb": round(size_mb, 2),
                        "directory": str(directory)
                    }
                    
                    total_files += len(files)
                    total_size_mb += size_mb
                else:
                    stats[name] = {
                        "file_count": 0,
                        "size_mb": 0,
                        "directory": str(directory),
                        "status": "directory_not_found"
                    }
            
            # Add totals
            stats["total"] = {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting storage stats: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def backup_results():
        """Create backup of results files"""
        try:
            backup_dir = config.RESULTS_DIR / "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup Excel files
            excel_files = ["training_results.csv", "testing_results.csv"]
            
            for file_name in excel_files:
                source = config.RESULTS_DIR / file_name
                if os.path.exists(source):
                    backup_name = f"{file_name.replace('.csv', '')}_{timestamp}.csv"
                    backup_path = backup_dir / backup_name
                    shutil.copy2(source, backup_path)
            
            return True, f"Results backed up to {backup_dir}"
            
        except Exception as e:
            return False, f"Backup failed: {str(e)}"
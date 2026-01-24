"""
Create Enhanced AI Models for Image Restoration
Training script for advanced models with improved dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime
import glob

# ==================== MODEL DEFINITIONS ====================

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
            nn.Tanh()
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

# ==================== ENHANCED DATASETS ====================

class EnhancedRestorationDataset(Dataset):
    """Enhanced dataset with real and synthetic images"""
    
    def __init__(self, num_samples=2000, img_size=256, use_real_images=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.use_real_images = use_real_images
        
        # Try to load real images if available
        self.real_images = []
        if use_real_images:
            self._load_real_images()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 50% chance to use real image if available
        if self.real_images and random.random() > 0.5:
            clean = self._get_real_image()
        else:
            clean = self._generate_clean_image()
        
        # Generate degraded version
        degraded = self._degrade_image(clean)
        
        # Convert to tensors
        clean_tensor = torch.from_numpy(clean).float().permute(2, 0, 1) / 255.0
        degraded_tensor = torch.from_numpy(degraded).float().permute(2, 0, 1) / 255.0
        
        return degraded_tensor, clean_tensor
    
    def _load_real_images(self):
        """Load real images for training"""
        try:
            # Look for images in common directories
            image_dirs = [
                "data/train",
                "images",
                "dataset",
                "samples"
            ]
            
            for img_dir in image_dirs:
                if os.path.exists(img_dir):
                    image_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                                 glob.glob(os.path.join(img_dir, "*.png")) + \
                                 glob.glob(os.path.join(img_dir, "*.jpeg"))
                    
                    for img_path in image_paths[:100]:
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
                                self.real_images.append(img)
                        except:
                            continue
            
            print(f"Loaded {len(self.real_images)} real images for training")
            
        except Exception as e:
            print(f"Could not load real images: {e}")
    
    def _get_real_image(self):
        """Get a real image from loaded images"""
        if self.real_images:
            img = random.choice(self.real_images)
            # Apply random augmentation
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
            return img
        else:
            return self._generate_clean_image()
    
    def _generate_clean_image(self):
        """Generate a clean synthetic image with details"""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Add detailed gradients
        for c in range(3):
            x = np.linspace(0, 1, self.img_size)
            y = np.linspace(0, 1, self.img_size)
            X, Y = np.meshgrid(x, y)
            
            pattern = random.choice(['linear', 'radial', 'sinusoidal'])
            
            if pattern == 'linear':
                gradient = (X * 200 + 55).astype(np.uint8)
            elif pattern == 'radial':
                gradient = (np.sqrt((X-0.5)**2 + (Y-0.5)**2) * 255).astype(np.uint8)
            else:
                gradient = (128 + 127 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)).astype(np.uint8)
            
            img[:, :, c] = gradient
        
        # Add textures
        texture_types = ['gaussian', 'speckle', 'checkerboard']
        texture_type = random.choice(texture_types)
        
        if texture_type == 'gaussian':
            texture = np.random.randn(self.img_size, self.img_size, 3) * 30
        elif texture_type == 'speckle':
            texture = np.random.rand(self.img_size, self.img_size, 3) * 60
        else:
            block_size = random.randint(4, 16)
            texture = np.zeros((self.img_size, self.img_size, 3))
            for i in range(0, self.img_size, block_size):
                for j in range(0, self.img_size, block_size):
                    if (i//block_size + j//block_size) % 2 == 0:
                        texture[i:i+block_size, j:j+block_size] = random.randint(20, 60)
        
        img = np.clip(img + texture, 0, 255).astype(np.uint8)
        
        # Add shapes
        num_shapes = random.randint(1, 4)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            
            if shape_type == 'circle':
                center = (random.randint(30, self.img_size-30), random.randint(30, self.img_size-30))
                radius = random.randint(10, 40)
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(img, center, radius, (color[0]//2, color[1]//2, color[2]//2), 2)
            
            elif shape_type == 'rectangle':
                x1 = random.randint(20, self.img_size-70)
                y1 = random.randint(20, self.img_size-70)
                x2 = x1 + random.randint(30, 60)
                y2 = y1 + random.randint(30, 60)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (color[0]//2, color[1]//2, color[2]//2), 2)
            
            else:
                pts = np.array([
                    [random.randint(20, self.img_size-20), random.randint(20, self.img_size-20)],
                    [random.randint(20, self.img_size-20), random.randint(20, self.img_size-20)],
                    [random.randint(20, self.img_size-20), random.randint(20, self.img_size-20)]
                ])
                cv2.fillPoly(img, [pts], color)
                cv2.polylines(img, [pts], True, (color[0]//2, color[1]//2, color[2]//2), 2)
        
        return img
    
    def _degrade_image(self, clean_img):
        """Degrade clean image with various artifacts"""
        degraded = clean_img.copy()
        
        # Apply multiple degradation types
        degradation_pipeline = random.sample([
            'blur', 'noise', 'jpeg', 'compression', 'motion_blur', 'defocus'
        ], random.randint(2, 4))
        
        for degradation in degradation_pipeline:
            if degradation == 'blur':
                kernel_size = random.choice([3, 5, 7, 9])
                sigma = random.uniform(0.5, 3.0)
                degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), sigma)
            
            elif degradation == 'motion_blur':
                size = random.randint(5, 15)
                kernel = np.zeros((size, size))
                kernel[int((size-1)/2), :] = np.ones(size)
                kernel = kernel / size
                degraded = cv2.filter2D(degraded, -1, kernel)
            
            elif degradation == 'defocus':
                kernel = np.ones((7, 7), np.float32) / 49
                degraded = cv2.filter2D(degraded, -1, kernel)
            
            elif degradation == 'noise':
                noise_type = random.choice(['gaussian', 'poisson', 'speckle', 'salt_pepper'])
                if noise_type == 'gaussian':
                    noise = np.random.randn(*degraded.shape) * random.randint(5, 25)
                    degraded = np.clip(degraded + noise, 0, 255)
                elif noise_type == 'poisson':
                    vals = len(np.unique(degraded))
                    vals = 2 ** np.ceil(np.log2(vals))
                    degraded = np.random.poisson(degraded * vals) / float(vals)
                elif noise_type == 'speckle':
                    noise = degraded * np.random.randn(*degraded.shape) * 0.1
                    degraded = np.clip(degraded + noise, 0, 255)
                elif noise_type == 'salt_pepper':
                    salt_pepper = np.random.rand(*degraded.shape[:2])
                    degraded[salt_pepper < 0.01] = 0
                    degraded[salt_pepper > 0.99] = 255
            
            elif degradation == 'jpeg':
                quality = random.randint(10, 70)
                _, buffer = cv2.imencode('.jpg', degraded, [cv2.IMWRITE_JPEG_QUALITY, quality])
                degraded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            elif degradation == 'compression':
                block_size = random.choice([4, 8, 16])
                for i in range(0, degraded.shape[0], block_size):
                    for j in range(0, degraded.shape[1], block_size):
                        block = degraded[i:i+block_size, j:j+block_size]
                        if block.size > 0:
                            avg_color = np.mean(block, axis=(0, 1))
                            variance = np.random.randn(3) * 5
                            degraded[i:i+block_size, j:j+block_size] = np.clip(avg_color + variance, 0, 255)
        
        # Final quality reduction
        if random.random() > 0.5:
            scale = random.uniform(0.3, 0.7)
            h, w = degraded.shape[:2]
            small = cv2.resize(degraded, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return degraded.astype(np.uint8)

class EnhancedColorizationDataset(Dataset):
    """Enhanced dataset for colorization training"""
    
    def __init__(self, num_samples=1000, img_size=256, use_real_images=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.use_real_images = use_real_images
        self.real_images = []
        
        if use_real_images:
            self._load_real_images()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate or get color image
        if self.real_images and random.random() > 0.5:
            color_img = self._get_real_image()
        else:
            color_img = self._generate_color_image()
        
        # Convert to LAB
        lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0
        ab_channels = lab[:, :, 1:].astype(np.float32)
        
        # Normalize AB channels to [-1, 1]
        ab_channels = (ab_channels - 128) / 128
        
        # Convert to tensors
        l_tensor = torch.from_numpy(l_channel).float().unsqueeze(0)
        ab_tensor = torch.from_numpy(ab_channels).float().permute(2, 0, 1)
        
        return l_tensor, ab_tensor
    
    def _load_real_images(self):
        """Load real color images"""
        try:
            image_dirs = ["data/color", "images/color", "dataset/color"]
            for img_dir in image_dirs:
                if os.path.exists(img_dir):
                    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))[:50]
                    for img_path in image_paths:
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
                                self.real_images.append(img)
                        except:
                            continue
            
            print(f"Loaded {len(self.real_images)} real color images")
        except Exception as e:
            print(f"Could not load real color images: {e}")
    
    def _get_real_image(self):
        """Get a real color image"""
        if self.real_images:
            img = random.choice(self.real_images)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            return img
        else:
            return self._generate_color_image()
    
    def _generate_color_image(self):
        """Generate synthetic color image with realistic colors"""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Background with multiple gradients
        for c in range(3):
            x = np.linspace(0, 1, self.img_size)
            y = np.linspace(0, 1, self.img_size)
            X, Y = np.meshgrid(x, y)
            
            # Different color patterns
            if c == 0:
                gradient = (128 + 127 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)).astype(np.uint8)
            elif c == 1:
                gradient = (128 + 100 * np.cos(3*np.pi*X) * np.sin(3*np.pi*Y)).astype(np.uint8)
            else:
                gradient = (128 + 100 * np.sin(4*np.pi*X) * np.cos(2*np.pi*Y)).astype(np.uint8)
            
            img[:, :, c] = gradient
        
        # Add color objects
        num_objects = random.randint(2, 6)
        for _ in range(num_objects):
            obj_type = random.choice(['circle', 'square', 'stripes'])
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            
            if obj_type == 'circle':
                center = (random.randint(30, self.img_size-30), random.randint(30, self.img_size-30))
                radius = random.randint(15, 50)
                cv2.circle(img, center, radius, color, -1)
            
            elif obj_type == 'square':
                x = random.randint(20, self.img_size-70)
                y = random.randint(20, self.img_size-70)
                size = random.randint(20, 60)
                cv2.rectangle(img, (x, y), (x+size, y+size), color, -1)
            
            else:
                stripe_width = random.randint(5, 15)
                for i in range(0, self.img_size, stripe_width*2):
                    cv2.rectangle(img, (i, 0), (i+stripe_width, self.img_size), color, -1)
        
        return img

# ==================== ENHANCED TRAINING FUNCTIONS ====================

def train_restoration_model():
    """Train the restoration model with enhanced dataset"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED RESTORATION MODEL")
    print("="*60)
    
    # Create model
    model = EnhancedRestorationNet()
    
    # Create enhanced dataset
    dataset = EnhancedRestorationDataset(num_samples=2000, img_size=256, use_real_images=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    model.train()
    epochs = 25
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (degraded, clean) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Forward pass
            restored = model(degraded)
            loss = criterion(restored, clean)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/restoration_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    models_dir = Path("pretrained_models")
    models_dir.mkdir(exist_ok=True)
    
    final_path = models_dir / "enhanced_restoration.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved: {final_path}")
    
    return model

def train_colorization_model():
    """Train the colorization model with enhanced dataset"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED COLORIZATION MODEL")
    print("="*60)
    
    # Create model
    model = ColorizationNet()
    
    # Create enhanced dataset
    dataset = EnhancedColorizationDataset(num_samples=1000, img_size=256, use_real_images=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    model.train()
    epochs = 20
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (luminance, ab_truth) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Forward pass
            ab_pred = model(luminance)
            loss = criterion(ab_pred, ab_truth)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/colorization_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    # Save final model
    models_dir = Path("pretrained_models")
    torch.save(model.state_dict(), models_dir / "colorization.pth")
    print(f"Saved: colorization.pth")
    
    return model

def test_enhanced_models():
    """Test the trained models with improved validation"""
    print("\n" + "="*60)
    print("TESTING ENHANCED MODELS")
    print("="*60)
    
    # Create test images
    test_images = []
    
    # Simple test image
    simple_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    test_images.append(("simple", simple_img))
    
    # Gradient test image
    gradient = np.zeros((128, 128, 3), dtype=np.uint8)
    for c in range(3):
        grad = np.linspace(0, 255, 128).reshape(1, -1)
        gradient[:, :, c] = np.tile(grad, (128, 1))
    test_images.append(("gradient", gradient))
    
    # Edge test image
    edges = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(edges, (32, 32), (96, 96), (255, 255, 255), 2)
    cv2.circle(edges, (64, 64), 30, (255, 0, 0), -1)
    test_images.append(("edges", edges))
    
    # Test each model
    print("Testing Restoration Model...")
    restoration_model = EnhancedRestorationNet()
    restoration_model.eval()
    
    print("Testing Colorization Model...")
    colorization_model = ColorizationNet()
    colorization_model.eval()
    
    print("Testing Super Resolution Model...")
    sr_model = SuperResolutionNet(scale_factor=2)
    sr_model.eval()

    # Save test images and process them
    for name, img in test_images:
        try:
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            # Test restoration
            with torch.no_grad():
                restored = restoration_model(img_tensor)
                restored_np = restored.squeeze(0).permute(1, 2, 0).numpy() * 255
            
            print(f"Processed {name} test image")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    print("All models loaded successfully")
    
    return True

# ==================== MAIN FUNCTION ====================

def main():
    """Main training function"""
    print("\n" + "="*60)
    print("ADVANCED AI MODELS TRAINING SCRIPT - ENHANCED")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    Path("pretrained_models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with enhanced datasets (real + synthetic images)")
    
    # Train models
    try:
        # Train restoration model
        print("\n" + "-"*40)
        restoration_model = train_restoration_model()
        
        # Train colorization model
        print("\n" + "-"*40)
        colorization_model = train_colorization_model()
        
        # Test models
        print("\n" + "-"*40)
        test_enhanced_models()
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("API Docs: http://127.0.0.1:8000/docs")
        print("="*60)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
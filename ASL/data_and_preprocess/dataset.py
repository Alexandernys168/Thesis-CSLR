import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from ASL.model_and_config.config import CONFIG
import torchvision.transforms as transforms

class FastWLASLDataset(Dataset):
    def __init__(self, json_path, tensor_dir, augment=False, num_classes=None):
        self.tensor_dir = tensor_dir
        self.augment = augment
        self.load_mode = CONFIG.get("load_mode", "preprocessed")
        self.video_dir = CONFIG.get("video_dir", "")
        
        if self.load_mode == "on_the_fly" and not os.path.exists(self.video_dir):
            print(f"[WARNING] Video directory {self.video_dir} does not exist.")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Requirement 1: Dynamic Class Filtering
        if num_classes is not None:
            original_count = len(self.data)
            # Filter logic: Keep samples where label < num_classes
            self.data = [d for d in self.data if d['label'] < num_classes]
            print(f"Filtered dataset from {original_count} total samples to {len(self.data)} samples (Classes 0 to {num_classes-1}).")
            
    def __len__(self):
        return len(self.data)
        
    def _apply_augmentations(self, video_tensor):
        """Applies consistent spatial/color augmentations."""
        # Input: (C, T, H, W)
        
        # 1. Temporal Sub-sampling (If we have more frames than needed)
        # Note: In 'preprocessed' mode, we usually have exactly frames_per_clip.
        # But if 'on_the_fly' or if preprocessing stored more frames, we can sub-sample.
        c, t, h, w = video_tensor.shape
        target_frames = CONFIG.get("frames_per_clip", 64)
        
        if self.augment and t > target_frames:
             # Random start index
             start_idx = random.randint(0, t - target_frames)
             video_tensor = video_tensor[:, start_idx:start_idx+target_frames, :, :]
        elif t > target_frames:
             # Center crop temporally for validation
             start_idx = (t - target_frames) // 2
             video_tensor = video_tensor[:, start_idx:start_idx+target_frames, :, :]

        # Update t after temporal crop
        c, t, h, w = video_tensor.shape
        
        # 2. Random Horizontal Flip (p=0.5)
        if random.random() < 0.5:
             video_tensor = torch.flip(video_tensor, [-1])

        # 3. Random Rotation (-15 to 15 degrees)
        rot_range = CONFIG.get("aug_rotation_range", 15)
        if self.augment and rot_range > 0:
             angle = random.uniform(-rot_range, rot_range)
             # Rotate all frames by the same angle
             video_tensor = F.rotate(video_tensor, angle) # F.rotate handles (..., H, W)

        # 4. Random Crop (or Center Crop)
        th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
        
        if self.augment:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            # Center Crop (Validation/Testing)
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            
        video_tensor = video_tensor[..., i:i+th, j:j+tw]
        
        # 5. Random Erasing (Cutout)
        erase_prob = CONFIG.get("aug_erase_prob", 0.2)
        if self.augment and random.random() < erase_prob:
             # Erase a compatible rectangle on all frames
             # Scale: proportion of image area to erase
             scale = (0.02, 0.33)
             ratio = (0.3, 3.3)
             
             # Need to manually implement since torchvision.transforms.RandomErasing expects (C, H, W) 
             # and we want consistent erasing across T
             
             area = th * tw
             target_area = random.uniform(*scale) * area
             aspect_ratio = random.uniform(*ratio)
             
             h_rect = int(round(np.sqrt(target_area * aspect_ratio)))
             w_rect = int(round(np.sqrt(target_area / aspect_ratio)))
             
             if h_rect < th and w_rect < tw:
                 top = random.randint(0, th - h_rect)
                 left = random.randint(0, tw - w_rect)
                 
                 # Set region to 0 (Black)
                 video_tensor[..., top:top+h_rect, left:left+w_rect] = 0.0

        return video_tensor

    def _load_video_on_the_fly(self, video_id):
        """Loads video from .mp4, resizes, and samples frames."""
        # Use torchvision or opencv. Here we use torchvision for simplicity if available,
        # or fallback to custom logic if needed. For now, let's assume we use torchvision.io
        # Note: 'torchvision.io' requires specific backend. 
        # Simpler approach: use cv2 to be robust on Windows/Linux without ffmpeg quirks.
        import cv2
        
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video {video_path} not found")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # CV2 is BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
             raise ValueError("Empty video")
             
        # Convert to Tensor (T, H, W, C) -> (C, T, H, W)
        # Resize logic similar to preprocess.py
        
        # Sampling frames
        target_frames = CONFIG.get("frames_per_clip", 64)
        total_frames = len(frames)
        
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        sampled_frames = [frames[i] for i in indices]
        
        # Stack -> (T, H, W, C)
        video_tensor = torch.from_numpy(np.stack(sampled_frames))
        # Permute -> (C, T, H, W)
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        # Resize spatial dims
        resize_size = CONFIG.get("resize_size", 256)
        # Using functional interpolate for resizing logic
        # Expects (Batch, C, H, W) or (C, H, W). We have (C, T, H, W).
        # We can view as (C * T, H, W)
        C, T, H, W = video_tensor.shape
        video_tensor = video_tensor.view(C * T, H, W)
        # Need float for interpolation usually, but resize can work on byte sometimes.
        # Prefer float for safety in resize
        video_tensor = F.resize(video_tensor, [resize_size, resize_size], antialias=True)
        video_tensor = video_tensor.view(C, T, resize_size, resize_size)
        
        return video_tensor
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        video_id = sample['video_id']
        label = sample['label']
        
        tensor_path = os.path.join(self.tensor_dir, f"{video_id}.pt")
        
        try:
            if self.load_mode == "preprocessed":
                if not os.path.exists(tensor_path):
                    raise FileNotFoundError
                # Load Tensor (C, T, H, W)
                video_tensor = torch.load(tensor_path)
                
            elif self.load_mode == "on_the_fly":
                video_tensor = self._load_video_on_the_fly(video_id)
            
            # Apply Augmentations (Best done on uint8/raw tensor)
            video_tensor = self._apply_augmentations(video_tensor)
            
            # Normalize to 0.0 - 1.0 (Simple Scaling)
            # REMOVED: ImageNet Mean/Std subtraction (It was causing instability)
            video_tensor = video_tensor.float() / 255.0
            
            return video_tensor, label
            
        except FileNotFoundError:
            # Print warning so we know if data is missing!
            print(f"⚠️ WARNING: Missing tensor for {video_id} - Returning Black Frame!")
            
            # Fallback: Return Zeros
            c, t = 3, CONFIG.get("frames_per_clip", 64)
            th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
            return torch.zeros((c, t, th, tw), dtype=torch.float32), label

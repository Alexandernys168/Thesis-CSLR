import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from model_and_config.config import CONFIG
import torchvision.transforms as transforms

class VideoTransforms:
    def __init__(self, augment=False):
        self.augment = augment
        self.config = CONFIG

        self.jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)

    def temporal_sample(self, video_tensor):
        c, t, h, w = video_tensor.shape
        target_frames = self.config.get("frames_per_clip", 64)
        
        if self.augment and t > target_frames:
             # Random start index
             start_idx = random.randint(0, t - target_frames)
             video_tensor = video_tensor[:, start_idx:start_idx+target_frames, :, :]
        elif t > target_frames:
             # Center crop temporally for validation
             start_idx = (t - target_frames) // 2
             video_tensor = video_tensor[:, start_idx:start_idx+target_frames, :, :]
        return video_tensor

    def spatial_augment(self, video_tensor):
        # 1. Random Horizontal Flip (p=0.5)
        if self.augment and random.random() < self.config.get("aug_prob_flip", 0.5):
             video_tensor = torch.flip(video_tensor, [-1])

        # 2. Random Rotation (-15 to 15 degrees)
        rot_range = self.config.get("aug_rotation_range", 15)
        if self.augment and rot_range > 0:
             angle = random.uniform(-rot_range, rot_range)
             # Rotate all frames by the same angle
             video_tensor = F.rotate(video_tensor, angle) # F.rotate handles (..., H, W)
        return video_tensor

    def color_augment(self, video_tensor):
        if not self.augment or not self.config.get("aug_color_jitter", False):
            return video_tensor

        # OPTIMIZATION: Instead of manual reshapes and functional calls,
        # use the built-in ColorJitter on the (T, C, H, W) view.
        # torchvision.transforms.ColorJitter supports batches of images (T).
        
        # (C, T, H, W) -> (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        video_tensor = self.jitter(video_tensor)
        # (T, C, H, W) -> (C, T, H, W)
        return video_tensor.permute(1, 0, 2, 3)

    def crop(self, video_tensor):
        c, t, h, w = video_tensor.shape
        th, tw = self.config.get("crop_size", 224), self.config.get("crop_size", 224)
        
        if self.augment:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            # Center Crop (Validation/Testing)
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            
        return video_tensor[..., i:i+th, j:j+tw]

    def random_erase(self, video_tensor):
        th, tw = self.config.get("crop_size", 224), self.config.get("crop_size", 224)
        erase_prob = self.config.get("aug_erase_prob", 0.2)
        
        if self.augment and random.random() < erase_prob:
             # Erase a compatible rectangle on all frames
             # Scale: proportion of image area to erase
             scale = (0.02, 0.33)
             ratio = (0.3, 3.3)
             
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

    def normalize(self, video_tensor):
# OPTIMIZATION: Float division is slow. 
        # Convert to float and multiply by (1/255) is often faster.
        return video_tensor.to(torch.float32).mul_(1.0/255.0)

    def denormalize(self, video_tensor):
        # Inverse of normalize: scale back to 0-255 uint8 range for visualization
        # Input: Tensor (C, T, H, W) or (C, H, W) with values 0.0-1.0
        # Output: Tensor (C, T, H, W) or (C, H, W) with values 0-255
        
        # Clamp to ensure proper range
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        return (video_tensor * 255.0).byte()

    def __call__(self, video_tensor):
        # Apply all steps in order
        video_tensor = self.temporal_sample(video_tensor)
        video_tensor = self.crop(video_tensor)
        video_tensor = self.spatial_augment(video_tensor)
        video_tensor = self.color_augment(video_tensor)
        
        video_tensor = self.random_erase(video_tensor)
        return self.normalize(video_tensor)


class FastWLASLDataset(Dataset):
    def __init__(self, json_path, tensor_dir, augment=False, num_classes=None):
        self.tensor_dir = tensor_dir
        self.augment = augment
        self.load_mode = CONFIG.get("load_mode", "preprocessed")
        self.video_dir = CONFIG.get("video_dir", "")
        
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if num_classes is not None:
            self.data = [d for d in self.data if d['label'] < num_classes]
            
        # # Requirement 1: Dynamic Class Filtering
        # if num_classes is not None:
        #     original_count = len(self.data)
        #     # Filter logic: Keep samples where label < num_classes
        #     self.data = [d for d in self.data if d['label'] < num_classes]
        #     print(f"Filtered dataset from {original_count} total samples to {len(self.data)} samples (Classes 0 to {num_classes-1}).")

        # self.transforms = VideoTransforms(augment=augment)

        # # 3. RAM CACHE INITIALIZATION (The Fix)
        # # We only cache if we are in 'preprocessed' mode to save memory
        # self.cache = {}
        # if self.load_mode == "preprocessed":
        #     print(f"🚀 Loading {len(self.data)} tensors into RAM... (This may take a few minutes)")
        #     from tqdm import tqdm
            
        #     # Use tqdm so you don't panic during the wait
        #     for item in tqdm(self.data, desc="Caching Data"):
        #         vid = item['video_id']
        #         path = os.path.join(self.tensor_dir, f"{vid}.pt")
        #         try:
        #             # Load to CPU memory immediately
        #             # We store it in the dict so it stays in RAM
        #             self.cache[vid] = torch.load(path)
        #         except FileNotFoundError:
        #             # Handle missing files gracefully
        #             pass
        #     print(f"✅ Cached {len(self.cache)} tensors in RAM.")
            
    def __len__(self):
        return len(self.data)

    def _load_video_on_the_fly(self, video_id):
        """Loads video from .mp4, resizes, and samples frames."""
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
        C, T, H, W = video_tensor.shape
        video_tensor = video_tensor.reshape(C * T, H, W)
        # Need float for interpolation usually, but resize can work on byte sometimes.
        # Prefer float for safety in resize
        video_tensor = F.resize(video_tensor, [resize_size, resize_size], antialias=True)
        video_tensor = video_tensor.reshape(C, T, resize_size, resize_size)
        
        return video_tensor
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        video_id = sample['video_id']
        label = sample['label']
        

        # try:
        #     if self.load_mode == "preprocessed":
        #         # 4. INSTANT LOOKUP (No Disk I/O)
        #         if video_id in self.cache:
        #             # .clone() is crucial! 
        #             # If we don't clone, transforms will modify the cached version
        #             # and corrupt your data for the next epoch.
        #             video_tensor = self.cache[video_id].clone()
        #         else:
        #             raise FileNotFoundError
            
        #     # A. Temporal Sample
        #     c, t, h, w = video_tensor.shape
        #     target_frames = CONFIG.get("frames_per_clip", 64)
        #     if t > target_frames:
        #         start = random.randint(0, t - target_frames) if self.augment else (t - target_frames)//2
        #         video_tensor = video_tensor[:, start:start+target_frames, :, :]

        #     # B. Random Crop (Reduces 256x256 -> 224x224)
        #     # This is fast on CPU and saves transfer time
        #     h, w = video_tensor.shape[-2:]
        #     th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
        #     if self.augment:
        #         i = random.randint(0, h - th)
        #         j = random.randint(0, w - tw)
        #     else:
        #         i, j = (h - th) // 2, (w - tw) // 2
        #     video_tensor = video_tensor[..., i:i+th, j:j+tw]

        #     # Return raw tensor (likely uint8 or float). 
        #     # DO NOT NORMALIZE HERE. Leave it for GPU.
        #     return video_tensor, label
            
        # except FileNotFoundError:
        #     # Fallback: Return Zeros
        #     c, t = 3, CONFIG.get("frames_per_clip", 64)
        #     th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
        #     return torch.zeros((c, t, th, tw), dtype=torch.float32), label


        tensor_path = os.path.join(self.tensor_dir, f"{video_id}.pt")
        
        try:
            if self.load_mode == "preprocessed":
                if not os.path.exists(tensor_path):
                   raise FileNotFoundError
                # Load Tensor (C, T, H, W)
                video_tensor = torch.load(tensor_path)
               
            elif self.load_mode == "on_the_fly":
               video_tensor = self._load_video_on_the_fly(video_id)
          
            # --- 2. CPU SLICING (Critical for Batching & Speed) ---
            # We must cut the video to a fixed size HERE so the DataLoader can stack them.
            # We do NOT do color/rotation here (that is for GPU).
            
            c, t, h, w = video_tensor.shape
            target_frames = CONFIG.get("frames_per_clip", 64)
            crop_size = CONFIG.get("crop_size", 224)

            # A. Temporal Sample (Cut to 64 frames)
            if t > target_frames:
                # Random start for train, Center for val
                start = random.randint(0, t - target_frames) if self.augment else (t - target_frames)//2
                video_tensor = video_tensor[:, start:start+target_frames, :, :]
            elif t < target_frames:
                # Optional: Pad if video is too short (Simple replication padding)
                # For now, we assume most videos are long enough. 
                pass

            # B. Spatial Crop (Cut to 224x224)
            # This reduces data transfer to GPU by ~30%
            if self.augment:
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
            else:
                i = (h - crop_size) // 2
                j = (w - crop_size) // 2
                
            video_tensor = video_tensor[..., i:i+crop_size, j:j+crop_size]

            # --- 3. RETURN RAW TENSOR ---
            # Returns uint8 (0-255). 
            # Normalization and Color Jitter will happen in train_sagemaker.py
            return video_tensor, label
          
        except FileNotFoundError:
           # Print warning so we know if data is missing!
           print(f"⚠️ WARNING: Missing tensor for {video_id} - Returning Black Frame!")
          
           # Fallback: Return Zeros
           c, t = 3, CONFIG.get("frames_per_clip", 64)
           th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
           return torch.zeros((c, t, th, tw), dtype=torch.float32), label




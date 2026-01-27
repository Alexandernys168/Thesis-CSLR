import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from ASL.model_and_config.config import CONFIG
import torchvision.transforms as transforms

class VideoTransforms:
    def __init__(self, augment=False):
        self.augment = augment
        self.config = CONFIG

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

        # Parameters (standard defaults)
        brightness = 0.2
        contrast = 0.2
        saturation = 0.2
        hue = 0.1

        # Sample factors
        b_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        c_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        s_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        h_factor = random.uniform(-hue, hue)

        # Define transforms
        # We need to operate on (C, T, H, W). 
        # Reshape to (C, T*H, W) to use functional transforms efficiently and consistently across time.
        c, t, h, w = video_tensor.shape
        video_tensor = video_tensor.reshape(c, t * h, w)

        transforms_list = [
            lambda x: F.adjust_brightness(x, b_factor),
            lambda x: F.adjust_contrast(x, c_factor),
            lambda x: F.adjust_saturation(x, s_factor),
            lambda x: F.adjust_hue(x, h_factor)
        ]
        random.shuffle(transforms_list)

        for t_func in transforms_list:
            video_tensor = t_func(video_tensor)

        # Reshape back
        video_tensor = video_tensor.reshape(c, t, h, w)
        return video_tensor

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
        # Normalize to 0.0 - 1.0 (Simple Scaling)
        return video_tensor.float() / 255.0

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
        video_tensor = self.spatial_augment(video_tensor)
        video_tensor = self.color_augment(video_tensor)
        video_tensor = self.crop(video_tensor)
        video_tensor = self.random_erase(video_tensor)
        return self.normalize(video_tensor)


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

        self.transforms = VideoTransforms(augment=augment)
            
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
            video_tensor = self.transforms(video_tensor)
            
            return video_tensor, label
            
        except FileNotFoundError:
            # Print warning so we know if data is missing!
            print(f"⚠️ WARNING: Missing tensor for {video_id} - Returning Black Frame!")
            
            # Fallback: Return Zeros
            c, t = 3, CONFIG.get("frames_per_clip", 64)
            th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
            return torch.zeros((c, t, th, tw), dtype=torch.float32), label

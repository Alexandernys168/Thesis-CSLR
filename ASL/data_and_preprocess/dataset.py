import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from config import CONFIG
import torchvision.transforms as transforms

class FastWLASLDataset(Dataset):
    def __init__(self, json_path, tensor_dir, augment=False):
        self.tensor_dir = tensor_dir
        self.augment = augment
        self.load_mode = CONFIG.get("load_mode", "preprocessed")
        self.video_dir = CONFIG.get("video_dir", "")
        
        if self.load_mode == "on_the_fly" and not os.path.exists(self.video_dir):
            print(f"[WARNING] Video directory {self.video_dir} does not exist.")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def _apply_augmentations(self, video_tensor):
        """Applies consistent spatial/color augmentations."""
        # Input: (C, T, H, W)
        
        # 1. Random Horizontal Flip (p=0.5)
        if random.random() < 0.5:
             video_tensor = torch.flip(video_tensor, [-1])

        # 2. Random Crop (or Center Crop)
        c, t, h, w = video_tensor.shape
        th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
        
        if self.augment:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            # Center Crop (Validation/Testing)
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            
        video_tensor = video_tensor[..., i:i+th, j:j+tw]
        
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
    
    
    def _compute_farneback_flow(self, frames):
        """
        Computes Dense Optical Flow between consecutive frames using Farneback algorithm.
        Input: List of numpy arrays (H, W, 3) (RGB)
        Output: Tensor (2, T, H, W)
        """
        import cv2
        
        flow_frames = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        # Iterate from 2nd frame
        for i in range(1, len(frames)):
            curr = frames[i]
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
            
            # Compute Flow
            # Parameters from typical flow research or OpenCV defaults
            # pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # flow is (H, W, 2)
            
            flow_frames.append(flow)
            prev_gray = curr_gray
            
        # If only 1 frame? 
        if len(flow_frames) == 0:
            # Handle edge case: duplicate zero flow
            h, w = prev_gray.shape
            flow_frames.append(np.zeros((h, w, 2), dtype=np.float32))
            
        # Stack -> (T, H, W, 2)
        # Note: We lost 1 frame in time dimension (N frames -> N-1 flows).
        # We can duplicate the last one to keep T consistent or pad first.
        # Let's pad the last flow to match RGB length if needed, or just let it be T-1.
        # I3D usually effectively works on T. 
        # Let's duplicate the last flow frame.
        flow_frames.append(flow_frames[-1])
        
        flow_stack = np.stack(flow_frames) # (T, H, W, 2)
        
        # To Tensor -> (2, T, H, W)
        flow_tensor = torch.from_numpy(flow_stack).permute(3, 0, 1, 2)
        return flow_tensor

    def _load_video_frames(self, video_id):
        # ... logic to load raw video or cached RGB tensor ...
        # For simplicity, reusing existing tensor loading if available, 
        # but to generate flow we ideally need original content or high-res RGB.
        # If we only have 256x256 RGB tensors, we can compute flow on them too.
        
        tensor_path = os.path.join(self.tensor_dir, f"{video_id}.pt")
        if os.path.exists(tensor_path):
             # Load (C, T, H, W)
             video_tensor = torch.load(tensor_path)
             # Convert back to list of numpy arrays for CV2
             # (C, T, H, W) -> (T, H, W, C)
             video_numpy = video_tensor.permute(1, 2, 3, 0).numpy()
             # Ideally assume it's float 0-1 or uint8? 
             # dataset says it saves them ... preprocessed usually saved as tensors.
             # If float, convert to uint8 0-255
             if video_numpy.max() <= 1.0:
                 video_numpy = (video_numpy * 255).astype(np.uint8)
             else:
                 video_numpy = video_numpy.astype(np.uint8)
                 
             return [video_numpy[i] for i in range(video_numpy.shape[0])]
        else:
             # Fallback to loading video file directly
             return self._load_video_on_the_fly_raw(video_id)

    def _load_video_on_the_fly_raw(self, video_id):
        # Same as _load_video_on_the_fly but returns list of frames
        import cv2
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video {video_path} not found")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _get_flow(self, video_id, rgb_frames=None):
        # Check cache
        flow_path = os.path.join(self.tensor_dir, f"{video_id}_flow.pt")
        
        if os.path.exists(flow_path):
            return torch.load(flow_path)
            
        # Compute
        if rgb_frames is None:
            rgb_frames = self._load_video_frames(video_id)
            
        # Ensure we resize frames to consistent size BEFORE flow?
        # Or compute flow on full res then resize? 
        # Computing on smaller res is much faster. 
        # Let's resize frames to resize_size first.
        resize_size = CONFIG.get("resize_size", 256)
        import cv2
        resized_frames = [cv2.resize(f, (resize_size, resize_size)) for f in rgb_frames]
        
        flow_tensor = self._compute_farneback_flow(resized_frames)
        
        # Cache it
        torch.save(flow_tensor, flow_path)
        return flow_tensor

    def __getitem__(self, idx):
        sample = self.data[idx]
        video_id = sample['video_id']
        label = sample['label']
        
        stream_type = CONFIG.get("stream_type", "rgb")
        
        try:
            rgb_tensor = None
            flow_tensor = None
            
            # --- Load RGB ---
            # Always load RGB if needed or if we need it to generate flow (and flow not cached)
            # We assume basic RGB tensor exists as per 'load_mode'
            tensor_path = os.path.join(self.tensor_dir, f"{video_id}.pt")
            if not os.path.exists(tensor_path):
                # Attempt to generate/load? ignoring for now, assume RGB exists
                 raise FileNotFoundError(f"RGB Tensor {video_id} not found")
            
            if stream_type in ["rgb", "two_stream"]:
                rgb_tensor = torch.load(tensor_path)
                rgb_tensor = self._apply_augmentations(rgb_tensor) # (C, T, H, W)
                rgb_tensor = rgb_tensor.float() / 255.0
                
            # --- Load Flow ---
            if stream_type in ["flow", "two_stream"]:
                # Check for cached flow
                flow_path = os.path.join(self.tensor_dir, f"{video_id}_flow.pt")
                if os.path.exists(flow_path):
                    flow_tensor = torch.load(flow_path)
                else:
                    # Generate flow
                    # Need RGB frames (source)
                    # We can load the saved RGB tensor to compute flow
                    rgb_source_tensor = torch.load(tensor_path) #(C, T, H, W)
                    # Convert to frames
                    perms = rgb_source_tensor.permute(1, 2, 3, 0).numpy() # (T, H, W, C)
                    # Assuming stored as ByteTensor (0-255)
                    frames = [perms[i].astype(np.uint8) for i in range(perms.shape[0])]
                    
                    flow_tensor = self._compute_farneback_flow(frames) #(2, T, H, W)
                    # Cache
                    torch.save(flow_tensor, flow_path)
                
                # Apply Augmentations to Flow?
                # Ideally same augmentations as RGB if Two-Stream (Spatial Consistency).
                # Implementation difficulty: randomization.
                # For now, let's treat separately or disable geometric augs for fusion simplicty,
                # OR set seed. 
                # Let's just apply independently for now or skip spatial augs for flow.
                # Actually, flow should be flipped if image is flipped.
                # NOT IMPLEMENTED: Synced augmentation.
                
                # Normalize Flow: Usually flows are pixels displacement. 
                # Model expects -1 to 1 or similar? Or just raw?
                # I3D flow inputs are typically expected to be in [-20, 20] range then scaled to [-1, 1].
                # Or just raw centered. 
                # Simple normalization: Clip to [-20, 20] and scale.
                flow_tensor = torch.clamp(flow_tensor, -20, 20)
                flow_tensor = flow_tensor / 20.0 # Scale to -1, 1
                
            if stream_type == "rgb":
                return rgb_tensor, label
            elif stream_type == "flow":
                return flow_tensor, label
            else:
                return (rgb_tensor, flow_tensor), label
            
        except FileNotFoundError:
            print(f"⚠️ WARNING: Missing data for {video_id} - Returning Black Frame!")
            # Handle zeroes
            c, t = 3, CONFIG.get("frames_per_clip", 64)
            th, tw = CONFIG.get("crop_size", 224), CONFIG.get("crop_size", 224)
            zeros = torch.zeros((c, t, th, tw), dtype=torch.float32)
            if stream_type == "rgb": return zeros, label
            if stream_type == "flow": return torch.zeros((2, t, th, tw), dtype=torch.float32), label
            return (zeros, torch.zeros((2, t, th, tw), dtype=torch.float32)), label

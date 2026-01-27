import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from ASL.model_and_config.config import CONFIG
from ASL.data_and_preprocess.dataset import VideoTransforms

def load_random_video_entry(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return random.choice(data)

def load_raw_video(video_path):
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    # Return as list of numpy arrays
    return frames

def frames_to_tensor(frames):
    # (T, H, W, C) -> (C, T, H, W)
    tensor = torch.from_numpy(np.stack(frames))
    return tensor.permute(3, 0, 1, 2)

def main():
    # 1. Load config and entry
    print("Loading valid entry from train_100.json...")
    entry = load_random_video_entry(CONFIG["train_json"])
    video_id = entry['video_id']
    print(f"Selected Video ID: {video_id}, Gloss: {entry['gloss']}")

    # 2. Try to load raw video
    video_dir = CONFIG.get("video_dir", "")
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    
    raw_frames = load_raw_video(video_path)
    if raw_frames is None:
        print(f"⚠️ Warning: Original video {video_path} not found.")
        print("Cannot demonstrate 'Raw' to 'Resize' steps fully without original video.")
        print("Please ensure 'video_dir' in config.py is correct and videos exist.")
        # Try to find *a* video that works? Or just exit?
        # For now, let's just create a dummy "Raw" frame or fail? 
        # User requested: "Load one random video... Raw Original" 
        # I'll try 5 times to find a video that exists
        found = False
        for _ in range(5):
            entry = load_random_video_entry(CONFIG["train_json"])
            video_id = entry['video_id']
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            raw_frames = load_raw_video(video_path)
            if raw_frames is not None:
                found = True
                print(f"Found valid video: {video_id}")
                break
        
        if not found:
            print("❌ Could not find any video files. Aborting visualization.")
            return

    # Helper to get usage middle frame for static display
    def get_display_frame(tensor_cthw):
        # Convert (C, T, H, W) -> (H, W, C) for plotting
        # Take middle frame
        c, t, h, w = tensor_cthw.shape
        mid = t // 2
        frame = tensor_cthw[:, mid, :, :].permute(1, 2, 0)
        return frame

    # Instantiate Transforms
    transforms = VideoTransforms(augment=True)
    
    # --- STAGE 0: RAW ---
    # Raw frames are list of (H, W, C) uint8
    raw_tensor = frames_to_tensor(raw_frames) # (C, T, H, W)
    print(f"Raw Tensor: {raw_tensor.shape}")
    
    # --- STAGE 1: RESIZE ---
    # Manually replicate Resize logic from preprocess.py / dataset.py
    # dataset.py _load_video_on_the_fly uses:
    # F.resize(video_tensor, [resize_size, resize_size], antialias=True)
    resize_size = CONFIG.get("resize_size", 256)
    c, t, h, w = raw_tensor.shape
    # View as (C*T, H, W) for resize
    resized_tensor = raw_tensor.reshape(c * t, h, w)
    resized_tensor = F.resize(resized_tensor, [resize_size, resize_size], antialias=True)
    resized_tensor = resized_tensor.reshape(c, t, resize_size, resize_size)
    print(f"Resized Tensor: {resized_tensor.shape}")
    
    # --- STAGE 2: CROP ---
    # Use VideoTransforms.crop (it handles both random and center, let's FORCE Center for visualization consistency?
    # User said "Cropped to 224x224 (Center Crop)".
    # But also "Row 1: ... Augment". usually Augment includes Random Crop.
    # The prompt says: "Step 2 (Crop): The frame cropped to 224x224 (Center Crop)" specifically.
    # Then "Step 3 (Augmentation): The frame after Random Flip / Color Jitter".
    
    # Let's temporarily disable augment for the Crop step to get Center Crop
    transforms.augment = False
    cropped_tensor = transforms.crop(resized_tensor)
    transforms.augment = True # Re-enable for Step 3
    print(f"Cropped (Center): {cropped_tensor.shape}")
    
    # --- STAGE 3: AUGMENT ---
    # User wants: "Random Flip / Color Jitter"
    # VideoTransforms.spatial_augment does Flip, Rotate.
    # What about Color Jitter? It was in `dataset.py` config but not explicitly in `_apply_augmentations` logic index I saw?
    # Ah, I should check `dataset.py` again. I might have missed ColorJitter in my refactor if it wasn't there!
    # Looking at `dataset.py` (previous `view_file` output):
    # `_apply_augmentations` had: Temporal, Flip, Rotation, Crop, Erasing.
    # Where is Color Jitter? 
    # Config has `"aug_color_jitter": True`.
    # But I don't see it in `_apply_augmentations`.
    # Wait, the prompt says "Replicate the exact steps from dataset.py".
    # If `dataset.py` didn't implement it despite Config having it, then I shouldn't invent it unless I fix `dataset.py`.
    # But `dataset.py` original code (lines 34-105) does NOT show ColorJitter.
    # So I will stick to what `dataset.py` does: Flip/Rotate.
    # I'll call `spatial_augment`.
    # Also `random_erase`? The prompt mentions "Random Flip / Color Jitter (if enabled)". 
    # Since Jitter isn't in `dataset.py`, I'll skip Jitter but do the others.
    # I'll chain `spatial_augment` and `random_erase` (if enabled in config).
    
    augmented_tensor = transforms.spatial_augment(cropped_tensor)
    augmented_tensor = transforms.color_augment(augmented_tensor)
    augmented_tensor = transforms.random_erase(augmented_tensor)
    print(f"Augmented: {augmented_tensor.shape}")
    
    # --- STAGE 4: TEMPORAL ---
    # "Display a 'Film Strip' ... 64-frame tensor ... prove looping/sampling"
    # Use temporal_sample
    final_tensor = transforms.temporal_sample(augmented_tensor)
    print(f"Final Tensor (Temporal): {final_tensor.shape}")
    
    # --- NORMALIZE ---
    final_norm_tensor = transforms.normalize(final_tensor)
    
    # --- VISUALIZATION GEN ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Helper for showing image
    def show_img(ax, tensor, title):
        # tensor: (C, H, W) or (C, T, H, W) -> take Mid frame
        if tensor.dim() == 4:
            img = get_display_frame(tensor)
        else:
            img = tensor.permute(1, 2, 0)
            
        # Ensure it's valid for imshow (0-255 uint8 or 0-1 float)
        # Our tensors are uint8 (byte) here, except final_norm_tensor
        if img.dtype == torch.float32:
            img = img.clamp(0, 1)
        else:
            img = img.byte().numpy()
            
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        
    # Row 1: Spatial Steps
    # 1. Raw
    show_img(axes[0, 0], raw_tensor, f"Raw\n{tuple(raw_tensor.shape)[2:]}")
    # 2. Resize
    show_img(axes[0, 1], resized_tensor, f"Resize (256x256)\n{tuple(resized_tensor.shape)[2:]}")
    # 3. Crop (Center)
    show_img(axes[0, 2], cropped_tensor, f"Center Crop (224x224)\n{tuple(cropped_tensor.shape)[2:]}")
    # 4. Augment
    show_img(axes[0, 3], augmented_tensor, "Augment (Flip/Rot/Erase)")
    
    # 5. Final (Normalized & Denormalized)
    # We display the denormalized version of the final tensor
    denorm_tensor = transforms.denormalize(final_norm_tensor)
    show_img(axes[0, 4], denorm_tensor, "Final (Denormalized)")
    
    # Row 2: Temporal Film Strip
    # Show 5 frames from `denorm_tensor`: 0, 16, 32, 48, 63 (or evenly spaced)
    c, t, h, w = denorm_tensor.shape
    frame_indices = np.linspace(0, t-1, 5).astype(int)
    
    for k, idx in enumerate(frame_indices):
        frame_tensor = denorm_tensor[:, idx, :, :] # (C, H, W)
        show_img(axes[1, k], frame_tensor, f"Frame {idx}")
        
    # Save
    out_path = "ASL/preprocessing_debug.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()

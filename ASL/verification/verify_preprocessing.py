import os
import json
import random
import cv2
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Configuration
JSON_PATH = r"a:\Thesis-CSLR\ASL\train_100.json"
TENSOR_DIR = r"a:\Thesis-CSLR\ASL\data_tensors"
# Video directory inferred from previous exploration of preprocess.py
VIDEO_DIR = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\videos"
SAMPLE_COUNT = 5

def load_json_data(path):
    print(f"Loading metadata from {path}...")
    with open(path, 'r') as f:
        return json.load(f)

def get_original_video_info(video_id, start_frame, end_frame):
    """
    Extracts the middle frame and frame count of the trimmed segment.
    """
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return None, 0, (0, 0), "File not found"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, (0, 0), "Could not open video"

    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # WLASL uses 1-based indexing for start_frame.
    # Logic matching preprocess.py:
    start = max(0, start_frame - 1)
    end = end_frame
    
    if end == -1 or end > total_frames_vid:
        end = total_frames_vid

    segment_len = end - start
    
    if segment_len <= 0:
        cap.release()
        return None, 0, (width, height), "Invalid segment length"

    # Get Middle Frame for Verification
    mid_idx = start + segment_len // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, segment_len, (width, height), "Could not read middle frame"
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, segment_len, (width, height), None

def process_sample(sample):
    video_id = sample['video_id']
    print(f"\n--- Checking Sample: {video_id} (Gloss: {sample.get('gloss', 'Unknown')}) ---")
    
    # 1. Load Tensor
    tensor_path = os.path.join(TENSOR_DIR, f"{video_id}.pt")
    tensor_loaded = False
    tensor_data = None
    
    if os.path.exists(tensor_path):
        try:
            tensor_data = torch.load(tensor_path)
            tensor_loaded = True
        except Exception as e:
            print(f"[ERROR] Failed to load tensor: {e}")
    else:
        print(f"[ERROR] Tensor file missing: {tensor_path}")

    if not tensor_loaded:
        return

    # 2. Process Tensor Data
    # Expected layout from preprocess.py: (C, T, H, W)
    # We want to convert to (T, H, W, C) for visualization
    
    # Check shape
    C, T, H, W = tensor_data.shape
    print(f"Tensor Shape: {tensor_data.shape}")
    
    # Verification Check
    print(f"Verification: Tensor Resolution {W}x{H} (Expected 224x224 by user specs, 256x256 by preprocess.py)")
    if W == 224 and H == 224:
        print("  [OK] Resolution is 224x224.")
    else:
        print(f"  [NOTE] Resolution is {W}x{H}.")

    print(f"Verification: Tensor Frame Count {T} (Expected 64)")
    if T == 64:
        print("  [OK] Frame count is 64.")
    else:
        print(f"  [FAIL] Frame count is {T}.")

    # Decode
    # Permute to (T, H, W, C)
    tensor_vis = tensor_data.permute(1, 2, 3, 0)
    
    # Handle Dtypes
    if torch.is_floating_point(tensor_vis):
        print("  Tensor is float, converting to uint8 (x255)")
        tensor_vis = (tensor_vis * 255).clamp(0, 255).byte()
    elif tensor_vis.dtype == torch.uint8:
        print("  Tensor is uint8, using as is")
    else:
        print(f"  Tensor is {tensor_vis.dtype}, casting to byte")
        tensor_vis = tensor_vis.byte()
        
    tensor_np = tensor_vis.cpu().numpy()

    # 3. Load Original Video
    orig_frame, orig_len, orig_res, err = get_original_video_info(
        video_id, sample['frame_start'], sample['frame_end']
    )
    
    if orig_frame is None:
        print(f"[ERROR] Could not load original video: {err}")
    else:
        print(f"Original Video Resolution: {orig_res[0]}x{orig_res[1]}")
        print(f"Original Segment Length: {orig_len} frames")

    # 4. Generate Visualizations
    
    # A. Side-by-Side Montage
    if orig_frame is not None:
        tensor_mid_idx = T // 2
        tensor_mid_frame = tensor_np[tensor_mid_idx]
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(orig_frame)
        axs[0].set_title(f"Original Clip (Middle Frame)\nRes: {orig_res[0]}x{orig_res[1]}")
        axs[0].axis('off')
        
        axs[1].imshow(tensor_mid_frame)
        axs[1].set_title(f"Processed Tensor (Frame {tensor_mid_idx})\nRes: {W}x{H}")
        axs[1].axis('off')
        
        out_png = f"verify_{video_id}_montage.png"
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Generated Montage: {out_png}")
        
    # B. GIF Generation
    out_gif = f"verify_{video_id}.gif"
    try:
        imageio.mimsave(out_gif, tensor_np, fps=12, loop=0)
        print(f"Generated GIF: {out_gif}")
    except Exception as e:
        print(f"[ERROR] Failed to generate GIF: {e}")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Metadata file not found: {JSON_PATH}")
        return

    db = load_json_data(JSON_PATH)
    
    # Filter only entries that likely have files (opt) or just random sample
    print(f"Total entries: {len(db)}")
    
    if len(db) < SAMPLE_COUNT:
        samples = db
    else:
        samples = random.sample(db, SAMPLE_COUNT)
        
    for sample in samples:
        process_sample(sample)

if __name__ == "__main__":
    main()

import os
import json
import cv2
import numpy as np
import torch
import collections
from collections import Counter
from tqdm import tqdm

# Configuration
JSON_PATH = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\WLASL_v0.3.json"
VIDEO_DIR = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\videos"
OUTPUT_DIR = r"a:\Thesis-CSLR\ASL\data_tensors_1000"
NUM_CLASSES = 1000
TARGET_FRAMES = 64
RESIZE_SIZE = 256

def preprocess():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        content = json.load(f)

    print("Selecting top 100 glosses...")
    gloss_counts = {}
    for entry in content:
        gloss = entry['gloss']
        count = len(entry['instances'])
        gloss_counts[gloss] = count

    counter = Counter(gloss_counts)
    top_1000 = counter.most_common(NUM_CLASSES)
    top_glosses = {item[0] for item in top_1000}
    
    # Save class mapping if not exists, though create_subset.py did this.
    # We'll regenerate it to be self-contained or rely on it.
    
    tasks = []
    print("Collecting processing tasks...")
    for entry in content:
        gloss = entry['gloss']
        if gloss in top_glosses:
            for inst in entry['instances']:
                video_id = inst['video_id']
                frame_start = inst['frame_start']
                frame_end = inst['frame_end']
                split = inst['split'] 
                
                # Check validation/train/test here if we wanted to filter, 
                # but we will process ALL available videos for these classes 
                # and let the dataset split them based on the JSON later.
                
                tasks.append({
                    'video_id': video_id,
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'gloss': gloss
                })

    print(f"Total videos to process: {len(tasks)}")
    
    success_count = 0
    fail_count = 0
    failures = []
    
    for task in tqdm(tasks, desc="Preprocessing Videos"):
        video_id = task['video_id']
        start_frame = task['frame_start']
        end_frame = task['frame_end']
        
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.pt")
        
        # Skip if already exists (resume capability)
        if os.path.exists(output_path):
            success_count += 1
            continue
            
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            fail_count += 1
            failures.append({"video_id": video_id, "reason": "Missing File", "path": video_path})
            continue
            
        try:
            frames = load_and_process_video(video_path, start_frame, end_frame)
            if frames is not None:
                # frames is (T, H, W, C) numpy array
                # Convert to tensor
                # Layout for R3D is (C, T, H, W) usually, but we can save as (T, H, W, C) 
                # and permute on load. Let's stick to (C, T, H, W) to match standard Pytorch video format.
                
                tensor = torch.from_numpy(frames) # T, H, W, C
                tensor = tensor.permute(3, 0, 1, 2) # C, T, H, W
                
                # Verify type is uint8
                if tensor.dtype != torch.uint8:
                    tensor = tensor.to(torch.uint8)
                    
                torch.save(tensor, output_path)
                success_count += 1
            else:
                fail_count += 1
                failures.append({"video_id": video_id, "reason": "Load Failed (Empty/Corrupt)"})
        except Exception as e:
            # print(f"Error processing {video_id}: {e}")
            fail_count += 1
            failures.append({"video_id": video_id, "reason": f"Exception: {str(e)}"})
            
    print(f"Preprocessing Complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Failed (missing/corrupt): {fail_count}")

    if failures:
        fail_log_path = os.path.join(OUTPUT_DIR, "preprocessing_failures.json")
        with open(fail_log_path, "w") as f:
            json.dump(failures, f, indent=4)
        print(f"Detailed failure log saved to {fail_log_path}")

def load_and_process_video(path, start_frame, end_frame):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 0-based index correction
    start = max(0, start_frame - 1)
    end = end_frame
    
    # Logic to handle "Already Cropped" videos
    # If the JSON specifies a start frame way outside the video length,
    # but the duration roughly matches the video length, assume it's pre-cropped.
    
    if start >= total_frames:
        expected_len = end - start
        
        # Allow small margin of error (e.g., +/- 15 frames)
        if abs(total_frames - expected_len) < 15 or (total_frames > 0 and expected_len > total_frames):
             # It looks like the video is the segment itself
             # print(f"Detected cropped video {path}: JSON {start}-{end} vs Total {total_frames}")
             start = 0
             end = total_frames
        else:
             cap.release()
             return None
             
    if end < 0 or end > total_frames:
        end = total_frames
        
    segment_len = end - start
    if segment_len <= 0:
        cap.release()
        return None
        
    frames = []
    
    # Optimized Read Strategy
    if segment_len < TARGET_FRAMES:
        # Read all
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(segment_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        if not frames:
            cap.release()
            return None
            
        # Loop padding
        while len(frames) < TARGET_FRAMES:
            frames.extend(frames[:TARGET_FRAMES - len(frames)])
            
    else:
        # Uniform sampling
        indices = np.linspace(start, end-1, TARGET_FRAMES).astype(int)
        last_idx = -1
        for i in indices:
            if i != last_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                if frames: frames.append(frames[-1])
                else: 
                    cap.release()
                    return None
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            last_idx = i
            
    cap.release()
    
    # Ensure count
    if len(frames) < TARGET_FRAMES:
        while len(frames) < TARGET_FRAMES:
             frames.append(frames[-1])
             
    # Resize and Format
    # Doing resize here on CPU is the bottleneck usually, but we only do it once now!
    processed_frames = []
    for frame in frames:
        # Resize to 256x256
        frame = cv2.resize(frame, (RESIZE_SIZE, RESIZE_SIZE))
        # Keep as uint8 (0-255)
        processed_frames.append(frame)
        
    return np.stack(processed_frames) # (T, H, W, C)

if __name__ == "__main__":
    preprocess()

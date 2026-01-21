import os
import json
import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
from config import CONFIG

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def get_smart_crop_box(frame, results, padding=0.2):
    """
    Calculates the bounding box union of hands and face with padding.
    Returns (x1, y1, x2, y2) relative to frame size.
    """
    h, w, _ = frame.shape
    x_coords = []
    y_coords = []
    
    # Helper to add landmarks
    def add_landmarks(landmarks):
        if landmarks:
            for lm in landmarks.landmark:
                x_coords.append(lm.x * w)
                y_coords.append(lm.y * h)
                
    add_landmarks(results.left_hand_landmarks)
    add_landmarks(results.right_hand_landmarks)
    add_landmarks(results.face_landmarks)
    
    # If no detections, return Center Crop box logic (handled by caller or return None)
    if not x_coords:
        return None
        
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Add Padding
    width = max_x - min_x
    height = max_y - min_y
    
    pad_w = width * padding
    pad_h = height * padding
    
    x1 = max(0, int(min_x - pad_w))
    y1 = max(0, int(min_y - pad_h))
    x2 = min(w, int(max_x + pad_w))
    y2 = min(h, int(max_y + pad_h))
    
    return x1, y1, x2, y2

def process_video_smart(video_path, start_frame, end_frame, holistic):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, start_frame - 1)
    end = end_frame if (end_frame > 0 and end_frame <= total_frames) else total_frames
    
    # Read Logic (Same as original preprocess but with smart crop)
    segment_len = end - start
    if segment_len <= 0:
        cap.release()
        return None
        
    frames = []
    target_len = CONFIG["frames_per_clip"]
    
    # Read All Strategy (if short) or Sample (if long)
    # For simplicity and accuracy of motion, let's stick to uniform sampling or read-all-then-sample
    # To keep consistent with original logic:
    
    indices = np.linspace(start, end-1, target_len).astype(int)
    last_idx = -1
    
    for i in indices:
        if i != last_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        ret, frame = cap.read()
        if not ret:
            if frames: frames.append(frames[-1]) # Pad
            else: 
                cap.release()
                return None
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        last_idx = i
        
    cap.release()
    
    # Validation
    if len(frames) < target_len:
         while len(frames) < target_len:
             frames.append(frames[-1])

    # SMART CROP & RESIZE
    processed_frames = []
    
    for frame in frames:
        # MediaPipe Detection
        results = holistic.process(frame)
        
        # Get Crop Box
        box = get_smart_crop_box(frame, results)
        
        if box:
            x1, y1, x2, y2 = box
            # Fallback for empty box
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                 # Center Crop Fallback
                 h, w = frame.shape[:2]
                 s = min(h, w)
                 y1 = (h - s) // 2
                 x1 = (w - s) // 2
                 frame_cropped = frame[y1:y1+s, x1:x1+s]
            else:
                 frame_cropped = frame[y1:y2, x1:x2]
        else:
            # Fallback: Center Crop (Square)
            h, w = frame.shape[:2]
            s = min(h, w)
            y1 = (h - s) // 2
            x1 = (w - s) // 2
            frame_cropped = frame[y1:y1+s, x1:x1+s]
            
        # Resize to CONFIG size (256)
        frame_resized = cv2.resize(frame_cropped, (CONFIG["resize_size"], CONFIG["resize_size"]))
        processed_frames.append(frame_resized)
        
    # Stack -> (T, H, W, C)
    tensor_np = np.stack(processed_frames)
    
    # To Tensor (C, T, H, W)
    tensor = torch.from_numpy(tensor_np)
    tensor = tensor.permute(3, 0, 1, 2)
    
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)
        
    return tensor

def main():
    # 1. Ensure output directory exists
    if not os.path.exists(CONFIG["tensor_dir"]):
        os.makedirs(CONFIG["tensor_dir"])
        
    # 2. Load BOTH Train and Validation JSONs
    # (We need smart crops for both sets to run the experiment!)
    json_files = [CONFIG['train_json'], CONFIG['val_json']]
    all_entries = []

    print(f"Loading metadata from {len(json_files)} files...")
    for j_file in json_files:
        if os.path.exists(j_file):
            with open(j_file, 'r') as f:
                data = json.load(f)
                all_entries.extend(data)
        else:
            print(f"⚠️ Warning: JSON file not found: {j_file}")

    # 3. Filter tasks (Skip existing tensors to save time)
    tasks = []
    # Ensure VIDEO_DIR is correct. 
    # If not in CONFIG, hardcode it or use the one from your previous script.
    VIDEO_DIR = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\videos" 
    
    print(f"Checking {len(all_entries)} candidates...")
    for entry in all_entries:
        video_id = entry['video_id']
        out_path = os.path.join(CONFIG["tensor_dir"], f"{video_id}.pt")
        
        # Only add to task list if the .pt file doesn't exist yet
        if not os.path.exists(out_path):
             tasks.append(entry)
             
    print(f"🚀 Starting processing for {len(tasks)} new videos...")
    
    # 4. Initialize MediaPipe Holistic
    with mp_holistic.Holistic(
        static_image_mode=False, # False is faster for video (uses tracking)
        min_detection_confidence=0.5, 
        model_complexity=1) as holistic:
        
        success = 0
        fail = 0
        
        for task in tqdm(tasks, desc="Smart Cropping"):
            video_id = task['video_id']
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            out_path = os.path.join(CONFIG["tensor_dir"], f"{video_id}.pt")
            
            # Skip if raw video is missing
            if not os.path.exists(video_path):
                # Optional: Print warning only for first few misses
                # print(f"Missing raw video: {video_id}") 
                fail += 1
                continue
                
            try:
                # Call the processing function
                # ENSURE process_video_smart returns a uint8 tensor (0-255)
                tensor = process_video_smart(video_path, task['frame_start'], task['frame_end'], holistic)
                
                if tensor is not None:
                    torch.save(tensor, out_path)
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                fail += 1
                
    print(f"✅ Done. Success: {success}, Failed: {fail}")
    print(f"Tensors saved to: {CONFIG['tensor_dir']}")

if __name__ == "__main__":
    main()
import os
import json
import cv2
import numpy as np
import torch
import boto3
import tempfile
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()

class Settings:
    AWS_SERVER_PUBLIC_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SERVER_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

settings = Settings()

# --- CONFIGURATION ---
S3_BUCKET = "alexander-thesis-cslr"
S3_JSON_KEY = "datasets/wlasl2000/wlasl-complete/WLASL_v0.3.json"
S3_VIDEO_PREFIX = "datasets/wlasl2000/wlasl-complete/videos"

# Note: You mentioned 2000+ objects, which matches the Top 100 subset (~2000 videos).
# If you want the Top 1000 (~12k videos), change this to 1000.
NUM_CLASSES = 1000 
TARGET_FRAMES = 64
RESIZE_SIZE = 256
S3_OUTPUT_PREFIX = f"data_tensors_{NUM_CLASSES}"

# Initialize S3 Session
session = boto3.Session(
    aws_access_key_id=settings.AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=settings.AWS_SERVER_SECRET_KEY,
    region_name=settings.AWS_REGION
)
s3_client = session.client('s3')

def preprocess():
    # 1. Download Metadata from S3
    print(f"Downloading metadata from s3://{S3_BUCKET}/{S3_JSON_KEY}...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
        try:
            s3_client.download_file(S3_BUCKET, S3_JSON_KEY, tmp_json.name)
            with open(tmp_json.name, 'r') as f:
                content = json.load(f)
        except Exception as e:
            print(f"❌ CRITICAL: Could not download JSON: {e}")
            return
    os.remove(tmp_json.name)

    # 2. Select Top Glosses
    print(f"Selecting top {NUM_CLASSES} glosses...")
    gloss_counts = {entry['gloss']: len(entry['instances']) for entry in content}
    top_n = Counter(gloss_counts).most_common(NUM_CLASSES)
    top_glosses = {item[0] for item in top_n}
    
    tasks = []
    for entry in content:
        if entry['gloss'] in top_glosses:
            for inst in entry['instances']:
                tasks.append(inst)

    print(f"✅ Found {len(tasks)} videos to process.")
    
    # 3. Process Videos
    success_count = 0
    fail_count = 0
    
    for task in tqdm(tasks, desc="Processing"):
        video_id = task['video_id']
        s3_video_key = f"{S3_VIDEO_PREFIX}/{video_id}.mp4"
        s3_output_key = f"{S3_OUTPUT_PREFIX}/{video_id}.pt"
        
        # Check if already exists in S3 (Optional - Resume capability)
        # try:
        #     s3_client.head_object(Bucket=S3_BUCKET, Key=s3_output_key)
        #     success_count += 1
        #     continue
        # except:
        #     pass

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid, \
             tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_pt:
            
            try:
                # Download Video
                s3_client.download_file(S3_BUCKET, s3_video_key, tmp_vid.name)
                
                # --- USE YOUR ROBUST LOGIC HERE ---
                frames = load_and_process_video(tmp_vid.name, task['frame_start'], task['frame_end'])
                
                if frames is not None:
                    # Convert to Tensor (C, T, H, W)
                    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).to(torch.uint8)
                    torch.save(tensor, tmp_pt.name)
                    
                    # Upload Tensor
                    s3_client.upload_file(tmp_pt.name, S3_BUCKET, s3_output_key)
                    success_count += 1
                else:
                    fail_count += 1
                    # print(f"⚠️ Failed to process video content: {video_id}")
            
            except s3_client.exceptions.NoSuchKey:
                fail_count += 1
                # print(f"⚠️ Missing in S3: {s3_video_key}")
            except Exception as e:
                fail_count += 1
                print(f"❌ Error {video_id}: {e}")
            finally:
                if os.path.exists(tmp_vid.name): os.remove(tmp_vid.name)
                if os.path.exists(tmp_pt.name): os.remove(tmp_pt.name)

    print("\n---------------- SUMMARY ----------------")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed:     {fail_count}")

def load_and_process_video(path, start_frame, end_frame):
    """
    Original Robust Logic from your script.
    Handles 'Already Cropped' videos and padding correctly.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 0-based index correction
    start = max(0, start_frame - 1)
    end = end_frame
    
    # --- CRITICAL FIX FOR WLASL DATASET ---
    # Logic to handle "Already Cropped" videos
    if start >= total_frames:
        expected_len = end - start
        
        # If the video file length matches the expected segment length, assume it's pre-cropped
        if abs(total_frames - expected_len) < 15 or (total_frames > 0 and expected_len > total_frames):
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
    
    # Optimized Read Strategy from your script
    if segment_len < TARGET_FRAMES:
        # Read all available frames
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
    
    # Final Size Check & Resize
    if len(frames) < TARGET_FRAMES:
        while len(frames) < TARGET_FRAMES:
             frames.append(frames[-1])
             
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (RESIZE_SIZE, RESIZE_SIZE))
        processed_frames.append(frame)
        
    return np.stack(processed_frames)

if __name__ == "__main__":
    preprocess()
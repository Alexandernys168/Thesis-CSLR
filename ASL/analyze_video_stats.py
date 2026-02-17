import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

def analyze_videos(video_dir):
    print(f"Scanning videos in: {video_dir}")
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print("No .mp4 files found!")
        return

    print(f"Found {len(video_files)} videos.")
    
    frame_counts = []
    
    for video_path in tqdm(video_files, desc="Analyzing Frames"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
            
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames > 0:
            frame_counts.append(frames)
        cap.release()

    if not frame_counts:
        print("Could not read frame counts from any videos.")
        return

    frame_counts = np.array(frame_counts)
    
    print("\n--- Video Frame Statistics ---")
    print(f"Total Videos Processed: {len(frame_counts)}")
    print(f"Mean Frames:   {np.mean(frame_counts):.2f}")
    print(f"Median Frames: {np.median(frame_counts):.2f}")
    print(f"Min Frames:    {np.min(frame_counts)}")
    print(f"Max Frames:    {np.max(frame_counts)}")
    print(f"Std Dev:       {np.std(frame_counts):.2f}")
    
    # Percentiles
    print(f"25th Percentile: {np.percentile(frame_counts, 25):.2f}")
    print(f"75th Percentile: {np.percentile(frame_counts, 75):.2f}")
    print(f"90th Percentile: {np.percentile(frame_counts, 90):.2f}")
    print(f"95th Percentile: {np.percentile(frame_counts, 95):.2f}")

    print("\n--- Distribution Buckets ---")
    buckets = [0, 30, 48, 64, 100, 200, 1000]
    labels = ["<30", "30-48", "48-64", "64-100", "100-200", ">200"]
    
    hist, _ = np.histogram(frame_counts, bins=buckets)
    
    for label, count in zip(labels, hist):
        percentage = (count / len(frame_counts)) * 100
        print(f"{label.ljust(10)}: {count:5d} ({percentage:.1f}%)")

if __name__ == "__main__":
    VIDEO_DIR = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\videos"
    analyze_videos(VIDEO_DIR)

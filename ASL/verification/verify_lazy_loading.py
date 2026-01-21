from config import CONFIG
from dataset import FastWLASLDataset
import os
import torch

def verify_dataset():
    print("--- Verifying Dataset Loading Modes ---")
    
    # 1. Test Preprocessed Mode (Default)
    print("\n[Test 1] Preprocessed Mode")
    CONFIG["load_mode"] = "preprocessed"
    try:
        ds = FastWLASLDataset(CONFIG['train_json'], CONFIG['tensor_dir'], augment=False)
        print(f"Dataset initialized with {len(ds)} samples.")
        # Try fetching one item if possible
        if len(ds) > 0:
            # Depending on if data exists, this might fail or succeed. 
            # We just want to check if logic flows until file access.
            try:
                x, y = ds[0]
                print(f"Sample 0 shape: {x.shape}, Label: {y}")
            except FileNotFoundError:
                print("Sample 0 file not found (expected if data not processed).")
            except Exception as e:
                print(f"Error fetching sample: {e}")
    except Exception as e:
        print(f"Failed to init dataset: {e}")

    # 2. Test On-the-Fly Mode
    print("\n[Test 2] On-the-Fly Mode")
    CONFIG["load_mode"] = "on_the_fly"
    # Set a dummy video dir
    CONFIG["video_dir"] = r"a:\Thesis-CSLR\ASL\videos"
    
    try:
        ds = FastWLASLDataset(CONFIG['train_json'], CONFIG['tensor_dir'], augment=False)
        print(f"Dataset initialized with {len(ds)} samples.")
        
        # We probably can't fetch a video if files don't exist, but we can check if it tries to load using cv2/torchvision logic
        # without crashing on imports.
        import cv2
        print("cv2 imported successfully.")
        
    except Exception as e:
        print(f"Failed during on-the-fly check: {e}")

if __name__ == "__main__":
    verify_dataset()

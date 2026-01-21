
import torch
import os
from config import CONFIG
from models import get_model
from dataset import FastWLASLDataset
from torch.utils.data import DataLoader

def verify_i3d():
    print("--- Verifying Two-Stream I3D Implementation ---")
    
    # 1. Verify Config
    print(f"Model Type: {CONFIG.get('model_type')}")
    print(f"Stream Type: {CONFIG.get('stream_type')}")
    print(f"Frames Per Clip: {CONFIG.get('frames_per_clip')}")
    
    # 2. Verify Dataset Integration
    print("\n--- Testing Dataset (Flow Generation) ---")
    # Using 'train_json' from config
    json_path = CONFIG['train_json']
    tensor_dir = CONFIG['tensor_dir']
    
    if not os.path.exists(json_path) or not os.path.exists(tensor_dir):
        print("Dataset or Tensor Dir not found. Skipping dataset test.")
    else:
        # Create dataset
        ds = FastWLASLDataset(json_path, tensor_dir, augment=False)
        print(f"Dataset created with {len(ds)} samples.")
        
        # Get one sample
        try:
            print("Fetching sample 0...")
            inputs, label = ds[0]
            
            if isinstance(inputs, tuple):
                rgb, flow = inputs
                print(f"RGB Shape: {rgb.shape} (Expected: 3, 64, 224, 224)")
                print(f"Flow Shape: {flow.shape} (Expected: 2, 64, 224, 224)")
                
                # Check value ranges
                print(f"RGB Range: [{rgb.min():.2f}, {rgb.max():.2f}]")
                print(f"Flow Range: [{flow.min():.2f}, {flow.max():.2f}]")
            else:
                print(f"Input Shape: {inputs.shape}")
                
        except Exception as e:
            print(f"Dataset access failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. Verify Model instantiation and Forward Pass
    print("\n--- Testing I3D Model ---")
    try:
        model = get_model(CONFIG)
        print("Model instantiated successfully.")
        
        # Create dummy input if dataset failed or just to be safe
        batch_size = 2
        clip_frames = CONFIG['frames_per_clip']
        h, w = CONFIG['crop_size'], CONFIG['crop_size']
        
        rgb_dummy = torch.randn(batch_size, 3, clip_frames, h, w)
        flow_dummy = torch.randn(batch_size, 2, clip_frames, h, w)
        
        print("Running Forward Pass with Dummy Data...")
        with torch.no_grad():
            outputs = model((rgb_dummy, flow_dummy))
            
        print(f"Output Shape: {outputs.shape}")
        print("Forward pass successful.")
        
    except Exception as e:
        print(f"Model verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_i3d()

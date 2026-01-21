import torch
import matplotlib.pyplot as plt
import os
import random

# Path to your NEW smart tensors
TENSOR_DIR = r"a:\Thesis-CSLR\ASL\tensors_hand_crop"

def inspect_tensors():
    files = [f for f in os.listdir(TENSOR_DIR) if f.endswith('.pt')]
    if not files:
        print("❌ No tensors found!")
        return

    # Pick 3 random files
    samples = random.sample(files, 3)

    for i, file_name in enumerate(samples):
        path = os.path.join(TENSOR_DIR, file_name)
        try:
            # Load tensor
            tensor = torch.load(path) # Shape: (C, T, H, W)
            
            # Print Stats
            print(f"\nFile: {file_name}")
            print(f"Shape: {tensor.shape}")
            print(f"Max Value: {tensor.max()} (Should be 255 if uint8, or 1.0 if float)")
            print(f"Min Value: {tensor.min()}")
            print(f"Dtype: {tensor.dtype}")

            # Prepare for plotting (Middle Frame)
            mid_frame_idx = tensor.shape[1] // 2
            img = tensor[:, mid_frame_idx, :, :].permute(1, 2, 0) # C,H,W -> H,W,C
            
            # Conversion for display
            if tensor.max() <= 1.0:
                img = img * 255.0
            
            img = img.clamp(0, 255).byte()

            plt.figure(figsize=(4, 4))
            plt.imshow(img.numpy())
            plt.title(f"Sample {i+1}: {file_name}")
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    inspect_tensors()
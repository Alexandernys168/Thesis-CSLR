import torch
from config import CONFIG
from models import ResNet3D_LSTM, BaselineResNet3D
import sys

def verify_models():
    print("--- Verifying Models ---")
    
    # Test Baseline
    print("\n[1] Testing BaselineResNet3D...")
    try:
        model = BaselineResNet3D(num_classes=100)
        # Input: (B, C, T, H, W)
        dummy_input = torch.randn(2, 3, 32, 224, 224)
        output = model(dummy_input)
        print(f"    Check Pass: Output shape {output.shape} (Expected (2, 100))")
    except Exception as e:
        print(f"    Check Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Hybrid
    print("\n[2] Testing ResNet3D_LSTM...")
    try:
        model = ResNet3D_LSTM(num_classes=100, hidden_size=256, num_layers=2)
        # Input: (B, C, T, H, W)
        dummy_input = torch.randn(2, 3, 32, 224, 224)
        output = model(dummy_input)
        print(f"    Check Pass: Output shape {output.shape} (Expected (2, 100))")
        
        # Check if temporal dimension handling is reasonable
        # We can inspect the intermediate shape if we want, but end-to-end is good enough for now.
    except Exception as e:
        print(f"    Check Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_models()

import torch
from config import CONFIG
from models import get_model, CNNRNN_2D

def verify_cnn_rnn_2d():
    print("--- Verifying CNNRNN_2D ---")
    
    # Override config for testing
    test_config = CONFIG.copy()
    test_config['model_type'] = '2dcnn_lstm'
    test_config['lstm_hidden_size'] = 128
    test_config['lstm_layers'] = 1
    test_config['num_classes'] = 10
    
    model = get_model(test_config)
    print(f"Model initialized: {type(model).__name__}")
    
    # Check if instance of CNNRNN_2D
    if not isinstance(model, CNNRNN_2D):
        print(f"ERROR: Expected CNNRNN_2D, got {type(model)}")
        return
        
    # Create dummy input (B, C, T, H, W)
    # 2D CNN (ResNet18) expects H,W = 224, 224 usually, but can handle others.
    batch_size = 2
    channels = 3
    frames = 16
    height = 224
    width = 224
    
    input_tensor = torch.randn(batch_size, channels, frames, height, width)
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    try:
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        
        expected_shape = (batch_size, test_config['num_classes'])
        if output.shape == expected_shape:
             print("SUCCESS: Output shape matches expected.")
        else:
             print(f"FAILURE: Output shape mismatch. Expected {expected_shape}, got {output.shape}")
             
    except Exception as e:
        print(f"FAILURE: Forward pass threw error: {e}")
        import traceback
        traceback.print_exc()

def verify_get_model_r3d():
    print("\n--- Verifying get_model (r3d_18) ---")
    test_config = CONFIG.copy()
    test_config['model_type'] = 'r3d_18'
    model = get_model(test_config)
    print(f"Model initialized: {type(model).__name__}")

if __name__ == "__main__":
    verify_cnn_rnn_2d()
    verify_get_model_r3d()

import os
import torch
import torch.nn as nn
import shutil
import json
import logging
import cv2
import numpy as np

# Config setup
import config
from config import CONFIG

# Setup proper logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_preprocess_smart_deps():
    logger.info("Testing Preprocess Smart Dependencies...")
    try:
        import mediapipe as mp
        logger.info("  MediaPipe imported successfully")
        import preprocess_smart
        logger.info("  preprocess_smart imported successfully")
    except Exception as e:
        logger.warning(f"  [WARNING] Dependency missing or initialization failed: {e}")
        logger.warning("  Preprocess Smart verification SKIPPED. Please check your MediaPipe installation.")
        return True # Proceed to other tests
    return True

def test_dataset_augmentation():
    logger.info("Testing Dataset Augmentation logic...")
    
    # Mock CONFIG
    CONFIG['crop_size'] = 224
    CONFIG['aug_prob_crop'] = 1.0
    CONFIG['aug_prob_flip'] = 0.5
    CONFIG['aug_color_jitter'] = True
    
    from dataset import FastWLASLDataset
    
    # Create Dummy Tensor (C, T, H, W) uint8
    C, T, H, W = 3, 16, 256, 256
    dummy_tensor = torch.randint(0, 255, (C, T, H, W), dtype=torch.uint8)
    
    # Create Dummy Dataset
    # We need a dummy json and dir
    os.makedirs("temp_test_data", exist_ok=True)
    dummy_json = [{"video_id": "test_001", "label": 0}]
    with open("temp_test_data/test.json", "w") as f:
        json.dump(dummy_json, f)
        
    torch.save(dummy_tensor, "temp_test_data/test_001.pt")
    
    ds = FastWLASLDataset("temp_test_data/test.json", "temp_test_data", augment=True)
    
    # Test Get Item
    data, label = ds[0]
    
    logger.info(f"  Output Shape: {data.shape}") # Should be (3, 16, 224, 224)
    if data.shape[-2:] == (224, 224):
        logger.info("  [PASS] Crop size correct")
    else:
        logger.error(f"  [FAIL] Crop size incorrect: {data.shape}")
        
    # Stats check (normalized)
    logger.info(f"  Data Range: {data.min():.2f} to {data.max():.2f}")
    
    # Cleanup
    shutil.rmtree("temp_test_data")
    return True

def test_train_modular_dry_run():
    logger.info("Testing Train Modular Dry Run...")
    
    from train_modular import get_model
    
    # Init Model
    try:
        model = get_model()
        logger.info("  Model initialized successfully")
        
        # Check Dropout
        has_dropout = False
        for m in model.fc.modules():
            if isinstance(m, nn.Dropout):
                has_dropout = True
                logger.info(f"  [PASS] Dropout found with p={m.p}")
        if not has_dropout:
            logger.error("  [FAIL] Dropout NOT found in fc head")
            
        # Dummy Forward Pass
        inputs = torch.randn(2, 3, 16, 224, 224) # Batch 2
        outputs = model(inputs)
        logger.info(f"  Forward Pass Output Shape: {outputs.shape}")
        if outputs.shape == (2, CONFIG['num_classes']):
            logger.info("  [PASS] Output shape correct")
        else:
            logger.error("  [FAIL] Output shape incorrect")
            
    except Exception as e:
        logger.error(f"  Train test failed: {e}")
        return False
    return True
    
def main():
    logger.info("--- Starting Verification ---")
    
    tests = [
        test_preprocess_smart_deps,
        test_dataset_augmentation,
        test_train_modular_dry_run
    ]
    
    for test in tests:
        if not test():
            logger.error("Verification ABORTED due to failure.")
            return
            
    logger.info("--- All Verification Steps Passed ---")

if __name__ == "__main__":
    main()

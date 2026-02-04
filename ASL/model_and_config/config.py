import os

# -------------------------------------------------------------------
#   Path Setup
# -------------------------------------------------------------------
# Robust Root Calculation
# Assumes this file is at: Project/ASL/model_and_config/config.py
# 1 up: ASL/model_and_config
# 2 up: ASL
# 3 up: Project (Root)
current_file_path = os.path.abspath(__file__)
ASL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

NUM_CLASSES = 1000
S3_BUCKET = 'alexander-thesis-cslr'
# The folder inside the bucket
S3_PREFIX = f"data_tensors_{NUM_CLASSES}" 

# Local Cache Path (Where we download to)
LOCAL_DATA_CACHE = os.path.join(ASL_ROOT, "data_cache", f"data_tensors_{NUM_CLASSES}")


CONFIG = {

    # --- S3 Settings (Exported for Main script) ---
    "s3_bucket": S3_BUCKET,
    "s3_prefix": S3_PREFIX,

    # Paths
    "train_json": os.path.join(ASL_ROOT, "ASL", f"train_{NUM_CLASSES}.json"),
    "val_json": os.path.join(ASL_ROOT, "ASL", f"val_{NUM_CLASSES}.json"),
    # Default output directory for smart tensor processing
    "tensor_dir": LOCAL_DATA_CACHE, 
    "checkpoint_dir": os.path.join(ASL_ROOT, "ASL", "checkpoints"),
    "log_file": os.path.join(ASL_ROOT, "ASL/logging", "experiment_logs.csv"),
    "extended_log_file": os.path.join(ASL_ROOT, "ASL/logging", "experiment_logs_detailed.csv"),
    
    # Model Configuration
    # Model Configuration
    "model_type": "2dcnn_lstm", # Options: "r3d_18", "r3d_lstm", "2dcnn_lstm", "r3d_attention"
    "num_classes": NUM_CLASSES,
    "pretrained": True,
    "dropout_prob": 0.5,
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    
    # Data Configuration
    "load_mode": "preprocessed", # Options: "preprocessed" (fast, cached) or "on_the_fly" (slow, saves disk)
    "stream_type": "rgb", # Options: "rgb", "flow", "two_stream"
    "video_dir": os.path.join(ASL_ROOT, "1", "wlasl-complete", "videos"), # Path to raw videos (for 'on_the_fly')
    "use_hand_crop": False, # If True, assumes tensors are hand-cropped
    "frames_per_clip": 64,
    "resize_size": 256, # Resize frames to this size
    "crop_size": 224,   # Input size to model
    
    # Augmentation
    "augment": True,
    "aug_prob_flip": 0.5,
    "aug_prob_crop": 1.0, # Always random crop during training if augment=True
    "aug_color_jitter": True,
    "aug_rotation_range": 15,
    "aug_erase_prob": 0.2,
    
    # Training Hyperparameters
    "use_lr_scheduler": True, 
    "batch_size": 5, # I3D is heavy, possibly reduce batch size
    "epochs": 50,
    "learning_rate": 1e-4, # I3D usually likes lower LR or SGD
    "weight_decay": 1e-4, # Optional regularization
    "clip_grad_norm": 1.0,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    
    # Run Metadata
    "config_id": f"2DCNN_LSTM_v2_WITH_{NUM_CLASSES}", # Tag for the experiment log
}

# Ensure directories exist
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

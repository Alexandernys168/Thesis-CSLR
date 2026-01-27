import os

# Get the absolute path to the ASL directory (2 levels up from this file)
ASL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Paths
    "train_json": os.path.join(ASL_ROOT, "train_1000.json"),
    "val_json": os.path.join(ASL_ROOT, "val_1000.json"),
    # Default output directory for smart tensor processing
    "tensor_dir": os.path.join(ASL_ROOT, "data_tensors_1000"), 
    "checkpoint_dir": os.path.join(ASL_ROOT, "checkpoints"),
    "log_file": os.path.join(ASL_ROOT, "experiment_logs.csv"),
    
    # Model Configuration
    # Model Configuration
    "model_type": "r3d_18", # Options: "r3d_18", "r3d_lstm", "2dcnn_lstm", "i3d_rgb", "i3d_flow", "i3d_two_stream"
    "num_classes": 1000,
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
    "epochs": 100,
    "learning_rate": 1e-4, # I3D usually likes lower LR or SGD
    "weight_decay": 1e-4, # Optional regularization
    "clip_grad_norm": 1.0,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    
    # Run Metadata
    "config_id": "r3d_18_v1_1000", # Tag for the experiment log
}

# Ensure directories exist
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

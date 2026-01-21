import os

CONFIG = {
    # Paths
    "train_json": r"a:\Thesis-CSLR\ASL\train_100.json",
    "val_json": r"a:\Thesis-CSLR\ASL\val_100.json",
    # Default output directory for smart tensor processing
    "tensor_dir": r"a:\Thesis-CSLR\ASL\data_tensors", # Alternative: r"a:\Thesis-CSLR\ASL\tensors_hand_crop"
    "checkpoint_dir": r"a:\Thesis-CSLR\ASL\checkpoints",
    "log_file": r"a:\Thesis-CSLR\ASL\experiment_logs.csv",
    
    # Model Configuration
    # Model Configuration
    "model_type": "i3d_two_stream", # Options: "r3d_18", "r3d_lstm", "2dcnn_lstm", "i3d_rgb", "i3d_flow", "i3d_two_stream"
    "num_classes": 100,
    "pretrained": True,
    "dropout_prob": 0.5,
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    
    # Data Configuration
    "load_mode": "preprocessed", # Options: "preprocessed" (fast, cached) or "on_the_fly" (slow, saves disk)
    "stream_type": "two_stream", # Options: "rgb", "flow", "two_stream"
    "video_dir": r"a:\Thesis-CSLR\ASL\1\wlasl-complete\videos", # Path to raw videos (for 'on_the_fly')
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
    "batch_size": 2, # I3D is heavy, possibly reduce batch size
    "epochs": 50,
    "learning_rate": 1e-4, # I3D usually likes lower LR or SGD
    "weight_decay": 1e-4, # Optional regularization
    "clip_grad_norm": 1.0,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    
    # Run Metadata
    "config_id": "i3d_two_stream_v1", # Tag for the experiment log
}

# Ensure directories exist
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

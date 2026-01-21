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
    "model_type": "2dcnn_lstm", # Options: "r3d_18", "r3d_lstm", "2dcnn_lstm"
    "num_classes": 100,
    "pretrained": True,
    "dropout_prob": 0.5,
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    
    # Data Configuration
    "load_mode": "preprocessed", # Options: "preprocessed" (fast, cached) or "on_the_fly" (slow, saves disk)
    "video_dir": r"a:\Thesis-CSLR\ASL\videos", # Path to raw videos (for 'on_the_fly')
    "use_hand_crop": False, # If True, assumes tensors are hand-cropped
    "frames_per_clip": 64,
    "resize_size": 256,
    "crop_size": 224,
    
    # Augmentation
    "augment": True,
    "aug_prob_flip": 0.5,
    "aug_prob_crop": 1.0, # Always random crop during training if augment=True
    "aug_color_jitter": True,
    "aug_rotation_range": 15,
    "aug_erase_prob": 0.2,
    
    # Training Hyperparameters
    "use_lr_scheduler": True,
    "batch_size": 7, # Keep low for VRAM, but accumulation handles effective batch
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4, # Optional regularization
    "clip_grad_norm": 1.0,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    
    # Run Metadata
    "config_id": "2dcnn_lstm_hard_augmentations_no_hand_crop_v1", # Tag for the experiment log
}

# Ensure directories exist
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

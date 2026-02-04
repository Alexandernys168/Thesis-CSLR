import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from dotenv import load_dotenv
import boto3
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------------
#   1. Environment & Path Setup (MUST BE FIRST)
# -------------------------------------------------------------------

# Assuming this script is at: Project/scripts/training/script.py
# We go up 2 levels to find the Root
current_dir = os.path.dirname(os.path.abspath(__file__))
ASL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
if ASL_ROOT not in sys.path:
    sys.path.append(ASL_ROOT)

# Load Environment Variables Explicitly
# Assumes .env is in the Project Root (ASL_ROOT)
env_path = os.path.join(ASL_ROOT, ".env")
loaded = load_dotenv(env_path)

if not loaded:
    # Try looking one level up just in case
    env_path = os.path.join(ASL_ROOT, "..", ".env")
    loaded = load_dotenv(env_path)

if not loaded:
    print(f"[WARNING] .env file not found. Ensure AWS keys are set.")

# -------------------------------------------------------------------
#   2. Imports (AFTER Environment Setup)
# -------------------------------------------------------------------

try:
    # This import will now work safely because Config no longer connects to AWS
    from model_and_config.config import CONFIG 
    from logger import ExperimentLogger
    from data_and_preprocess.dataset import FastWLASLDataset
    from model_and_config.models import get_model
except ImportError as e:
    print("Could not import ASL modules. Check your python path.")
    raise e

# -------------------------------------------------------------------
#   3. AWS Authentication
# -------------------------------------------------------------------

try:
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    # Create the S3 client here to be used by the downloader
    s3_client_global = session.client('s3')
    print("AWS Session initialized successfully.")
except Exception as e:
    print(f"[ERROR] AWS Credentials missing or invalid: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
#   4. Robust Boto3 Downloader
# -------------------------------------------------------------------

def download_one_file(bucket, key, local_path, client):
    """Helper to download a single file if it doesn't exist."""
    if os.path.exists(local_path):
        return 
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(bucket, key, local_path)
    except Exception as e:
        print(f"Failed to download {key}: {e}")

def sync_s3_data_python(bucket_name, prefix, local_dir):
    """
    Downloads a folder from S3 using pure Python and threads.
    """
    print(f"\n--- S3 Download Started ---")
    print(f"Bucket: {bucket_name}")
    print(f"Prefix: {prefix}")
    print(f"Local:  {local_dir}")
    
    # Use the global client to list objects
    paginator = s3_client_global.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    download_list = []
    print("Scanning S3 bucket for files...")
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Calculate local path relative to the prefix
                # If key is "data/file.pt" and prefix is "data", rel is "file.pt"
                # If we want the full structure, we might need to adjust logic based on your bucket
                # This logic assumes we map S3:Prefix/... -> Local:Dir/...
                rel_path = os.path.relpath(key, prefix)
                local_file_path = os.path.join(local_dir, rel_path)
                
                if not os.path.exists(local_file_path):
                    download_list.append((bucket_name, key, local_file_path))
    
    if not download_list:
        print("Folder is up to date.")
        return

    print(f"Downloading {len(download_list)} files...")
    
    # Download in Parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(download_one_file, b, k, l, s3_client_global) 
            for b, k, l in download_list
        ]
        for _ in tqdm(range(len(futures)), desc="Downloading"):
            futures.pop(0).result()
            
    print("--- Download Complete ---\n")

# -------------------------------------------------------------------
#   5. Metrics & Training (Standard)
# -------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        num_classes = output.size(1)
        maxk = min(maxk, num_classes)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k > num_classes:
                res.append(0.0) 
                continue
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top10 = AverageMeter('Acc@10', ':6.2f')
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch+1}", unit="batch")
    
    for inputs, labels in pbar:
        
        inputs = inputs.to(device, non_blocking=True)
             
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed Precision
        with torch.amp.autocast('cuda', enabled=True):
             outputs = model(inputs)
             loss = criterion(outputs, labels)
             
        # Backward & Step
        scaler.scale(loss).backward()
        
        # Gradient Clipping (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['clip_grad_norm'])
        
        scaler.step(optimizer)
        scaler.update()
        
        # Stats
        acc1, acc5, acc10 = accuracy(outputs, labels, topk=(1, 5, 10))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1, inputs.size(0))
        top5.update(acc5, inputs.size(0))
        top10.update(acc10, inputs.size(0))
        
        pbar.set_postfix({'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg})
        
    return losses.avg, top1.avg, top5.avg, top10.avg

def validate(model, loader, criterion, device, epoch):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top10 = AverageMeter('Acc@10', ':6.2f')
    
    pbar = tqdm(loader, desc=f"Val Ep {epoch+1}", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            
            inputs = inputs.to(device, non_blocking=True)
                 
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            acc1, acc5, acc10 = accuracy(outputs, labels, topk=(1, 5, 10))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
            top5.update(acc5, inputs.size(0))
            top10.update(acc10, inputs.size(0))
            
            pbar.set_postfix({'val_loss': losses.avg, 'val_top1': top1.avg, 'val_top5': top5.avg})
            
    return losses.avg, top1.avg, top5.avg, top10.avg

def main():
    # Setup Directories
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

    # --- S3 DOWNLOAD STEP ---
    # We now use the values loaded from CONFIG
    try:
        sync_s3_data_python(
            CONFIG['s3_bucket'], 
            CONFIG['s3_prefix'], 
            CONFIG['tensor_dir']
        )
    except Exception as e:
        print(f"Critical Error downloading data: {e}")
        return


    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Logger
    logger = ExperimentLogger(CONFIG['log_file'], CONFIG.get('extended_log_file'))
    
    # Requirement 2: Save Config Snapshot
    logger.save_config_snapshot(CONFIG)
    
    # Setup Dataset
    # Verify Tensor Dir
    if not os.path.exists(CONFIG['tensor_dir']):
        print(f"[WARNING] Tensor directory {CONFIG['tensor_dir']} does not exist.")
        return

    print("Loading Datasets...")
    # Train: Augmentations ON (if enabled in config)
    train_ds = FastWLASLDataset(
        CONFIG['train_json'], 
        CONFIG['tensor_dir'], 
        augment=CONFIG['augment'],
        num_classes=CONFIG.get('num_classes')
    )
    # Val: Augmentations OFF
    val_ds = FastWLASLDataset(
        CONFIG['val_json'], 
        CONFIG['tensor_dir'], 
        augment=False,
        num_classes=CONFIG.get('num_classes')
    )
    
    
    # Recommended settings for High-Performance Training
    pin_memory = True
    num_workers = CONFIG.get('num_workers', 4) # Allow config override
    persistent_workers = True
    
    print(f"DataLoader Config: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    print(f"Training on {len(train_ds)} samples. Validating on {len(val_ds)} samples.")
    
    # Model
    model = get_model(CONFIG).to(device)
    
    # Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # Requirement 1: Adaptive Learning Rate Scheduler
    scheduler = None
    if CONFIG.get("use_lr_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,  
            min_lr=1e-6
        )
        print("Initialized ReduceLROnPlateau scheduler.")
    
    # Loop
    best_acc = 0.0
    
    # Early Stopping State
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = CONFIG.get('early_stopping_patience', 10)
    min_delta = CONFIG.get('early_stopping_min_delta', 0.001)
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        train_loss, train_acc1, train_acc5, train_acc10 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc1, val_acc5, val_acc10 = validate(model, val_loader, criterion, device, epoch)
        
        # Scheduler Step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
             scheduler.step(val_loss)
             # Update current_lr in case it changed
             current_lr = optimizer.param_groups[0]['lr']

        # Logging
        # Passing full tuples of (loss, top1, top5, top10) to logger
        logger.log_epoch(CONFIG, epoch, 
                         (train_loss, train_acc1, train_acc5, train_acc10), 
                         (val_loss, val_acc1, val_acc5, val_acc10), 
                         best_acc, current_lr)
        
        print(f"Summary Ep {epoch+1}: Train Loss {train_loss:.4f} Top1 {train_acc1:.2f}% | Val Loss {val_loss:.4f} Top1 {val_acc1:.2f}%")
        
        # Save Best (based on Top-1 Accuracy)
        if val_acc1 > best_acc:
            print(f"New Best Accuracy! ({best_acc:.2f} -> {val_acc1:.2f})")
            best_acc = val_acc1
            save_path = os.path.join(CONFIG['checkpoint_dir'], f"{CONFIG['config_id']}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            
        # Early Stopping Logic (Monitoring Val Loss)
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early Stopping Counter: {early_stop_counter}/{patience} (Best Loss: {best_val_loss:.4f})")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
    print(f"\nTraining Complete. Best Validation Top-1 Accuracy: {best_acc:.2f}")

if __name__ == "__main__":
    main()

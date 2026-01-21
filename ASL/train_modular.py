import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler

# Modular Imports
from config import CONFIG
from logger import ExperimentLogger
from dataset import FastWLASLDataset
from models import get_model



def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch+1}", unit="batch")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
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
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
        
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Val Ep {epoch+1}", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'val_loss': loss.item(), 'val_acc': correct/total if total > 0 else 0})
            
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Logger
    logger = ExperimentLogger(CONFIG['log_file'])
    
    # Requirement 2: Save Config Snapshot
    logger.save_config_snapshot(CONFIG)
    
    # Setup Dataset
    # Verify Tensor Dir
    if not os.path.exists(CONFIG['tensor_dir']):
        print(f"[WARNING] Tensor directory {CONFIG['tensor_dir']} does not exist.")
        print("Please run preprocess_smart.py first.")
        # We might continue if it's just checking, but usually we stop.
        # However, for 'verification', maybe we want to run anyway? 
        # No, can't train without data.
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
    
    # Docker / AWS Optimization
    in_docker = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", "false").lower() == "true"
    is_linux = os.name == 'posix'
    
    # Recommended settings for High-Performance Training
    pin_memory = True if (in_docker or is_linux) else False
    num_workers = CONFIG.get('num_workers', 4) # Allow config override
    persistent_workers = True if (num_workers > 0 and (in_docker or is_linux)) else False
    
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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Scheduler Step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
             scheduler.step(val_loss)
             # Update current_lr in case it changed
             current_lr = optimizer.param_groups[0]['lr']

        # Logging
        logger.log_epoch(CONFIG, epoch, (train_loss, train_acc), (val_loss, val_acc), best_acc, current_lr)
        
        print(f"Summary Ep {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
        
        # Save Best (based on Accuracy)
        if val_acc > best_acc:
            print(f"New Best Accuracy! ({best_acc:.4f} -> {val_acc:.4f})")
            best_acc = val_acc
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
            
    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()

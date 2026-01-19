import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from dataset import FastWLASLDataset

# Configuration
TRAIN_JSON = r"a:\Thesis-CSLR\ASL\train_100.json"
VAL_JSON = r"a:\Thesis-CSLR\ASL\val_100.json"
TENSOR_DIR = r"a:\Thesis-CSLR\ASL\data_tensors"
BATCH_SIZE = 4 # Increased for AMP and Uint8 inputs
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 100
CHECKPOINT_PATH = "wlasl100_fast_pretrained.pth"

def get_model(num_classes):
    # Load pretrained R3D-18
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    
    # Modify the classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", unit="batch")
    
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # AMP Context
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 1. Scale Loss & Backward
        scaler.scale(loss).backward()
        
        # 2. Unscale Gradients (Crucial step for clipping with AMP)
        scaler.unscale_(optimizer)

        # 3. Clip Gradients (The Fix for Exploding Loss)
        # Caps the gradient vector norm to 1.0, preventing massive jumps
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4. Optimizer Step
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
            
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validating", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # AMP inference (optional but faster)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
            
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing Fast Datasets...")
    # Check if preprocessing is done (or at least started)
    if not os.path.exists(TENSOR_DIR):
        print("Tensor directory not found! Run preprocess.py first.")
        return

    train_ds = FastWLASLDataset(TRAIN_JSON, TENSOR_DIR)
    val_ds = FastWLASLDataset(VAL_JSON, TENSOR_DIR)
    
    # High-performance DataLoader settings
    # pin_memory=True causes "resource already mapped" errors on some Windows/CUDA configs
    # Disabling it for stability.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    print("Initializing Model...")
    model = get_model(NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') # For mixed precision
    
    best_acc = 0.0
    
    print("Starting High-Speed Training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_dur = end_time - start_time
        
        print(f"  Duration: {epoch_dur:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            print(f"  Validation Accuracy improved from {best_acc:.4f} to {val_acc:.4f}. Saving...")
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            
    print("Training Complete.")
    print(f"Best Verification Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()

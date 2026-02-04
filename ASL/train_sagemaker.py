import sys
import os
import argparse
import torch
import time
import torch.nn as nn
import torchvision.transforms.functional as F_vis
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from logger import ExperimentLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import tarfile

# --- SageMaker Setup ---
# Add current directory to path so we can import ASL modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Modular Imports
try:
    from model_and_config.config import CONFIG as BASE_CONFIG
    from data_and_preprocess.dataset import FastWLASLDataset
    from model_and_config.models import get_model
except ImportError as e:
    print(f"Import Error: {e}")
    raise e

# -------------------------------------------------------------------
#   Metrics & Utilities (Preserved)
# -------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res



# -------------------------------------------------------------------
# GPU Augmentations Calculations (instead for CPU)
# -------------------------------------------------------------------
class GPUAugment(nn.Module):
    """
    Runs extremely fast augmentations on the GPU.
    Expects input: (Batch, Channel, Time, Height, Width)
    """
    def __init__(self, augment=True):
        super().__init__()
        self.augment = augment

        self.rot_range = 15

        # Kinetics-400 Normalization Stats (RGB)
        # We reshape them to (1, 3, 1, 1, 1) for broadcasting over (B, C, T, H, W)
        self.register_buffer('mean', torch.tensor([0.4321, 0.3946, 0.3764]).view(1, 3, 1, 1, 1))
        self.register_buffer('std',  torch.tensor([0.2280, 0.2214, 0.2169]).view(1, 3, 1, 1, 1))


    def forward(self, x):
        # Input: (B, C, T, H, W) uint8 0-255
        
        # 1. SMART CHECK: Do we need to divide by 255?
        # If it's already float and max value is small (<= 1.0), it's already normalized.
        if x.dtype == torch.float32 and x.max() <= 1.0:
            pass 
        else:
            # It's uint8 or float-0-255. Divide it.
            x = x.float().div(255.0)

        if not self.augment:
            # Just Normalize and return
            x = x.sub_(self.mean).div_(self.std)
            return x

        # 1. Random Horizontal Flip (Batch-wise is fast)
        # We apply to the whole batch with 50% prob, or per-item
        if torch.rand(1) < 0.5:
             x = torch.flip(x, [-1]) # Flip Width dimension



        # 3. Color Jitter (Simplified)
        # Apply simple random brightness/contrast scaling
        # Brightness
        if torch.rand(1) < 0.5:
            factor = torch.rand(1, device=x.device) * 0.4 + 0.8 # 0.8 to 1.2
            x = x * factor
        
        # 4. Random Erase (Optional - Fast on GPU)
        if torch.rand(1) < 0.2:
            # Create a block mask
            B, C, T, H, W = x.shape
            mask_h = int(H * 0.2)
            mask_w = int(W * 0.2)
            y = torch.randint(0, H - mask_h, (1,))
            x_loc = torch.randint(0, W - mask_w, (1,))
            x[..., y:y+mask_h, x_loc:x_loc+mask_w] = 0.0

        # 5. Random Rotation

        if self.rot_range > 0:
            angle = random.uniform(-self.rot_range, self.rot_range)
            
            # SAVE SHAPE
            B, C, T, H, W = x.shape
            
            # FOLD TIME INTO CHANNELS: (B, C, T, H, W) -> (B, C*T, H, W)
            # This makes PyTorch treat the frames as "extra channels" of a 2D image
            x = x.view(B, C * T, H, W) 
            
            # ROTATE (Now it works because input is 4D)
            x = F_vis.rotate(x, angle)
            
            # UNFOLD BACK: (B, C*T, H, W) -> (B, C, T, H, W)
            x = x.view(B, C, T, H, W)   

        x = x.sub_(self.mean).div_(self.std)
        return x
# -------------------------------------------------------------------
#   Training Functions
# -------------------------------------------------------------------




def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, model_type):

    # -----------------------------------------------------------
    # 1. SETUP ACCUMULATION
    # Get physical batch size from loader
    physical_bs = loader.batch_size
    # Target Effective Batch Size (Standardize this to 64 or 66)

    if model_type == 'r3d_18':
        target_bs = 128
    elif model_type == '2dcnn_lstm':
        target_bs = 96
    elif model_type == 'r3d_attention':
        target_bs = 66
    else:
        target_bs = 66
    # Calculate steps dynamically (e.g., 66 // 11 = 6 steps)
    accum_steps = max(1, target_bs // physical_bs)
    # -----------------------------------------------------------
    # Calculate Log Interval (1/4th of the epoch)
    total_batches = len(loader)
    # Ensure we don't divide by zero if loader is tiny
    log_interval = max(1, total_batches // 4)
    # -----------------------------------------------------------

    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # 1. Add this line
    top10 = AverageMeter('Acc@10', ':6.2f') 
    
    gpu_transform = GPUAugment(augment=True).to(device)

    start_time = time.time()

    #### pbar = tqdm(loader, desc=f"Train Ep {epoch+1}", unit="batch", file=sys.stdout, mininterval=5.0)
    
    
    for i,  (inputs, labels) in enumerate(loader):
       

        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Apply Transforms ON GPU (Instant)
        with torch.no_grad():
            inputs = gpu_transform(inputs)

        optimizer.zero_grad()
        #with torch.amp.autocast('cuda', enabled=True):
        # 3. USE BFLOAT16: Faster & stable on A10G GPUs
        # -----------------------
        # Use for 3D-CNN 
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # -----------------------
        # Use for 2D-CNN + LSTM
        # -----------------------
        if model_type == 'r3d_18' or model_type == 'r3d_attention':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
        else:  # 2DCNN+LSTM
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
            
        loss = criterion(outputs, labels)

        # NORMALIZE LOSS
        loss = loss / accum_steps  
             
        # 4. BACKWARD PASS (Accumulate Gradients)     
        scaler.scale(loss).backward()


        # 5. OPTIMIZER STEP (Only every 'accum_steps' or at end of loop)
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)

            if (i + 1) % 100 == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                # print(f"Gradient norm: {total_norm:.4f}")

            scaler.update()
            optimizer.zero_grad() # Reset for next accumulation
        
        # --- METRICS & LOGGING ---
        # (We multiply loss back by accum_steps just for logging display purposes)
        current_loss = loss.item() * accum_steps        
        acc1, acc5, acc10 = accuracy(outputs, labels, topk=(1, 5, 10))
        
        losses.update(current_loss, inputs.size(0))
        top1.update(acc1, inputs.size(0))
        top5.update(acc5, inputs.size(0))
        # 3. Add this line
        top10.update(acc10, inputs.size(0)) 
        
        #### pbar.set_postfix({'loss': losses.avg, 'top1': top1.avg})
        
        # --- LOGGING (Exactly 4 times per epoch) ---
        # We check if we are at a 1/4th milestone OR the very last batch
        if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
            elapsed = time.time() - start_time
            # Calculate speed (images per second)
            speed = (i + 1) * inputs.size(0) / max(1.0, elapsed)
            
            print(
                f"Train Ep: [{epoch+1}][{i+1}/{total_batches}] "
                f"Time: {elapsed:.0f}s ({speed:.1f} img/s) | "
                f"Loss: {losses.avg:.4f} | "
                f"Top1: {top1.avg:.2f}% | "
                f"Top5: {top5.avg:.2f}% | "
                f"Top10: {top10.avg:.2f}% "
            )

    # 4. Return top10 too
    return losses.avg, top1.avg, top5.avg, top10.avg

def validate(model, loader, criterion, device, epoch, model_type):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # 1. Add meter
    top10 = AverageMeter('Acc@10', ':6.2f')


    # 1. DEFINE NORMALIZER (Augment=False just normalizes 0-255 -> 0.0-1.0)
    # Ensure you have GPUAugment class defined/imported in this file
    gpu_normalizer = GPUAugment(augment=False).to(device)

    # Calculate Log Interval
    total_batches = len(loader)
    log_interval = max(1, total_batches // 4)
    start_time = time.time()

    
    # pbar = tqdm(loader, desc=f"Val Ep {epoch+1}", unit="batch", file=sys.stdout)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            # Move raw uint8 data to GPU
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 2. APPLY NORMALIZATION ON GPU
            inputs = gpu_normalizer(inputs) # Converts ByteTensor -> FloatTensor

            # -----------------------
            # Use for 3D-CNN 
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # -----------------------
            # with autocast(enabled=True):
            if model_type == 'r3d_18' or model_type == 'r3d_attention':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
            else:  # 2DCNN+LSTM
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 2. Update accuracy call
            acc1, acc5, acc10 = accuracy(outputs, labels, topk=(1, 5, 10))
            
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
            top5.update(acc5, inputs.size(0))
            # 3. Update meter
            top10.update(acc10, inputs.size(0))
            
            # --- LOGGING (Exactly 4 times per epoch) ---
            if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
                elapsed = time.time() - start_time
                print(
                    f"Val Ep:   [{epoch+1}][{i+1}/{total_batches}] "
                    f"Time: {elapsed:.0f}s | "
                    f"Loss: {losses.avg:.4f} | "
                    f"Top1: {top1.avg:.2f}% | "
                    f"Top10: {top10.avg:.2f}%"
                )
            
    # 4. Return top10
    return losses.avg, top1.avg, top5.avg, top10.avg

# -------------------------------------------------------------------
#   Main Execution
# -------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Sagemaker Paths ---
    # SM_CHANNEL_TRAINING: Where AWS puts your S3 data
    # SM_MODEL_DIR: Where you save the model to be uploaded to S3 automatically
    DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING") 
    MODEL_DIR = os.environ.get("SM_MODEL_DIR")

    # --- 1. CONFIGURE PATHS FOR SAGEMAKER ---
    # SageMaker automatically defines these environment variables
    # output_data_dir is usually /opt/ml/output/data
    OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
    
    # Define log paths INSIDE the magic folder
    log_file_path = os.path.join(OUTPUT_DATA_DIR, "experiment_logs.csv")
    ext_log_file_path = os.path.join(OUTPUT_DATA_DIR, "experiment_logs_detailed.csv")

    # --- 2. INITIALIZE LOGGER ---
    print(f"📝 Logging to: {log_file_path}")
    logger = ExperimentLogger(log_file=log_file_path, extended_log_file=ext_log_file_path)

    # --- 3. PREPARE CONFIG DICT (Required by your logger) ---
    # We map your args to the dictionary keys your logger expects
    current_config = {
        "model_type": "2D-CNN-LSTM",      # Or pass via args.model_type
        "config_id": args.experiment_name,
        "batch_size": args.batch_size,
        "lr": args.learning_rate
    }

    patience = 25

    # ---- More Configs ... ---------------------
    if args.model_type == 'r3d_18':
        # Plain 3DCNN - needs most help
        warmup_start_lr = 5e-7
        warmup_epochs = 5
        patience = 40
        label_smooth = 0.05
        
    elif args.model_type == 'r3d_attention':
        # Working well, minor adjustments
        warmup_start_lr = 1e-6
        warmup_epochs = 5
        patience = 25
        label_smooth = 0.1
        
    elif args.model_type == '2dcnn_lstm':
        warmup_start_lr = 3e-6
        warmup_epochs = 5
        patience = 25
        label_smooth = 0.1
    # --------------------------------------------


    # --- Config Overlay ---
    CONFIG = BASE_CONFIG.copy()
    CONFIG.update({
        "tensor_dir": DATA_DIR, 
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        # We assume JSON files are in the 'ASL' folder uploaded with source code
        "train_json": f"train_{args.num_classes}.json",
        "val_json": f"val_{args.num_classes}.json",
    })

    print("Initializing Datasets...")
    train_ds = FastWLASLDataset(CONFIG['train_json'], CONFIG['tensor_dir'], augment=True, num_classes=args.num_classes)
    val_ds = FastWLASLDataset(CONFIG['val_json'], CONFIG['tensor_dir'], augment=False, num_classes=args.num_classes)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True  # Add this
        )
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2
    )
    
    model = get_model(CONFIG).to(device)
    # 1. Filter only the trainable parameters (ignore frozen backbone)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # 2. Initialize Optimizer (Pass LR only ONCE as a keyword arg)
    optimizer = optim.AdamW(
        trainable_params, 
        lr=CONFIG['learning_rate'], 
        weight_decay=0.01
        )

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=patience, 
        verbose=True, 
        min_lr=1e-6
    )


    RESUME_DIR = os.environ.get("SM_CHANNEL_RESUME") # Maps to /opt/ml/input/data/resume
    start_epoch = 0

    if RESUME_DIR and os.path.exists(RESUME_DIR):
        print(f"♻️ Checking for checkpoints in {RESUME_DIR}...")
        
        # 1. Check if it's a compressed tar file (SageMaker default)
        tar_path = os.path.join(RESUME_DIR, "model.tar.gz")
        if os.path.exists(tar_path):
            print(f"📦 Found compressed model: {tar_path}. Extracting...")
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=RESUME_DIR)
                print("✅ Extraction complete.")
            except Exception as e:
                print(f"❌ Error extracting tar file: {e}")

        # 2. Now look for the actual weight file (model.pth)
        # Note: SageMaker creates model.pth directly at root, or sometimes inside a folder depending on how you saved it.
        # This checks the root of RESUME_DIR.
        checkpoint_path = os.path.join(RESUME_DIR, "model.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"⚖️ Loading weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Set start epoch
            start_epoch = args.start_epoch
            print(f"🚀 Successfully resumed! Starting from Epoch {start_epoch}")
        else:
            print(f"⚠️ Resume channel active, but 'model.pth' not found in {RESUME_DIR}. Starting from scratch.")



    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    #scaler = torch.amp.GradScaler('cuda')
    scaler = GradScaler()


    # 4. EARLY STOPPING VARIABLES
    early_stopping_patience = 25
    no_improve_epochs = 0
    best_val_loss = float('inf') # Track Loss for Early Stopping, not just Acc


    best_acc = 0.0



    # Define Warmup Params
    WARMUP_EPOCHS = warmup_epochs
    TARGET_LR = CONFIG['learning_rate'] # e.g., 1e-4
    START_LR = warmup_start_lr 
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")


        # --- MANUAL WARMUP LOGIC (FIXED) ---
        if epoch < WARMUP_EPOCHS:
            # Use (epoch + 1) so at epoch 4, we get 5/5 = 1.0
            warmup_lr = START_LR + (TARGET_LR - START_LR) * ((epoch + 1) / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup Epoch {epoch+1}/{WARMUP_EPOCHS}: LR set to {warmup_lr:.2e}")

        # --- CRITICAL: FORCE TARGET LR ONCE WARMUP ENDS ---
        elif epoch == WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = TARGET_LR
            print(f"✅ Warmup Complete. LR set to full target: {TARGET_LR:.2e}")
        
        # --- TRAIN LOOP ---
        # mininterval=30 prevents log spam in CloudWatch
        train_loader_tqdm = tqdm(
            train_loader, 
            desc=f"Train Ep {epoch+1}", 
            file=sys.stdout, 
            mininterval=30.0 
        )


        # Inside the train() function loop...

        train_loss, train_acc1, train_acc5, train_acc10 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args.model_type
        )

        # --- VAL LOOP ---
        val_loader_tqdm = tqdm(
            val_loader, 
            desc=f"Val Ep {epoch+1}", 
            file=sys.stdout, 
            mininterval=30.0
        )

        val_loss, val_acc1, val_acc5, val_acc10 = validate(
            model, val_loader, criterion, device, epoch, args.model_type
        )

        # --- KEY STEP: Standardized Logging for Regex ---
        # This formatting makes it easy for SageMaker to graph your progress
        #print(f"\n[Metrics] Epoch: {epoch+1}")
        # --- LOGGING ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Metrics] Train Loss: {train_loss:.4f} | Train Top1: {train_acc1:.2f} | Train Top5: {train_acc5:.2f} | Train Top10: {train_acc10:.2f}")
        
        print(f"[Metrics] Val Loss: {val_loss:.4f} | Val Top1: {val_acc1:.2f} | Val Top5: {val_acc5:.2f} | Val Top10: {val_acc10:.2f} | LR: {current_lr:.2e}")



        
        # --- 5. SCHEDULER STEP ---
        # Reduce LR if Validation Loss stops improving
        # --- 2. SCHEDULER LOGIC ---
        # Only let the Plateau scheduler take over AFTER warmup
        if epoch >= WARMUP_EPOCHS:
            scheduler.step(val_loss)
        
        # --- 6. EARLY STOPPING LOGIC ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0 # Reset counter
        else:
            no_improve_epochs += 1
            print(f"⏳ No improvement in Loss for {no_improve_epochs}/{patience} epochs.")

        # --- THE FIX: IMMUNITY CHECK ---
        # Don't even check the trigger if we are below Epoch 40
        if epoch > 40 and no_improve_epochs >= patience:
            print("🛑 Early Stopping Triggered! Training finished.")
            break


        if val_acc1 > best_acc:
            best_acc = val_acc1
            # Save strictly to MODEL_DIR
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
            print(f"Saved Best Model ({best_acc:.2f}%)")


        # --- 4. LOG THE EPOCH ---
        # Pack the stats exactly how your logger expects them
        train_stats = (train_loss, train_acc1, train_acc5, train_acc10) 
        val_stats = (val_loss, val_acc1, val_acc5, val_acc10)

        logger.log_epoch(
            config=current_config,
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            best_val_acc=best_acc, 
            current_lr=optimizer.param_groups[0]['lr']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyperparameters sent by the launcher
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--model-type", type=str, default="2dcnn_lstm")
    parser.add_argument("--start-epoch", type=int, default = 0)
    parser.add_argument("--experiment-name", type=str, default="default-experiment")
    
    args = parser.parse_args()
    train(args)
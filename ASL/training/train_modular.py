import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import random
import torchvision.transforms.functional as F_vis
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint

# Ensure the root directory (Thesis-CSLR) is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Modular Imports
from ASL.model_and_config.config import CONFIG
from ASL.logger import ExperimentLogger
from ASL.data_and_preprocess.dataset import FastWLASLDataset
from ASL.model_and_config.models import get_model


from torch.utils.checkpoint import checkpoint
import torch.nn as nn

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return checkpoint(self.layer, x, use_reentrant=False)

def apply_gradient_checkpointing(model):
    """
    Applies gradient checkpointing to heavy layers to save VRAM.
    Supports:
    1. R3D Backbone (model.backbone.layer1...4)
    2. 2DCNN Backbone (model.cnn[4...7])
    """
    layers_to_checkpoint = []
    
    if hasattr(model, 'backbone'):
        # R3D Case
        backbone = model.backbone
        # layer1, layer2, layer3, layer4
        layers_to_checkpoint = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        
        # Replace explicitly
        backbone.layer1 = CheckpointedLayer(backbone.layer1)
        backbone.layer2 = CheckpointedLayer(backbone.layer2)
        backbone.layer3 = CheckpointedLayer(backbone.layer3)
        backbone.layer4 = CheckpointedLayer(backbone.layer4)
        print("✅ Gradient Checkpointing enabled on R3D backbone layers 1-4.")
        
    elif hasattr(model, 'cnn'):
        # 2DCNN-LSTM Case
        # model.cnn is a nn.Sequential typically containing:
        # [0:conv1, 1:bn1, 2:relu, 3:maxpool, 4:layer1, 5:layer2, 6:layer3, 7:layer4, 8:avgpool]
        cnn = model.cnn
        # We need to modify the Sequential in-place. 
        # Accessing by index works for Sequential.
        
        # Indices for layer1 (4) to layer4 (7)
        target_indices = [4, 5, 6, 7]
        
        for idx in target_indices:
            if idx < len(cnn):
                layer = cnn[idx]
                cnn[idx] = CheckpointedLayer(layer)
                
        print("✅ Gradient Checkpointing enabled on 2DCNN layers 4-7.")
        
    else:
        print("⚠️ Could not find 'backbone' or 'cnn' to checkpoint. Skipping.")

    return model


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

        self.rot_range = 30

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
            x = x.reshape(B, C * T, H, W) 
            
            # ROTATE (Now it works because input is 4D)
            x = F_vis.rotate(x, angle)
            
            # UNFOLD BACK: (B, C*T, H, W) -> (B, C, T, H, W)
            x = x.reshape(B, C, T, H, W)   

        x = x.sub_(self.mean).div_(self.std)
        return x.to(memory_format=torch.channels_last_3d)
# -------------------------------------------------------------------
#   Training Functions
# -------------------------------------------------------------------



# -------------------------------------------------------------------
#   Mixup Utilities
# -------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, config):

    physical_bs = loader.batch_size
    model_type = config['model_type']
    
    if model_type == 'r3d_18':
        target_bs = 128
    elif model_type == '2dcnn_lstm':
        target_bs = 96
    elif model_type == 'r3d_attention':
        target_bs = 128
    else:
        target_bs = 66
    # Calculate steps dynamically (e.g., 66 // 11 = 6 steps)
    accum_steps = max(1, target_bs // physical_bs)
    
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # 1. Add this line
    top10 = AverageMeter('Acc@10', ':6.2f') 
    
    gpu_transform = GPUAugment(augment=True).to(device)

    # gpu_transform = torch.jit.script(gpu_transform)


    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch+1}", unit="batch")
    
    for i, (inputs, labels) in enumerate(pbar):
    
        # 1. Move to Device
        # Use conditional memory format
        if model_type in ['r3d_18', 'r3d_attention']:
            input_mem_fmt = torch.channels_last_3d
        else:
            input_mem_fmt = torch.contiguous_format

        inputs = inputs.to(device, non_blocking=True, memory_format=input_mem_fmt)
        labels = labels.to(device, non_blocking=True)

        # 2. Determine Precision Type (Do this once)
        # Using bfloat16 for everything to avoid overflow (Inf)
        amp_dtype = torch.bfloat16

        # --- BUG FIX: Removed 'optimizer.zero_grad()' from here ---
        # We only zero grads AFTER the optimizer step (bottom of loop).
        
        # 3. Start Mixed Precision Context EARLIER
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            
            # A. Augmentations now run in BF16/FP16 (Faster Bandwidth)
            # We use no_grad because we don't need to backprop through augmentations
            with torch.no_grad():
                inputs = gpu_transform(inputs)

            # B. Mixup Logic
            if config.get('use_mixup', False) and config.get('mixup_alpha', 0.0) > 0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config.get('mixup_alpha', 0.2), device)
                
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # Standard Forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # NORMALIZE LOSS for Accumulation
            loss = loss / accum_steps  

        # 4. BACKWARD PASS (Accumulate Gradients)     
        scaler.scale(loss).backward()
                
        # 5. OPTIMIZER STEP (Only every 'accum_steps' or at end of loop)
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)

            # ... (Logging code) ...

            scaler.update()
            
            # 6. RESET GRADIENTS (Correct placement for accumulation)
            optimizer.zero_grad()
        
        # Stats
        #running_loss += loss.item() * labels.size(0)
        #_, predicted = torch.max(outputs, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()

         # --- METRICS & LOGGING ---
        # (We multiply loss back by accum_steps just for logging display purposes)
        current_loss = loss.item() * accum_steps        
        acc1, acc5, acc10 = accuracy(outputs, labels, topk=(1, 5, 10))
        
        losses.update(current_loss, inputs.size(0))
        top1.update(acc1, inputs.size(0))
        top5.update(acc5, inputs.size(0))
        # 3. Add this line
        top10.update(acc10, inputs.size(0)) 
        
        pbar.set_postfix({'loss': losses.avg, 'top1' : top1.avg, 'top5' : top5.avg, 'top10' : top10.avg})
        
    #epoch_loss = running_loss / total if total > 0 else 0
    #epoch_acc = correct / total if total > 0 else 0
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

    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Val Ep {epoch+1}", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            
            # Conditional Memory Format for Inputs
            if model_type in ['r3d_18', 'r3d_attention']:
                input_mem_fmt = torch.channels_last_3d
            else:
                input_mem_fmt = torch.contiguous_format
                
            inputs = inputs.to(device, non_blocking=True, memory_format=input_mem_fmt)
                 
            labels = labels.to(device, non_blocking=True)
            
            # 2. APPLY NORMALIZATION ON GPU
            inputs = gpu_normalizer(inputs) # Converts ByteTensor -> FloatTensor

            # -----------------------
            # Use for 3D-CNN 
            # with torch.autocast(device_type='cuda', dftype=torch.bfloat16):
            # -----------------------
            # with autocast(enabled=True):
            # 3. Mixed Precision
            # 2DCNN+LSTM on float16 caused Inf loss during validation.
            # Switching to bfloat16 for ALL models (supported on Ampere/Ada)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)

            loss = criterion(outputs, labels)
                
            #running_loss += loss.item() * labels.size(0)
            #_, predicted = torch.max(outputs, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            
                        # 2. Update accuracy call
            acc1, acc5, acc10 = accuracy(outputs, labels, topk=(1, 5, 10))
            
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
            top5.update(acc5, inputs.size(0))
            # 3. Update meter
            top10.update(acc10, inputs.size(0))
            
            pbar.set_postfix({'val_loss': losses.avg, 'val_acc': top1.avg, 'top5' : top5.avg, 'top10' : top10.avg})
            
    #epoch_loss = running_loss / total if total > 0 else 0
    #epoch_acc = correct / total if total > 0 else 0
    return losses.avg, top1.avg, top5.avg, top10.avg

def main():
    # 0. GPU Optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Logger
    logger = ExperimentLogger(CONFIG['log_file'])
    
    # Define Model Directory
    MODEL_DIR = CONFIG['checkpoint_dir']
    
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

    weight_decay = 0.01
    
    # Default patience
    patience = 12
    
     # ---- More Configs ... ---------------------
    if CONFIG['model_type'] == 'r3d_18':
        # Plain 3DCNN - needs most help
        warmup_start_lr = 5e-7
        warmup_epochs = 5
        patience = 12
        label_smooth = 0.05
        
    elif CONFIG['model_type'] == 'r3d_attention':
        # Working well, minor adjustments
        warmup_start_lr = 1e-6
        warmup_epochs = 5
        patience = 25
        label_smooth = 0.1
        
    elif CONFIG['model_type'] == '2dcnn_lstm':
        warmup_start_lr = 3e-6
        warmup_epochs = 5
        label_smooth = 0.1
        weight_decay = 0.05
    
    # --- HYPERPARAMETERS ---
    # scheduler_patience = 12 # REMOVED: potentially shadowed model-specific patience
    early_stopping_patience = 50
    
    # --------------------------------------------


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
    num_workers = CONFIG.get('num_workers', 6) # Allow config override
    persistent_workers = True
    
    print(f"DataLoader Config: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor = 2,
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor = 2,
    )
    
    print(f"Training on {len(train_ds)} samples. Validating on {len(val_ds)} samples.")
    
    # Model
    # Conditional Memory Format (Channels Last 3D only for 5D tensors)
    if CONFIG['model_type'] in ['r3d_18', 'r3d_attention']:
        mem_fmt = torch.channels_last_3d
    else:
        mem_fmt = torch.contiguous_format
        
    model = get_model(CONFIG).to(device, memory_format=mem_fmt)

    # 2. Apply Checkpointing (STEP 2)
    # This must happen BEFORE torch.compile
    model = apply_gradient_checkpointing(model)

    # 3. Compile (STEP 1 - from previous advice)
    # Compiling a checkpointed model works very well in PyTorch 2.0+
    # model = torch.compile(model)


    # Separate parameters
    if hasattr(model, 'backbone'):
        backbone_params = model.backbone.parameters()
        head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    elif hasattr(model, 'cnn'):
        backbone_params = model.cnn.parameters()
        head_params = [p for n, p in model.named_parameters() if "cnn" not in n]
    else:
        # Fallback if neither exists (shouldn't happen with current models)
        backbone_params = []
        head_params = model.parameters()
    # 1. Filter only the trainable parameters (ignore frozen backbone)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # 2. Initialize Optimizer (Pass LR only ONCE as a keyword arg)
    if CONFIG['model_type'] == '2dcnn_lstm':
        optimizer = optim.AdamW(
            [
                {'params': backbone_params, 'lr': CONFIG['learning_rate'] * 0.1}, # 10x slower
                {'params': head_params, 'lr': CONFIG['learning_rate']}            # Standard speed
            ], 
            weight_decay=weight_decay,
            fused = True
        )
    else:
        optimizer = optim.AdamW(
            trainable_params, 
            lr=CONFIG['learning_rate'], 
            weight_decay=weight_decay,
            fused = True
        )
    # Requirement 1: Adaptive Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=patience, 
        min_lr=1e-6,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    #scaler = torch.amp.GradScaler('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    
    # 4. EARLY STOPPING VARIABLES
    # early_stopping_patience is defined above (10)
    no_improve_epochs = 0
    best_val_loss = float('inf') 
    best_acc = 0.0

    # Define Warmup Params
    WARMUP_EPOCHS = warmup_epochs
    TARGET_LR = CONFIG['learning_rate'] 
    START_LR = warmup_start_lr 

    # --- RESUME LOGIC ---
    start_epoch = 0
    # RESUME_PATH = r"a:\Thesis-CSLR\ASL\checkpoints\r3d_18_1000_best_checkpoint.pth" 
    RESUME_PATH = CONFIG.get('resume_checkpoint') # Set to None for fresh training, or update to a compatible checkpoint
    # Example: RESUME_PATH = os.path.join(MODEL_DIR, "best_checkpoint.pth")
    # You can set this variable manually to resume!
    
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming checkpoint from {RESUME_PATH}...")
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        
        # --- Support for Raw State Dicts (Legacy Checkpoints) ---
        if 'model_state_dict' not in checkpoint:
            print("⚠️ Checkpoint seems to be a raw state dictionary (no metadata). Wrapping it.")
            checkpoint = {'model_state_dict': checkpoint}
            
        # --- Compatibility Check ---
        ckpt_config = checkpoint.get('config', {})
        ckpt_model_type = ckpt_config.get('model_type')
        current_model_type = CONFIG['model_type']
        
        if ckpt_model_type and ckpt_model_type != current_model_type:
             raise RuntimeError(f"❌ Checkpoint Mismatch! Config expects '{current_model_type}' but checkpoint is '{ckpt_model_type}'. Please update config.py or use a compatible checkpoint.")
             
        # Fallback for legacy checkpoints (inspect keys)
        # Check if we are trying to load a 2D-CNN (has 'cnn') into a 3D-CNN (expects 'backbone')
        state_keys = list(checkpoint['model_state_dict'].keys())
        has_cnn_keys = any('cnn' in k for k in state_keys)
        has_backbone_keys = any('backbone' in k for k in state_keys)
        # Also check for direct parameter names if using raw dict (e.g., 'stem.0.weight')
        has_stem_keys = any('stem.' in k for k in state_keys)
        
        # Refine check: 
        # r3d_18 expects backbone or stem
        # 2dcnn expects cnn
        
        if current_model_type == 'r3d_18' and has_cnn_keys:
             raise RuntimeError(f"❌ Checkpoint Mismatch! Config expects 'r3d_18' (backbone/stem) but checkpoint seems to be 2D-CNN (has 'cnn' keys).")
        if current_model_type == '2dcnn_lstm' and (has_backbone_keys or has_stem_keys):
             raise RuntimeError(f"❌ Checkpoint Mismatch! Config expects '2dcnn_lstm' (cnn) but checkpoint seems to be 3D-CNN (has 'backbone' or 'stem' keys).")

        # --- Key Adaptation for Gradient Checkpointing ---
        # If the model uses Gradient Checkpointing (CheckpointedLayer), keys will have an extra '.layer.'
        # If the checkpoint was saved WITHOUT it, we need to add it.
        # Or vice versa.
        
        new_state_dict = {}
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(checkpoint['model_state_dict'].keys())
        
        for k, v in checkpoint['model_state_dict'].items():
            # If key matches perfectly, use it
            if k in model_keys:
                new_state_dict[k] = v
                continue
                
            # Case 1: Model has '.layer.' (Checkpointed), Checkpoint does not (Plain)
            # Try inserting '.layer' after 'layerX' or 'cnn.X'
            # R3D_18: backbone.layer1.0... -> backbone.layer1.layer.0...
            # 2DCNN: cnn.4.0... -> cnn.4.layer.0...
            
            # Simple Heuristic: If k is NOT in model, but k with '.layer' IS in model
            # We need to find valid insertion points.
            # Common patterns: ".layer1.", ".layer2.", ".layer3.", ".layer4."
            
            candidate_key = None
            
            # Try inserting .layer after layer1/2/3/4
            for i in range(1, 5):
                search_str = f"layer{i}."
                if search_str in k:
                    # insert .layer after it
                    # e.g. backbone.layer1.0.weight -> backbone.layer1.layer.0.weight
                    parts = k.split(search_str)
                    # parts[0] = backbone.
                    # parts[1] = 0.weight
                    candidate = f"{parts[0]}{search_str}layer.{parts[1]}"
                    if candidate in model_keys:
                        candidate_key = candidate
                        break
            
            # Try inserting .layer after cnn.4/5/6/7 (for 2DCNN)
            if not candidate_key and 'cnn.' in k:
                for i in range(4, 8):
                    search_str = f"cnn.{i}."
                    if search_str in k:
                        candidate = k.replace(search_str, f"cnn.{i}.layer.")
                        if candidate in model_keys:
                            candidate_key = candidate
                            break
                            
            if candidate_key:
                new_state_dict[candidate_key] = v
                # print(f"Mapping {k} -> {candidate_key}")
            else:
                # If no mapping found, keep original (will likely error if strict=True, but let's keep it)
                new_state_dict[k] = v
        
        # Replace the state dict
        checkpoint['model_state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Conditional Loading for Metadata
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Loaded optimizer state.")
        else:
            print("⚠️ Checkpoint missing optimizer state. Starting fresh optimizer.")
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        #     # Make sure to catch any scheduler mismatches if patience changed, 
        #     # but usually load_state_dict is fine.
        #     try:
        #         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #     except Exception as e:
        #         print(f"⚠️ Could not load scheduler state (patience changed?): {e}")
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        NEW_LR = 1e-4  # <--- Set your desired restart LR here
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = NEW_LR
            # param_group['weight_decay'] = 0.1
            
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"✅ Resumed from Epoch {start_epoch}, Best Acc: {best_acc:.2f}%")
    else:
        print("🚀 Starting training from scratch.")
        
    
    print(f"Starting training on {device}...")
    
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
        train_loss, train_acc1, train_acc5, train_acc10 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, CONFIG
        )

                # --- VAL LOOP ---
        val_loss, val_acc1, val_acc5, val_acc10 = validate(
            model, val_loader, criterion, device, epoch, CONFIG['model_type']
        )
        
        # Scheduler Step
        current_lr = optimizer.param_groups[0]['lr']



        print(f"[Metrics] Train Loss: {train_loss:.4f} | Train Top1: {train_acc1:.2f} | Train Top5: {train_acc5:.2f} | Train Top10: {train_acc10:.2f}")
        
        print(f"[Metrics] Val Loss: {val_loss:.4f} | Val Top1: {val_acc1:.2f} | Val Top5: {val_acc5:.2f} | Val Top10: {val_acc10:.2f} | LR: {current_lr:.2e}")


        # [REMOVED DUPLICATE SCHEDULER STEP]

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
            print(f"⏳ No improvement in Loss for {no_improve_epochs}/{early_stopping_patience} epochs.")

        # --- THE FIX: IMMUNITY CHECK ---
        # Don't even check the trigger if we are below Epoch 40
        # --- THE FIX: IMMUNITY CHECK ---
        # Don't even check the trigger if we are below Epoch 40
        if epoch > 40 and no_improve_epochs >= early_stopping_patience:
            print("🛑 Early Stopping Triggered! Training finished.")
            break

        if val_acc1 > best_acc:
            best_acc = val_acc1
            # Save strictly to MODEL_DIR
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'best_acc': best_acc,
                'config': CONFIG
            }
            torch.save(checkpoint, os.path.join(MODEL_DIR, f"{CONFIG['model_type']}_{CONFIG['num_classes']}_mixup_v1_best_checkpoint.pth"))
            # Also save just weights for easy inference if needed, or just rely on checkpoint
            # torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_weights.pth"))
            print(f"Saved Best Model Checkpoint ({best_acc:.2f}%)")

        
        # --- 4. LOG THE EPOCH ---
        # Pack the stats exactly how your logger expects them
        train_stats = (train_loss, train_acc1, train_acc5, train_acc10) 
        val_stats = (val_loss, val_acc1, val_acc5, val_acc10)

        logger.log_epoch(
            config=CONFIG,
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            best_val_acc=best_acc, 
            current_lr=optimizer.param_groups[0]['lr']
        )
            
    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()

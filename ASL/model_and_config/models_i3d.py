import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class TwoStreamR3D(nn.Module):
    """
    A Two-Stream Inflated 3D ConvNet (I3D) using a ResNet-18 backbone.
    Stream 1: Spatial (RGB) - Captures appearance.
    Stream 2: Temporal (Flow) - Captures motion.
    
    Both streams are initialized with pre-trained Kinetics-400 weights.
    """
    def __init__(self, num_classes, dropout_prob=0.5):
        super(TwoStreamR3D, self).__init__()
        
        # --- STREAM 1: RGB ---
        # Load pre-trained weights automatically
        print("Loading Pre-trained Weights for RGB Stream...")
        self.rgb_stream = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Remove the classification head (fc) to get features
        self.rgb_stream.fc = nn.Identity() 
        
        # --- STREAM 2: OPTICAL FLOW ---
        # Load pre-trained weights automatically
        print("Loading Pre-trained Weights for Flow Stream...")
        self.flow_stream = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        # FIX: The original model expects 3 channels (RGB). Flow has 2 channels (x, y).
        # We must modify the first convolution layer to accept 2 channels.
        old_conv = self.flow_stream.stem[0]
        new_conv = nn.Conv3d(
            in_channels=2, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=False
        )
        
        # TRANSFER LEARNING TRICK:
        # We average the pre-trained RGB weights to initialize the Flow weights.
        # This keeps the "shape" detection logic intact, giving a better start than random.
        with torch.no_grad():
            # old_weight shape: (64, 3, 3, 7, 7)
            # mean over channel dim (dim=1) -> (64, 1, 3, 7, 7)
            # repeat to 2 channels -> (64, 2, 3, 7, 7)
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True).repeat(1, 2, 1, 1, 1)
            
        self.flow_stream.stem[0] = new_conv
        self.flow_stream.fc = nn.Identity()

        # --- FUSION HEAD ---
        # Each stream outputs 512 features. Concatenated = 1024.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(512 * 2, num_classes)
        )

    def forward(self, x):
        # x is a tuple/list: [rgb_tensor, flow_tensor]
        # rgb_tensor: (Batch, 3, Frames, H, W)
        # flow_tensor: (Batch, 2, Frames, H, W)
        
        rgb_input, flow_input = x
        
        # 1. Forward pass RGB
        rgb_features = self.rgb_stream(rgb_input)   # Output: (Batch, 512)
        
        # 2. Forward pass Flow
        flow_features = self.flow_stream(flow_input) # Output: (Batch, 512)
        
        # 3. Late Fusion (Concatenate)
        combined_features = torch.cat((rgb_features, flow_features), dim=1) # (Batch, 1024)
        
        # 4. Classify
        logits = self.classifier(combined_features)
        return logits
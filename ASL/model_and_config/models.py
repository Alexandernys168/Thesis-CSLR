import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18, R3D_18_Weights

class BaselineResNet3D(nn.Module):
    """
    Standard ResNet3D-18 model for classification.
    Replaces the final fully connected layer to match num_classes.
    """
    def __init__(self, num_classes, pretrained=True, dropout_prob=0.5):
        super(BaselineResNet3D, self).__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ResNet3D_LSTM(nn.Module):
    """
    Hybrid architecture: 3D-CNN (ResNet18) + Bidirectional LSTM.
    
    The 3D-CNN serves as a feature extractor. Its temporal dimension is preserved 
    (or reduced by the network structure), and spatial dimensions are pooled.
    The resulting sequence of features is passed to the LSTM.
    """
    def __init__(self, num_classes, hidden_size=256, num_layers=2, pretrained=True, dropout_prob=0.5):
        super(ResNet3D_LSTM, self).__init__()
        
        # 1. Feature Extractor (ResNet3D-18)
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        resnet = r3d_18(weights=weights)
        
        # Remove the classification head (fc)
        # We also need to modify the final pooling layer to preserve the temporal dimension
        # Standard r3d_18 structure: stem -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        
        self.stem = resnet.stem
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Adaptive pooling to (Time, 1, 1). 
        # We output (C, T, 1, 1). 
        # Note: The 'T' dimension size depends on input size and strides.
        # r3d_18 has temporal downsampling in Stem (maybe), Layer3 (stride 2), Layer4 (stride 2).
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # 2. LSTM
        # ResNet18 output channels is 512 at layer4.
        self.input_dim = 512
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 3. Classifier
        # Bidirectional -> hidden_size * 2
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        # x: (Batch, C, T, H, W)
        
        # --- Feature Extraction ---
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Shape: (Batch, 512, T', H', W')
        
        x = self.avgpool(x)
        # Shape: (Batch, 512, T', 1, 1)
        
        # Reshape for LSTM: (Batch, Sequence_Length, Features)
        # Permute to (Batch, T', 512, 1, 1) -> Squeeze dimensions
        x = x.permute(0, 2, 1, 3, 4) 
        x = x.flatten(2) 
        # Shape: (Batch, T', 512)
        
        # --- Sequence Modeling ---
        # self.lstm returns: output, (h_n, c_n)
        # output: (Batch, Seq_Len, Num_Directions * Hidden_Size)
        # h_n: (Num_Layers * Num_Directions, Batch, Hidden_Size)
        self.lstm.flatten_parameters() # For multi-gpu / efficiency
        _, (hn, _) = self.lstm(x)
        
        # --- Classification ---
        # We use the final hidden state. 
        # Since it is bidirectional, we concatenate the forward validation of the last layer 
        # and backward validation of the last layer.
        # hn view: (num_layers, num_directions, batch, hidden_size)
        
        # Take the last layer's hidden states
        # Forward: hn[-2, :, :]
        # Backward: hn[-1, :, :]
        
        hn_last_layer = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        # Shape: (Batch, Hidden_Size * 2)
        out = self.fc(hn_last_layer)
        return out

class CNNRNN_2D(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, pretrained=True, dropout_prob=0.5):
        super(CNNRNN_2D, self).__init__()
        # 1. Visual Backbone (2D ResNet-18)
        # We utilize standard ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        # Remove the final FC layer to get a feature vector (512 dim)
        # We keep the AvgPool layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # This loop MUST be active for LR=1e-3
        #for param in self.cnn.parameters():
        #    param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_prob if num_layers > 1 else 0
        )
        # 3. Classifier Head
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size * 2, num_classes)
        )
    def forward(self, x):
        # Input shape: (Batch, C, T, H, W)
        b, c, t, h, w = x.size()

        # Merge Batch and Time to process frames in parallel
        c_in = x.view(b * t, c, h, w)

        # CNN Forward -> (B*T, 512, 1, 1)
        features = self.cnn(c_in)
        
        # Squeeze spatial dims -> (B*T, 512)
        features = features.view(b, t, -1) 

        # LSTM Forward
        self.lstm.flatten_parameters()
        # We only care about the hidden state of the last timestep
        _, (hn, _) = self.lstm(features)
        
        # Concatenate forward and backward hidden states from the last layer
        # hn shape: (num_layers * num_directions, batch, hidden_size)
        # We want the last layer's forward (idx -2) and backward (idx -1) states
        final_feature = torch.cat((hn[-2], hn[-1]), dim=1)
        
        return self.fc(final_feature)


class TemporalAttention(nn.Module):
    """
    Learns to weight important frames in the video clip.
    Input: (Batch, Channels, Time, H, W)
    Output: (Batch, Channels * H * W) weighted feature vector
    """
    def __init__(self, in_channels, hidden_dim=128):
        super(TemporalAttention, self).__init__()
        
        # Reduces (C, H, W) to a single scalar per frame
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) 
        
        # Learnable attention weights
        self.attention_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Softmax(dim=2) # Normalize weights across Time dimension
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        
        # 1. Global Average Pooling over Spatial dims -> (B, C, T, 1, 1)
        global_feat = self.avg_pool(x).squeeze(-1).squeeze(-1) # (B, C, T)
        
        # 2. Compute Attention Scores
        # weights: (B, 1, T)
        weights = self.attention_net(global_feat)
        
        # 3. Apply Attention to original features
        # Expand weights to (B, 1, T, 1, 1) to broadcast
        x = x * weights.unsqueeze(-1).unsqueeze(-1)
        
        return x

class ResNet3DWithAttention(nn.Module):
    """
    ResNet3D-18 with an added Temporal Attention module.
    """
    def __init__(self, num_classes, pretrained=True, dropout_prob=0.5):
        super(ResNet3DWithAttention, self).__init__()
        
        # 1. Load Backbone
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        
        # 2. Extract feature dimension (512 for ResNet18)
        in_features = self.backbone.fc.in_features
        
        # 3. Insert Attention Module
        # We need to hook into the model BEFORE the final pooling/FC layers.
        # ResNet structure: stem -> layer1 -> ... -> layer4 -> avgpool -> fc
        # We will keep layer1-4, then add attention, then pool.
        
        # Copy layers excluding avgpool and fc
        self.features = nn.Sequential(
            self.backbone.stem,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        
        self.attention = TemporalAttention(in_channels=in_features)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        # Extract features (B, 512, T/8, H/16, W/16)
        x = self.features(x)
        
        # Apply Attention (Weight important time steps)
        x = self.attention(x)
        
        # Pool and Classify
        x = self.avgpool(x) # (B, 512, 1, 1, 1)
        x = x.flatten(1)    # (B, 512)
        x = self.fc(x)
        
        return x

def get_model(config):
    """
    Initializes the model based on config dictionary.
    """
    model_type = config.get('model_type', '2dcnn_lstm')
    
    if model_type == 'r3d_18':
        model = BaselineResNet3D(
            num_classes=config['num_classes'],
            pretrained=config['pretrained'],
            dropout_prob=config['dropout_prob']
        )
    elif model_type == 'r3d_lstm':
        model = ResNet3D_LSTM(
            num_classes=config['num_classes'],
            hidden_size=config.get('lstm_hidden_size', 256),
            num_layers=config.get('lstm_layers', 2),
            pretrained=config['pretrained'],
            dropout_prob=config['dropout_prob']
        )
    elif model_type == '2dcnn_lstm':
        model = CNNRNN_2D(
            num_classes=config['num_classes'],
            hidden_size=config.get('lstm_hidden_size', 256),
            num_layers=config.get('lstm_layers', 2),
            pretrained=config['pretrained'],
            dropout_prob=config['dropout_prob']
        )
    elif model_type == 'r3d_attention':
        # This uses the new ResNet3DWithAttention class
        model = ResNet3DWithAttention(
            num_classes=config['num_classes'],
            pretrained=config['pretrained'],
            dropout_prob=config['dropout_prob']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model

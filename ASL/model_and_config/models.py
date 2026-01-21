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
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
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
        batch_size, c, t, h, w = x.size()

        # --- Spatial Feature Extraction ---
        # Merge Batch and Time to process frames in parallel as 2D images
        c_in = x.view(batch_size * t, c, h, w)

        # Pass through 2D CNN
        features = self.cnn(c_in).view(batch_size, t, -1)

        # LSTM Forward
        self.lstm.flatten_parameters()
        _, (hn, _) = self.lstm(features)
        
        # --- Classification ---
        # Concatenate the final forward and backward hidden states
        # hn shape: (num_layers*2, batch, hidden_size)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))

def get_model(config):
    """
    Initializes the model based on config dictionary.
    """
    model_type = config.get('model_type', 'r3d_18')
    
    # I3D Imports
    from ASL.model_and_config.models_i3d import InceptionI3d, TwoStreamI3D
    
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
    elif model_type == 'i3d_rgb':
        model = InceptionI3d(
            num_classes=config['num_classes'],
            in_channels=3,
            dropout_keep_prob=config['dropout_prob']
        )
    elif model_type == 'i3d_flow':
        model = InceptionI3d(
            num_classes=config['num_classes'],
            in_channels=2,
            dropout_keep_prob=config['dropout_prob']
        )
    elif model_type == 'i3d_two_stream':
        model = TwoStreamI3D(
            num_classes=config['num_classes'],
            dropout_prob=config['dropout_prob']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model

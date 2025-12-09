import torch
import torch.nn as nn

import numpy as np
from us_lib.data.normalization import NormalizationHandler

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, channels=[32, 64, 128, 256]):
        super().__init__()
        
        layers = []
        in_ch = input_channels
        
        for i, out_ch in enumerate(channels):
            kernel_size = 7 if i == 0 else 3  # First layer uses 7x7, rest use 3x3
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            # First block
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Second block
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=0, output_padding=0),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Third block
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0, output_padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Fourth block
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, output_padding=0),
            nn.Conv2d(32, 32, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.decoder(x)

class Bridge(nn.Module):
    def __init__(self, in_channels=256, out_channels=512):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.bridge(x)

class EncoderDecoderNetwork(nn.Module):
    def __init__(self, input_channels=1, channels = [32, 64, 128, 256], output_channels=1):
        super(EncoderDecoderNetwork, self).__init__()
        self.encoder = EncoderBlock(input_channels=input_channels, channels=channels)
        self.bridge = Bridge(in_channels=channels[-1], out_channels=512)
        self.decoder = DecoderBlock()
        self.output_conv = nn.Conv2d(32, output_channels, kernel_size=1)
    
    def forward(self, x):
        enc_out = self.encoder(x)
        bridge_out = self.bridge(enc_out)
        dec_out = self.decoder(bridge_out)
        out = self.output_conv(dec_out)
        
        return out
    
# CNN Encoder to Vector Block to replace the EncoderDecoder network from earlier
class EncoderToVector(nn.Module):
    def __init__(self, input_channels=5, channels=[32, 64, 128, 256], N=5, 
                 bridge_channels=512, fc_hidden=256, dropout=0.5):
        super(EncoderToVector, self).__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.N = N
        self.bridge_channels = bridge_channels
        self.fc_hidden = fc_hidden
        self.dropout_rate = dropout
        
        self.encoder = EncoderBlock(input_channels=input_channels, channels=channels)
        self.bridge = Bridge(in_channels=channels[-1], out_channels=bridge_channels)
        
        # Global average pooling reduces spatial dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Configurable FC layers
        self.fc1 = nn.Linear(bridge_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, N)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, input_channels, Y, Y]
        enc_out = self.encoder(x)
        bridge_out = self.bridge(enc_out)
        pooled = self.gap(bridge_out)  # [batch_size, bridge_channels, 1, 1]
        flat = self.flatten(pooled)     # [batch_size, bridge_channels]
        x = self.relu(self.fc1(flat))
        x = self.dropout(x)
        out = self.fc2(x)
        return out
    
    @classmethod
    def load(cls, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        
        model = cls(
            input_channels=checkpoint['input_channels'],
            channels=checkpoint.get('channels', [32, 64, 128, 256]),
            N=checkpoint['N'],
            bridge_channels=checkpoint.get('bridge_channels', 512),
            fc_hidden=checkpoint.get('fc_hidden', 256),
            dropout=checkpoint.get('dropout', 0.5)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        X_normalizer = NormalizationHandler.from_state(checkpoint['X_normalizer_state'])
        y_normalizer = NormalizationHandler.from_state(checkpoint['y_normalizer_state'])
        
        return model, X_normalizer, y_normalizer
import torch
import torch.nn as nn

import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self, input_channels = 1):
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
    
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
    def __init__(self):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.bridge(x)

class EncoderDecoderNetwork(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(EncoderDecoderNetwork, self).__init__()
        self.encoder = EncoderBlock(input_channels=input_channels)
        self.bridge = Bridge()
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
    def __init__(self, input_channels=1, N=10):
        super(EncoderToVector, self).__init__()
        self.encoder = EncoderBlock(input_channels=input_channels)
        self.bridge = Bridge()
        
        # Calculate the spatial dimensions after encoder
        # For 256x256 input: after 4 poolings with stride 2, 16x16
        # With 512 channels: 512 * 16 * 16 = 131072
        self.flatten = nn.Flatten()
        
        # Adjust this based on your actual input size
        # If input is HxW, after 4 stride-2 poolings: (H/16) * (W/16) * 512
        self.fc1 = nn.Linear(512 * 16 * 16, 512)  # Adjust first dimension
        self.fc2 = nn.Linear(512, N)  # Output N values
        self.relu = nn.ReLU()
    
    def forward(self, x):
        enc_out = self.encoder(x)
        bridge_out = self.bridge(enc_out)
        flat = self.flatten(bridge_out)
        x = self.relu(self.fc1(flat))
        out = self.fc2(x)
        return out
import torch
import torch.nn as nn

import numpy as np
from us_lib.data.normalization import NormalizationHandler

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, layer_size, output_size, dropout=0.1, use_batch_norm=True):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
       
        # self.activation = nn.ReLU()
        self.activation = nn.LeakyReLU()
        # self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
       
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
        self.layers.extend([nn.Linear(layer_size, layer_size) for _ in range(1, self.num_layers-1)])
        self.layers.append(nn.Linear(layer_size, output_size))
        
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(layer_size) for _ in range(self.num_layers-1)])
   
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
    @classmethod
    def load(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = cls(
            input_size=checkpoint['input_size'],
            num_layers=checkpoint['num_layers'],
            layer_size=checkpoint['layer_size'],
            output_size=checkpoint['output_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        X_normalizer = NormalizationHandler.from_state(checkpoint['X_normalizer_state'])
        y_normalizer = NormalizationHandler.from_state(checkpoint['y_normalizer_state'])

        return model, X_normalizer, y_normalizer

def weighted_mse_loss(predictions, targets, weights):
        """MSE loss weighted by POD mode importance"""
        squared_errors = (predictions - targets) ** 2
        weighted_errors = squared_errors * weights.unsqueeze(0)
        return weighted_errors.mean()
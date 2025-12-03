# Generic Imports
import os
import string
from venv import create
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
np.set_printoptions(linewidth=200)
from datetime import datetime

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from torchinfo import summary

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from scipy.linalg import svd

from data_loader import data_loader, create_cnn_matrix

# Personal Definitions
from us_lib.models.mlp import MLP, weighted_mse_loss
from us_lib.models.cnn import EncoderBlock, Bridge, EncoderToVector
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.models.pod import training_data_constructor, plot_field
from us_lib.data.parsing import extract_attributes
from us_lib.data.samples import iterate_folds, load_folds

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CNN Training Script')
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=bool, default=0,
                        help='Save (default: 0)')
    parser.add_argument('--path', type=str, default='./test_data/set_1',
                        help='Path to trial data relative to create_cnn.py')
    parser.add_argument('--verbose', type=bool, default=0,
                        help='Print the structure of the network (default: 0)')
    
    return parser.parse_args()


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Parse command line arguments
    args = parse_args()

    print(f"Training Path: {args.path}\n")
    
    print(f"Training with configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}\n")
    
    # Define input and output data locations
    input_path = Path(f"{args.path}/input.jsonl")
    output_path = Path(f"{args.path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    X, y = data_loader(args.path)
    mask_matrix = load_folds('folds_r0.npy')
    parameter_names = ["t_panel", "t_longitudinal_web", "t_longitudinal_flange", "t_transverse_web", "t_transverse_flange"]
    
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    # -------------------------------------------------------------------------------------------------------------------------
    # Create the convolutional input layers
    # -------------------------------------------------------------------------------------------------------------------------
    convolution_size = 80
    X = create_cnn_matrix(X, convolution_size, convolution_size)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # -------------------------------------------------------------------------------------------------------------------------
    # Create Model
    # -------------------------------------------------------------------------------------------------------------------------
    model = EncoderToVector(input_channels=num_features, N=y.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.verbose:
        summary(model, input_size=(1, num_features, convolution_size, convolution_size))
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {num_params:,}\n")

    # -------------------------------------------------------------------------------------------------------------------------
    # Define the optimizer and the loss function
    # -------------------------------------------------------------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    writer = SummaryWriter(log_dir=f"./cnn/runs/{timestamp}")

    # -------------------------------------------------------------------------------------------------------------------------
    # Normalize the data and train using k-fold cross validation
    # -------------------------------------------------------------------------------------------------------------------------
    X_normalizer = NormalizationHandler(method='std', excluded_axis=[1])
    y_normalizer = NormalizationHandler(method='std', excluded_axis=[1])

    average_error = 0
    best_model = None
    best_model_error = np.inf
    
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(iterate_folds(X, y, mask_matrix)):
        print(f"\nFold {fold_idx + 1}/{mask_matrix.shape[1]}")
        
        # Normalize data
        X_train = X_normalizer.fit_normalize(X_train)
        y_train = y_normalizer.fit_normalize(y_train)
        X_val = X_normalizer.normalize(X_val)
        y_val = y_normalizer.normalize(y_val)

        # Convert to tensors
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).float().to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Training loop for this fold
        model.train()
        for epoch in tqdm(range(args.epochs), desc=f"Fold {fold_idx + 1}"):
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for this epoch
            avg_loss = epoch_loss / len(train_loader)
            
            # Log to tensorboard
            writer.add_scalar("Loss/Train", avg_loss, fold_idx * args.epochs + epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            average_error += val_loss.item()
            
            print(f"Fold {fold_idx + 1} Validation Loss: {val_loss.item():.6f}")

        # Reset normalizers for next fold
        X_normalizer.soft_reset()
        y_normalizer.soft_reset()
        writer.flush()

        # Track best model
        if best_model is None or val_loss.item() < best_model_error:
            best_model_error = val_loss.item()
            # Save best model state
            best_model_state = model.state_dict().copy()

    average_error = average_error / (fold_idx + 1)
    print(f"\n{'='*60}")
    print(f"Average validation error across all folds: {average_error:.6f}")
    print(f"Best model validation error: {best_model_error:.6f}")
    print(f"{'='*60}\n")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final test on last validation set
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        
        # Denormalize for comparison
        outputs_denorm = y_normalizer.denormalize(outputs.cpu().numpy())
        y_val_denorm = y_normalizer.denormalize(y_val.cpu().numpy())
        
        print("Sample Predictions vs Actual (first 5 samples):")
        print(f"{'Predicted':<50} {'Actual':<50}")
        print("-" * 100)
        for i in range(min(5, len(outputs_denorm))):
            pred_str = np.array2string(outputs_denorm[i], precision=3, suppress_small=True)
            actual_str = np.array2string(y_val_denorm[i], precision=3, suppress_small=True)
            print(f"{pred_str:<50} {actual_str:<50}")

    # Save model if requested
    if args.save == 1:
        os.makedirs('models', exist_ok=True)
        save_path = f'models/cnn_model_{timestamp}.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'input_channels': num_features,
            'output_size': y.shape[1],
            'convolution_size': convolution_size,
            'best_val_error': best_model_error,
            'avg_val_error': average_error,
        }, save_path)
        print(f"\nModel saved to: {save_path}")

    writer.close()

if __name__ == '__main__':
    main()
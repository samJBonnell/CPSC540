# Generic Imports
import os
import string
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

from data_loader import data_loader

# Personal Definitions
from us_lib.models.mlp import MLP, weighted_mse_loss
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.models.pod import training_data_constructor, plot_field
from us_lib.data.parsing import extract_attributes
from us_lib.data.samples import iterate_folds, load_folds

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='POD-MLP Training Script')
    
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers in the MLP (default: 4)')
    parser.add_argument('--layer_size', type=int, default=16,
                        help='Size of each hidden layer (default: 16)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=bool, default=0,
                        help='Save (default: 0)')
    parser.add_argument('--path', type=str, default='./test_data/set_1',
                        help='Path to trial data relative to pod-mlp.py')
    parser.add_argument('--verbose', type=bool, default=0,
                        help='Print the structure of the network (default: 0)')
    
    return parser.parse_args()

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Parse command line arguments
    args = parse_args()

    print(f"Training Path: {args.path}\n")
    
    print(f"Training with configuration:")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Layer size: {args.layer_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    
    # Define input and output data locations
    input_path = Path(f"{args.path}/input.jsonl")
    output_path = Path(f"{args.path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    X, y = data_loader(args.path)
    mask_matrix = load_folds('folds_r0.npy')

    parameter_names = ["t_panel", "t_longitudinal_web", "t_longitudinal_flange", "t_transverse_web", "t_transverse_flange"]

    # -------------------------------------------------------------------------------------------------------------------------
    # Normalize the data !AFTER! we split the data
    # -------------------------------------------------------------------------------------------------------------------------
    X_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])
    y_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])

    # -------------------------------------------------------------------------------------------------------------------------
    # Create MLP object
    # -------------------------------------------------------------------------------------------------------------------------

    model = MLP(input_size=len(parameter_names), num_layers=args.num_layers, layers_size=args.layer_size, output_size=5, dropout=0.05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # -------------------------------------------------------------------------------------------------------------------------
    # Define the optimizer and the loss function
    # -------------------------------------------------------------------------------------------------------------------------

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    if args.verbose:
        summary(model, input_size=(250, 5))
    writer = SummaryWriter(log_dir=f"./mlp/runs/{timestamp}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {num_params:,}\n")

    average_error = 0
    best_model = None
    best_model_error = np.inf
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(iterate_folds(X, y, mask_matrix)):
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

        # Training loop
        model.train()
        for epoch in tqdm(range(args.epochs)):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log the loss for THIS epoch only
            writer.add_scalar("Loss/Train", loss.item(), fold_idx * args.epochs + epoch)

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val)
            average_error += val_loss.item()

        X_normalizer.soft_reset()
        y_normalizer.soft_reset()
        writer.flush()

        if best_model == None:
            best_model = model
            best_model_error = val_loss.item()
        elif val_loss.item() < best_model_error:
            best_model = model
            best_model_error = val_loss.item()

    average_error = average_error / (fold_idx + 1)
    print(f"Average validation error: {average_error}")

    # Final test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        input = X_val.to(device)
        labels = y_val.to(device)
    
        # Forward pass
        outputs = model(input)
        print(f"{outputs[15]}\t{y_val[15]}")
    # if args.save == 1:
    #     os.makedirs('models', exist_ok=True)
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'input_size': len(parameter_names),
    #         'num_layers': args.num_layers,
    #         'layer_size': args.layer_size,
    #         'output_size': args.num_modes,
    #     }, f'models/model_epoch_{epoch}_{timestamp}.pth')

if __name__ == '__main__':
    main()
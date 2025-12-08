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

from sklearn.model_selection import train_test_split, KFold
from scipy.linalg import svd

from data_loader import data_loader, create_cnn_matrix

# Personal Definitions
from us_lib.models.mlp import MLP, weighted_mse_loss
from us_lib.models.cnn import EncoderBlock, Bridge, EncoderToVector
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.models.pod import training_data_constructor, plot_field
from us_lib.data.parsing import extract_attributes

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CNN Training Script')
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=int, default=0,
                        help='Save model (default: 0)')
    parser.add_argument('--path', type=str, default='./test_data/set_1',
                        help='Path to trial data relative to create_cnn.py')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Print the structure of the network and test results (default: 0)')
    parser.add_argument('--model_name', type=str, default='0',
                        help='The name under which the model will be saved (default: 0)')
    parser.add_argument('--use_cv', type=int, default=0,
                        help='Use cross-validation for hyperparameter tuning (default: 0)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to hold out for final test set (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducible splits (default: 42)')
    parser.add_argument('--conv_size', type=int, default=80,
                        help='Size of convolutional input matrix (default: 80)')
    
    # Architecture hyperparameters
    parser.add_argument('--channels', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Channel sizes for encoder blocks (default: 32 64 128 256)')
    parser.add_argument('--bridge_channels', type=int, default=512,
                        help='Number of channels in bridge layer (default: 512)')
    parser.add_argument('--fc_hidden', type=int, default=256,
                        help='Hidden dimension for fully connected layer (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    return parser.parse_args()

def train_model(model, train_loader, criterion, optimizer, device, epochs, writer=None, epoch_offset=0, desc="Training"):
    """Train a model and return training history"""
    model.train()
    history = []
    
    for epoch in tqdm(range(epochs), desc=desc):
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        
        if writer:
            writer.add_scalar("Loss/Train", avg_loss, epoch_offset + epoch)
    
    return history

def evaluate_model(model, X_val, y_val, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    return val_loss.item()

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args = parse_args()

    print(f"Training Path: {args.path}\n")
    print(f"Training with configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Convolution size: {args.conv_size}x{args.conv_size}")
    print(f"  - Architecture:")
    print(f"    - Encoder channels: {args.channels}")
    print(f"    - Bridge channels: {args.bridge_channels}")
    print(f"    - FC hidden: {args.fc_hidden}")
    print(f"    - Dropout: {args.dropout}")
    print(f"  - Use cross-validation: {bool(args.use_cv)}")
    if args.use_cv:
        print(f"  - Number of folds: {args.n_folds}")
    print(f"  - Test set size: {args.test_size}")
    print(f"  - Random state: {args.random_state}")
    
    # Define input and output data locations
    input_path = Path(f"{args.path}/input.jsonl")
    output_path = Path(f"{args.path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    X, y = data_loader(args.path)
    parameter_names = ["t_panel", "t_longitudinal_web", "t_longitudinal_flange", "t_transverse_web", "t_transverse_flange"]
    
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    print(f"\nTotal samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print(f"Output size: {y.shape[1]}")
    
    # Create the convolutional input matrices
    X = create_cnn_matrix(X, args.conv_size, args.conv_size)
    print(f"CNN input shape: {X.shape}\n")
    
    # Step 1: Hold out test set
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"Test set (held out): {X_test.shape[0]} samples")
    print(f"Training pool: {X_remaining.shape[0]} samples\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=f"./cnn/runs/{timestamp}")

    if args.use_cv:
        # Step 2: Cross-validation on training pool
        print("="*80)
        print("PHASE 1: Cross-Validation for Hyperparameter Validation")
        print("="*80)
        
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
        fold_errors = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_remaining)):
            print(f"\nFold {fold_idx + 1}/{args.n_folds}")
            
            # Split data using indices from KFold
            X_train = X_remaining[train_idx]
            X_val = X_remaining[val_idx]
            y_train = y_remaining[train_idx]
            y_val = y_remaining[val_idx]
            
            print(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
            
            # Create new model for this fold with configurable architecture
            model = EncoderToVector(
                input_channels=num_features,
                channels=args.channels,
                N=y.shape[1],
                bridge_channels=args.bridge_channels,
                fc_hidden=args.fc_hidden,
                dropout=args.dropout
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
            
            # Create fresh normalizers for this fold
            X_normalizer = NormalizationHandler(method='std', excluded_axis=[1])
            y_normalizer = NormalizationHandler(method='std', excluded_axis=[1])
            
            # Normalize data
            X_train_norm = X_normalizer.fit_normalize(X_train)
            y_train_norm = y_normalizer.fit_normalize(y_train)
            X_val_norm = X_normalizer.normalize(X_val)
            y_val_norm = y_normalizer.normalize(y_val)
            
            # Convert to tensors
            X_train_t = torch.from_numpy(X_train_norm).float().to(device)
            y_train_t = torch.from_numpy(y_train_norm).float().to(device)
            X_val_t = torch.from_numpy(X_val_norm).float().to(device)
            y_val_t = torch.from_numpy(y_val_norm).float().to(device)
            
            # Create data loader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            # Train
            train_model(model, train_loader, criterion, optimizer, device,
                       args.epochs, writer, fold_idx * args.epochs, f"Fold {fold_idx + 1}")
            
            # Evaluate
            val_loss = evaluate_model(model, X_val_t, y_val_t, criterion, device)
            fold_errors.append(val_loss)
            print(f"  Validation Loss: {val_loss:.6f}")
        
        # Report CV results
        mean_cv_error = np.mean(fold_errors)
        std_cv_error = np.std(fold_errors)
        
        print(f"\n{'='*80}")
        print(f"Cross-Validation Results:")
        print(f"  Mean CV Error: {mean_cv_error:.6f} (+/- {std_cv_error:.6f})")
        print(f"  Individual Fold Errors: {[f'{e:.6f}' for e in fold_errors]}")
        print(f"{'='*80}\n")
    
    # Step 3: Train final model on ALL training data
    print("="*80)
    print("PHASE 2: Training Final Model on All Training Data")
    print("="*80)
    
    # Create fresh model for final training with configurable architecture
    final_model = EncoderToVector(
        input_channels=num_features,
        channels=args.channels,
        N=y.shape[1],
        bridge_channels=args.bridge_channels,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout
    ).to(device)
    
    if args.verbose:
        summary(final_model, input_size=(1, num_features, args.conv_size, args.conv_size))
    
    num_params = sum(p.numel() for p in final_model.parameters())
    print(f"\nNumber of parameters: {num_params:,}")
    print(f"Training on {X_remaining.shape[0]} samples\n")
    
    # Create fresh normalizers for final training
    X_normalizer_final = NormalizationHandler(method='std', excluded_axis=[1])
    y_normalizer_final = NormalizationHandler(method='std', excluded_axis=[1])
    
    # Normalize all training data
    X_train_final = X_normalizer_final.fit_normalize(X_remaining)
    y_train_final = y_normalizer_final.fit_normalize(y_remaining)
    
    # Convert to tensors
    X_train_final_t = torch.from_numpy(X_train_final).float().to(device)
    y_train_final_t = torch.from_numpy(y_train_final).float().to(device)
    
    # Create data loader
    train_dataset_final = TensorDataset(X_train_final_t, y_train_final_t)
    train_loader_final = DataLoader(train_dataset_final, batch_size=args.batch_size, shuffle=True)
    
    # Train final model
    optimizer_final = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    train_model(final_model, train_loader_final, criterion, optimizer_final, device,
               args.epochs, writer, 0 if not args.use_cv else args.n_folds * args.epochs,
               "Final Training")
    
    # Step 4: Evaluate on held-out test set
    print("\n" + "="*80)
    print("PHASE 3: Final Evaluation on Held-Out Test Set")
    print("="*80)
    
    # Normalize test data using final training statistics
    X_test_norm = X_normalizer_final.normalize(X_test)
    y_test_norm = y_normalizer_final.normalize(y_test)
    
    X_test_t = torch.from_numpy(X_test_norm).float().to(device)
    y_test_t = torch.from_numpy(y_test_norm).float().to(device)
    
    # Final evaluation
    test_loss = evaluate_model(final_model, X_test_t, y_test_t, criterion, device)
    
    print(f"\nFinal Test Set Performance:")
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    print(f"  Test RMSE: {np.sqrt(test_loss):.6f}")
    print("="*80 + "\n")
    
    # Display sample predictions
    if args.verbose:
        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(X_test_t)
            
            # Denormalize for comparison
            outputs_denorm = y_normalizer_final.denormalize(test_outputs.cpu().numpy())
            y_test_denorm = y_normalizer_final.denormalize(y_test_t.cpu().numpy())
            
            print("Sample Test Set Predictions vs Actual (first 5 samples):")
            print(f"{'Predicted':<50} {'Actual':<50}")
            print("-" * 100)
            for i in range(min(5, len(outputs_denorm))):
                pred_str = np.array2string(outputs_denorm[i], precision=3, suppress_small=True)
                actual_str = np.array2string(y_test_denorm[i], precision=3, suppress_small=True)
                print(f"{pred_str:<50} {actual_str:<50}")
    
    # Save final model
    if args.save:
        os.makedirs('models', exist_ok=True)
        save_dict = {
            'model_state_dict': final_model.state_dict(),
            'input_channels': final_model.input_channels,
            'channels': final_model.channels,
            'N': final_model.N,
            'bridge_channels': final_model.bridge_channels,
            'fc_hidden': final_model.fc_hidden,
            'dropout': final_model.dropout_rate,
            'X_normalizer_state': X_normalizer_final.get_state(),
            'y_normalizer_state': y_normalizer_final.get_state(),
            'test_loss': test_loss,
            'random_state': args.random_state,
            'conv_size': args.conv_size
        }
        if args.use_cv:
            save_dict['cv_mean_error'] = mean_cv_error
            save_dict['cv_std_error'] = std_cv_error
            save_dict['fold_errors'] = fold_errors
            save_dict['n_folds'] = args.n_folds
        torch.save(save_dict, f'models/cnn_{args.model_name}.pth')
        print(f"Model saved to models/cnn_{args.model_name}.pth\n")
    
    writer.close()

if __name__ == '__main__':
    main()
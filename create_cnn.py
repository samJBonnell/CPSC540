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
    
    num_samples = X.shape[0]
    num_features = X.shape[1]
    # -------------------------------------------------------------------------------------------------------------------------
    # Create the convolutional input layers
    # -------------------------------------------------------------------------------------------------------------------------
    # Create the n x d x N_i x N_i parameter input space:
    convolution_size = 80
    X = create_cnn_matrix(X, convolution_size, convolution_size)

    encoder = EncoderBlock(input_channels=1)
    bridge = Bridge()

    # Use your actual input dimensions here
    test_input = torch.randn(1, 1, 256, 256)  # (batch, channels, height, width)

    with torch.no_grad():
        enc_out = encoder(test_input)
        bridge_out = bridge(enc_out)
        print(f"Shape after bridge: {bridge_out.shape}")
        # Example output: torch.Size([1, 512, 16, 16])
        
        flattened_size = bridge_out.shape[1] * bridge_out.shape[2] * bridge_out.shape[3]
        print(f"Flattened size for fc1: {flattened_size}")

#     # -------------------------------------------------------------------------------------------------------------------------
#     # Normalize the data !AFTER! we split the data
#     # -------------------------------------------------------------------------------------------------------------------------
#     X_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])
#     y_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])

#     # -------------------------------------------------------------------------------------------------------------------------
#     # Create MLP object
#     # -------------------------------------------------------------------------------------------------------------------------

#     model = MLP(input_size=len(parameter_names), num_layers=args.num_layers, layers_size=args.layer_size, output_size=5, dropout=0.05)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # -------------------------------------------------------------------------------------------------------------------------
#     # Define the optimizer and the loss function
#     # -------------------------------------------------------------------------------------------------------------------------

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

#     if args.verbose:
#         summary(model, input_size=(250, 5))
#     writer = SummaryWriter(log_dir=f"./mlp/runs/{timestamp}")

#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"\nNumber of parameters: {num_params:,}\n")

#     average_error = 0
#     best_model = None
#     best_model_error = np.inf
#     for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(iterate_folds(X, y, mask_matrix)):
#         # Normalize data
#         X_train = X_normalizer.fit_normalize(X_train)
#         y_train = y_normalizer.fit_normalize(y_train)
#         X_val = X_normalizer.normalize(X_val)
#         y_val = y_normalizer.normalize(y_val)

#         # Convert to tensors
#         X_train = torch.from_numpy(X_train).float().to(device)
#         y_train = torch.from_numpy(y_train).float().to(device)
#         X_val = torch.from_numpy(X_val).float().to(device)
#         y_val = torch.from_numpy(y_val).float().to(device)

#         # Training loop
#         model.train()
#         for epoch in tqdm(range(args.epochs)):
#             outputs = model(X_train)
#             loss = criterion(outputs, y_train)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # Log the loss for THIS epoch only
#             writer.add_scalar("Loss/Train", loss.item(), fold_idx * args.epochs + epoch)

#         # Evaluation
#         model.eval()
#         with torch.no_grad():
#             outputs = model(X_val)
#             val_loss = criterion(outputs, y_val)
#             average_error += val_loss.item()

#         X_normalizer.soft_reset()
#         y_normalizer.soft_reset()
#         writer.flush()

#         if best_model == None:
#             best_model = model
#             best_model_error = val_loss.item()
#         elif val_loss.item() < best_model_error:
#             best_model = model
#             best_model_error = val_loss.item()

#     average_error = average_error / (fold_idx + 1)
#     print(f"Average validation error: {average_error}")

#     # Final test
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         input = X_val.to(device)
#         labels = y_val.to(device)
    
#         # Forward pass
#         outputs = model(input)
#         print(f"{outputs[15]}\t{y_val[15]}")
#     # if args.save == 1:
#     #     os.makedirs('models', exist_ok=True)
#     #     torch.save({
#     #         'model_state_dict': model.state_dict(),
#     #         'input_size': len(parameter_names),
#     #         'num_layers': args.num_layers,
#     #         'layer_size': args.layer_size,
#     #         'output_size': args.num_modes,
#     #     }, f'models/model_epoch_{epoch}_{timestamp}.pth')

if __name__ == '__main__':
    main()

# # Print
# print("Prediction mse|Prediction mse+L1reg | Actual")
# print(comparison.detach().numpy())    
# # Data input!
# input_matrix_size = int(np.sqrt(len(stress_vectors[0])))
# y = np.zeros((len(stress_vectors), input_matrix_size, input_matrix_size))
# for i, vector in enumerate(stress_vectors):
#     np_vector = np.array(vector)
#     y[i, :, :] = np_vector.reshape((input_matrix_size, input_matrix_size)) / 1e6
# # Need to create a patterning for the CNN interface
# # Currently, we have rows and columns that correspond to the size of input of the CNN
# # The snapshots are N x n vectors, parameters are d x 1 vectors. We need to create a 
# # sqrt(N) x sqrt(N) x n block for snapshots and a sqrt(N) x sqrt(N) x d input for each n_i example.

# # Create the n x d x N_i x N_i parameter input space:
# input_convolution_size = 80
# template_convolution = np.ones(shape=(input_convolution_size, input_convolution_size), dtype=float)
# X = np.ndarray(shape=(num_samples, num_features, input_convolution_size, input_convolution_size))

# for i, parameter_set in enumerate(parameters):
#     # For each of the features, create an N_i x N_i input matrix that we set as the value of each feature across the entire matrix
#     for j, value in enumerate(parameter_set):
#         X[i, j, :, :] = (template_convolution.copy()) * value

# parameters = np.array(parameters)
# indices = np.arange(X.shape[0])
# # Create training and testing splits
# X_train, X_test, y_train, y_test, train_indicies, test_indicies = train_test_split(
#     X, y, indices, 
#     test_size = 0.2,
#     random_state=None
# )

# # -------------------------------------------------------------------------------------------------------------------------
# # Normalize the data !AFTER! we split the data
# # -------------------------------------------------------------------------------------------------------------------------
# # We need to normalize the X_train and then normalize the X_test with the same values
# X_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])
# y_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1, 2])

# # We need to noramlize the y_train and then normalize the y_test with the same values
# X_train = X_normalizer.fit_normalize(X_train)
# y_train = y_normalizer.fit_normalize(y_train)

# X_test = X_normalizer.normalize(X_test)
# y_test = y_normalizer.normalize(y_test)

# # Convert into torch compatible versions
# X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).float()

# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).float()

# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# # -------------------------------------------------------------------------------------------------------------------------
# # Define the optimizer and the loss function
# # -------------------------------------------------------------------------------------------------------------------------
# model = EncoderDecoderNetwork(input_channels=num_features, output_channels = 1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

# # We can now train a network!

# # -------------------------------------------------------------------------------------------------------------------------
# # Create the summary writer and Tensorboard writer
# # -------------------------------------------------------------------------------------------------------------------------
# if args.verbose:
#     summary(model, input_size=(1, 4, 80, 80))
# writer = SummaryWriter(log_dir=f"./cnn/runs/{timestamp}")

# num_params = sum(p.numel() for p in model.parameters())
# print(f"\nNumber of parameters: {num_params:,}\n")
# # -------------------------------------------------------------------------------------------------------------------------
# # Run the training of the model
# # -------------------------------------------------------------------------------------------------------------------------
# model.train()
# for epoch in tqdm(range(args.epochs)):
#     total_loss = 0

#     for input, labels in train_loader:
#         input = input.to(device)
#         labels = labels.to(device)

#         outputs = model(input)
#         outputs = outputs.squeeze(1)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss

#     avg_epoch_loss = total_loss / len(train_loader)
#     writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)

# writer.flush()

# # -------------------------------------------------------------------------------------------------------------------------
# # Evaluate the performance of the model
# # -------------------------------------------------------------------------------------------------------------------------
# model.eval()
# test_loss = 0
# with torch.no_grad():
#     for input, labels in test_loader:
#         input = input.to(device)
#         labels = labels.to(device)

#         outputs = model(input)
#         outputs = outputs.squeeze(1)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()

# avg_test_loss = test_loss / len(test_loader)
# print(f"\nTest Error: {avg_test_loss:.4f}")
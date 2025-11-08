# Generic Imports
import os
import string

import torch.optim.adam
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
np.set_printoptions(linewidth=200)
from datetime import datetime

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from sklearn.model_selection import train_test_split

# Personal Definitions
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.data.parsing import extract_attributes
from us_lib.models.cnn import EncoderDecoderNetwork
from us_lib.visuals import plot_field

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Machine Learning Base Model')
    
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
    
    # Define input and output data locations
    input_path = Path(f"{args.path}/input.jsonl")
    output_path = Path(f"{args.path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    # Load data records
    records = load_records(input_path, output_path)
    records, eigenvalues = extract_attributes(records, attributes= ['eigenvalue'])
    eigenvalues = eigenvalues['eigenvalue']

    # Extract parameters
    parameters = []
    for rec in records:
        row = [
            rec.input.t_panel,                    
            rec.input.t_longitudinal_web,       
            rec.input.t_longitudinal_flange,
            rec.input.h_longitudinal_web,
            rec.input.w_longitudinal_flange
        ]
        parameters.append(row)

    X = np.array(parameters, dtype=float)
    y = np.array(eigenvalues, dtype=float)

    '''
    Beyond this point, we have a dataset that operates as we have defined through the rest of the class:
        X : n x d (samples x dimensions)
        y : n x 1 (samples x eigenvalues)
    '''

    print(X.shape)
    print(y.shape)

if __name__ == '__main__':
    main()
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

# Personal Definitions
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.data.parsing import extract_attributes
from us_lib.models.cnn import EncoderDecoderNetwork
from us_lib.visuals import plot_field
from us_lib.data.samples import create_folds, iterate_folds
from data_loader import data_loader

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Creation of folds for entire project')
    
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
    args = parse_args()
    X, y = data_loader(args.path)

    mask_matrix = create_folds(X, 10, save_to_file=True, file_name='test')
    with np.printoptions(threshold=np.inf, linewidth=np.inf):
        print(mask_matrix)

if __name__ == '__main__':
    main()
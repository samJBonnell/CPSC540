# Generic Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
np.set_printoptions(linewidth=200)
from datetime import datetime
from pathlib import Path

# ML Imports
import torch
from data_loader import data_loader, create_cnn_matrix

# Personal Definitions
from us_lib.models.cnn import EncoderToVector
from us_lib.models.mlp import MLP

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run ML Model')
    parser.add_argument('--path', type=str, default='./test_data/set_1',
                        help='Path to trial data relative to create_cnn.py')
    parser.add_argument('--model_name', type=str, default='0',
                        help='The name under which the model will be saved (default: 0)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Define input and output data locations
    input_path = Path(f"{args.path}/input.jsonl")
    output_path = Path(f"{args.path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    X, y = data_loader(args.path)
    
    num_samples = X.shape[0]
    num_features = X.shape[1]

    if "cnn" in args.model_name:
        convolution_size = 80
        X = create_cnn_matrix(X, convolution_size, convolution_size)
        model, X_normalizer, y_normalizer = EncoderToVector.load(f'models/{args.model_name}.pth')
    elif "mlp" in args.model_name:
        model, X_normalizer, y_normalizer= MLP.load(f'models/{args.model_name}.pth')
    else:
        raise ValueError(f"{args.model_name} is not a viable model. Must contain 'cnn' or 'mlp'.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize the data using the same model as before
    X = X_normalizer.normalize(X)
    y = y_normalizer.normalize(y)

    # Convert to tensors
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    # Set the model to evaluation mode and send it to the device
    model.eval()
    model = model.to(device)

    max_index = num_samples - 1
    while True:
        user_input = input(f"Enter index (0-{max_index}) or 'q' to quit: ").strip()
        
        if user_input.lower() in ['q', 'quit']:
            break
        
        if not user_input.isdigit():
            print("Please enter a valid number")
            continue
        
        index = int(user_input)
        
        if index > max_index:
            print(f"Index too large. Max is {max_index}")
        else:
            with torch.no_grad():
                outputs = model(X[index, :].unsqueeze(0))

                # Denormalize for comparison
                outputs_denorm = y_normalizer.denormalize(outputs.cpu().numpy())
                y_denorm = y_normalizer.denormalize(y[index, :].unsqueeze(0).cpu().numpy())
                
                print("Sample Predictions vs Actual:")
                print(f"{'Predicted':<50} {'Actual':<50}")
                print("-" * 100)
                for i in range(min(5, len(outputs_denorm[0]))):
                    print(f"{outputs_denorm[0][i]:<50.3f} {y_denorm[0][i]:<50.3f}")
        print("\n")

if __name__ == '__main__':
    main()
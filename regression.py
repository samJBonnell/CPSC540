import torch
import argparse
import numpy as np
from tqdm import tqdm

from data_loader import data_loader
from us_lib.models.regression import LinearModel, make_loss
from us_lib.data.samples import create_folds, load_folds, iterate_folds
from us_lib.data.normalization import NormalizationHandler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='POD-MLP Training Script')
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=bool, default=0,
                        help='Save (default: 0)')
    parser.add_argument('--path', type=str, default='./data/test/non-var-thickness',
                        help='Path to trial data relative to pod-mlp.py')
    parser.add_argument('--verbose', type=bool, default=0,
                        help='Print the structure of the network (default: 0)')
    
    return parser.parse_args()

#############################################################
# Load Data and Create Mask#
#############################################################
args = parse_args()
X, y = data_loader(args.path)

# Create normalization handlers
X_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])
y_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])

mask_matrix = load_folds('folds_r0.npy')

#############################################################
# Linear Regresssion Models #
#############################################################

# Hyperparameters
learning_rate = args.learning_rate
loss_type = 'mse'
l1_reg = 0.001
l2_reg = 0.001
epochs = args.epochs

error = 0
for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(iterate_folds(X, y, mask_matrix)):
    # We need to noramlize the y_train and then normalize the y_val with the same values
    X_train = X_normalizer.fit_normalize(X_train)
    y_train = y_normalizer.fit_normalize(y_train)

    X_val = X_normalizer.normalize(X_val)
    y_val = y_normalizer.normalize(y_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train[:,1], dtype=torch.float32).reshape(-1,1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val[:,1], dtype=torch.float32).reshape(-1,1)

    n, d = X_train.shape

    model = LinearModel(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Adam optimizer to reduce weights

    # create loss function
    loss_fn = make_loss(loss_type=loss_type, l1_reg=l1_reg, l2_reg=l2_reg, model=model)

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    pred = model(X_val)

    mse = torch.mean((pred - y_val)**2).item()
    error += mse

    X_normalizer.soft_reset()
    y_normalizer.soft_reset()

print(f"MSE average: {error/fold_idx}")
print(f"X: {X_val[15]}, y: {y_val[15]}, y_pred: {model(X_val[15])}")

comparison = torch.cat([pred, y_val], dim=1)

# Print
print("Prediction mse|Prediction mse+L1reg | Actual")
print(comparison.detach().numpy())
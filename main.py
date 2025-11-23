# main code to run models

import torch
import numpy as np

from us_lib.models.regression import train_model
from data_loader import data_loader
from us_lib.data.samples import create_folds

#############################################################
# Load Data and Create Mask#
#############################################################
X, y = data_loader("./test_data/set_1")

#############################################################
# Linear Regresssion Models #
#############################################################
# L2 regression (ridge)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y[:,1], dtype=torch.float32).reshape(-1, 1)
new_X = torch.tensor(X[200:300, :], dtype=torch.float32)
new_y = torch.tensor(y[200:300], dtype=torch.float32).reshape(-1, 1)


model = train_model(X, y, l2_reg=0.01)
model_mse_l1_reg = train_model(X, y, l1_reg=0.01)

# Predict new sample
pred = model(new_X)
pred_mse_l1_reg = model_mse_l1_reg(new_X)
#print("Predictions:", pred)
mse = torch.mean((pred - new_y)**2).item()
#print("MSE:", mse)
comparison = torch.cat([pred, pred_mse_l1_reg, new_y], dim=1)  # dim=1 → concatenate columns

# Print
print("Prediction mse|Prediction mse+L1reg | Actual")
print(comparison.detach().numpy())



#############################################################
# Test linear model #
#############################################################
# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Number of samples and features
n_samples = 100
n_features = 3

# Generate random features
X = np.random.rand(n_samples, n_features)

# True weights
true_w = np.array([2.0, -3.5, 1.0])
true_b = 0.5

# Generate linear target with small noise
y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1  # small Gaussian noise

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Print shapes and a few samples
print("X shape:", X_tensor.shape)
print("y shape:", y_tensor.shape)
print("First 5 X samples:\n", X_tensor[:5])
print("First 5 y samples:\n", y_tensor[:5])

model = train_model(X_tensor, y_tensor, l2_reg=0.01)
model_mse_l1_reg = train_model(X_tensor, y_tensor, l2_reg=0.0)
# Predict new sample
pred = model(X_tensor)

#print("Trained weights:", model.weight.data)
#print("Trained bias:", model.bias.data)

mse = torch.mean((pred - y_tensor)**2).item()
pred_mse_l1_reg = model_mse_l1_reg(X_tensor)
#print("MSE:", mse)
comparison = torch.cat([pred, pred_mse_l1_reg, y_tensor], dim=1)  # dim=1 → concatenate columns

# Print
print("Prediction mse|Prediction mse+L1reg | Actual")
print(comparison.detach().numpy())

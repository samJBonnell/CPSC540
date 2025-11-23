# Linear Models 

import torch
import torch.nn as nn

import numpy as np

class LinearModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True) # define weight vector with bias 

    def forward(self, x): #type of model for pytorch 
        return self.linear(x)


def make_loss(loss_type="mse", l1_reg=0.0, l2_reg=0.0, model=None):  # define loss function 
    """
    loss_type: "mse" (mean squared error) or "l1" (L1 norm of error)
    l1_reg: weight for L1 regularization
    l2_reg: weight for L2 regularization
    """

    if loss_type == "mse":
        base_loss_fn = nn.MSELoss()
    elif loss_type == "l1":
        base_loss_fn = nn.L1Loss()
    else:
        # make custom error function
        raise ValueError("loss_type must be 'mse' or 'l1'")

    def loss_fn(pred, target): #pred = ... and target = ... 
        # Base loss
        loss = base_loss_fn(pred, target)

        # L1 regularization (sum of absolute weights)
        if l1_reg > 0:
            l1_penalty = sum(torch.sum(torch.abs(p)) for p in model.parameters())
            loss += l1_reg * l1_penalty

        # L2 regularization (sum of squared weights)
        if l2_reg > 0:
            l2_penalty = sum(torch.sum(p**2) for p in model.parameters())
            loss += l2_reg * l2_penalty

        return loss
    return loss_fn #retrun the loss funtion 


def train_model(X, y, loss_type="mse", l1_reg=0.0, l2_reg=0.0, lr=0.01, epochs=500):
    n, d = X.shape

    model = LinearModel(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Adam optimizer to reduce weights

    # create loss function
    loss_fn = make_loss(loss_type=loss_type, l1_reg=l1_reg, l2_reg=l2_reg, model=model)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        #if epoch % 100 == 0: # show loss every 100 steps 
        #    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return model

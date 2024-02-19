import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def train(
    model:nn.Module,
    criterion:callable,
    optimizer:torch.optim.Optimizer,
    data_loader:DataLoader,
    device:str
) -> float:
    '''train one epoch
    
    Args:
        model: model
        criterion: loss
        optimizer: optimizer
        data_loader: data loader
        device: device
    '''
    model.train()
    total_loss = 0.
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(data_loader)
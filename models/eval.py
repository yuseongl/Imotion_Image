import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from typing import Optional

def evaluate(
    model:nn.Module,
    criterion:callable,
    data_loader:DataLoader,
    device:str,
    metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
    '''evaluate
    
    Args:
        model: model
        criterions: list of criterion functions
        data_loader: data loader
        device: device
    '''
    model.eval()
    correct = 0
    total_loss = 0.
    total = 0
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            total_loss += criterion(outputs, y).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
                
            if metric is not None:
                #output = torch.round(output)
                metric.update_state(outputs, y)

    total_loss = total_loss/len(data_loader)
    accuracy = get_metric(correct ,total )
    
    return total_loss, accuracy

def get_metric(correct,  total):
    # Calculate accuracy
    accuracy = 100 * correct / total # 정확도 계산
    # Print and store test loss
    return accuracy
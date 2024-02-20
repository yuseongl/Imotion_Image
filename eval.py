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
    correct_top5 = 0.
    total_loss = 0.
    total = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            total_loss += criterion(outputs, y).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_top5 += top5_pred.eq(y.view(1, -1).expand_as(top5_pred)).reshape(-1).sum().item()
            
            if metric is not None:
                #output = torch.round(output)
                metric.update_state(outputs, y)

    total_loss = total_loss/len(data_loader)
    accuracy, accuracy_top5 = get_metric(correct ,total, correct_top5)
    
    
    
    return total_loss, accuracy, all_labels, all_predictions, accuracy_top5

def get_metric(correct, total, correct_top5):
    # Calculate accuracy
    accuracy = 100 * correct / total # 정확도 계산
    accuracy_top5 = 100 * correct_top5 / total
    
    return accuracy, accuracy_top5
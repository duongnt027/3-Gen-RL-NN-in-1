from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader
import math

def tensor_pop(tensor: torch.Tensor, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove and return element at index from tensor"""
    if tensor.size(0) == 0:
        raise IndexError("Cannot pop from empty tensor")
    
    element = tensor[index:index+1]
    remaining = torch.cat([tensor[:index], tensor[index+1:]], dim=0)
    return element.squeeze(0), remaining


def report_generator(model, test_ds, device="cpu"):
    """Evaluate VAE reconstruction quality"""
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1).float()
            
            x_recon, mean, logvar = model.forward(x_batch, y_batch)
            loss = model.loss_fn(x_batch, x_recon, mean, logvar)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"VAE Test Loss: {avg_loss:.4f}")
    return avg_loss


def report_detector(model, test_ds, device="cpu"):
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    # print(len(test_ds))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model.forward(x_batch).squeeze()
            all_preds.append(y_pred.cpu())
            all_labels.append(y_batch.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    pred_binary = (all_preds > 0.5).float()
    
    accuracy = (pred_binary == all_labels).float().mean()
    
    y_true = all_labels.numpy().astype(int)
    y_pred = pred_binary.numpy().astype(int)
    
    print("=" * 60)
    print("DETECTOR PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print()
    
    target_names = ['Normal (0)', 'Anomaly (1)']
    report = classification_report(y_true, y_pred, 
                                 target_names=target_names, 
                                 digits=4,
                                 zero_division=0)
    print("Classification Report:")
    print(report)
    print()

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    
    precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0
    
    precision_anomaly = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_anomaly = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_anomaly = 2 * (precision_anomaly * recall_anomaly) / (precision_anomaly + recall_anomaly) if (precision_anomaly + recall_anomaly) > 0 else 0
    
    results = {
        'accuracy': accuracy.item(),
        'precision_normal': precision_normal,
        'recall_normal': recall_normal,
        'f1_normal': f1_normal,
        'precision_anomaly': precision_anomaly,
        'recall_anomaly': recall_anomaly,
        'f1_anomaly': f1_anomaly,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': all_preds.numpy()
    }
    
    return results

def split_dataset(dataset_X, dataset_y, ratios=[70, 30]):
    assert sum(ratios) == 100, "Tổng tỷ lệ phải bằng 100"
    assert len(ratios) >= 1, "Phải có ít nhất 1 split"
    
    anor_indices = (dataset_y == 1).nonzero(as_tuple=True)[0]
    nor_indices = (dataset_y == 0).nonzero(as_tuple=True)[0]
    
    if len(anor_indices) < len(ratios) or len(nor_indices) < len(ratios):
        raise ValueError("Mỗi class phải có ít nhất 1 sample cho mỗi split")
    
    def split_indices(indices, ratios):
        """Chia indices theo tỷ lệ"""
        n_total = len(indices)
        splits = []
        start_idx = 0
        
        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                end_idx = n_total
            else:
                n_samples = int(n_total * ratio / 100)
                end_idx = start_idx + n_samples
            
            splits.append(indices[start_idx:end_idx])
            start_idx = end_idx
        
        return splits
    
    anor_splits = split_indices(anor_indices, ratios)
    nor_splits = split_indices(nor_indices, ratios)
    
    X_splits = []
    y_splits = []
    
    for anor_idx, nor_idx in zip(anor_splits, nor_splits):
        combined_indices = torch.cat([anor_idx, nor_idx])
        
        X_split = dataset_X[combined_indices]
        y_split = dataset_y[combined_indices]
        
        X_splits.append(X_split)
        y_splits.append(y_split)
    
    return X_splits, y_splits
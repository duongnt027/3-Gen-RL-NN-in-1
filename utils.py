import math
import torch
import numpy as np

def split_dataset(dataset_X, dataset_y, ratios=[70, 30], min_samples_per_split=2):
    assert sum(ratios) == 100, "Split ratios must sum to 100"
    assert len(ratios) >= 1, "Must have at least one split"
    
    X_anor = dataset_X[dataset_y == 1]
    X_nor = dataset_X[dataset_y == 0]
    y_anor = dataset_y[dataset_y == 1]
    y_nor = dataset_y[dataset_y == 0]
    
    total_samples = len(dataset_X)
    min_required = len(ratios) * min_samples_per_split
    
    if total_samples < min_required:
        raise ValueError(f"Not enough samples ({total_samples}) for {len(ratios)} splits with minimum {min_samples_per_split} samples each")
    
    if len(X_anor) < len(ratios) or len(X_nor) < len(ratios):
        raise ValueError("Each class must have at least 1 sample per split")
    
    def compute_balanced_indices(length, ratios, min_per_split=1):
        n_splits = len(ratios)
        
        counts = [min_per_split] * n_splits
        remaining = length - sum(counts)
        
        if remaining < 0:
            raise ValueError("Not enough samples to satisfy minimum requirements")
        
        if remaining > 0:
            total_ratio = sum(ratios)
            for i, ratio in enumerate(ratios):
                additional = math.floor(remaining * ratio / total_ratio)
                counts[i] += additional
            
            leftover = remaining - sum(counts[min_per_split:])
            for i in range(leftover):
                counts[i % n_splits] += 1
        
        indices = [0]
        for count in counts:
            indices.append(indices[-1] + count)
        
        return indices
    
    idx_X_anor = compute_balanced_indices(len(X_anor), ratios, min_per_split=1)
    idx_X_nor = compute_balanced_indices(len(X_nor), ratios, min_per_split=1)
    
    X_splits = []
    y_splits = []
    
    for i in range(len(ratios)):
        X_anor_split = X_anor[idx_X_anor[i]:idx_X_anor[i+1]]
        X_nor_split = X_nor[idx_X_nor[i]:idx_X_nor[i+1]]
        y_anor_split = y_anor[idx_X_anor[i]:idx_X_anor[i+1]]
        y_nor_split = y_nor[idx_X_nor[i]:idx_X_nor[i+1]]
        
        X_split = torch.cat([X_anor_split, X_nor_split], dim=0)
        y_split = torch.cat([y_anor_split, y_nor_split], dim=0)
        
        if len(X_split) < min_samples_per_split:
            print(f"Warning: Split {i} has only {len(X_split)} samples (< {min_samples_per_split})")
        
        X_splits.append(X_split)
        y_splits.append(y_split)
    
    return X_splits, y_splits

def report_detector(model, test_dataset, device):
    for x_batch, y_batch in test_dataset:
        x_batch = x_batch.to(device)
        y_truth = y_batch.to(device)

        y_pred = model.infer_fn(x_batch).squeeze(1)
        TP = np.sum((y_pred == 1) & (y_truth == 1))
        TN = np.sum((y_pred == 0) & (y_truth == 0))
        FP = np.sum((y_pred == 1) & (y_truth == 0))
        FN = np.sum((y_pred == 0) & (y_truth == 1))
    
        # Metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0   # Sensitivity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall (Sensitivity)": recall,
            "Specificity": specificity,
            "F1-score": f1
        }


def report_generator(model, test_dataset):
    pass

def report_rl(model, test_dataset):
    pass
from torch.utils.data import Dataset
import torch

class AnorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, test_dataset=False):
        self.X = X
        self.y = y
        self.X_cur = None
        self.y_cur = None
        if not test_dataset:
            self.balance_fn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        if self.X_cur is None:
            raise ValueError("You must call balance_fn() before using the dataset")
        return self.X_cur.shape[0]

    def __getitem__(self, idx):
        x = self.X_cur[idx]
        y = self.y_cur[idx]
        return x, y

    def balance_fn(self):
        X_anor = self.X[self.y == 1]
        X_nor = self.X[self.y == 0]
        
        if X_anor.shape[0] > X_nor.shape[0]:
            self.X_cur = self.X
            self.y_cur = self.y
            return True
            
        X_nor = X_nor[:X_anor.shape[0]]
        y_anor = torch.ones(X_anor.shape[0], dtype=torch.float)
        y_nor = torch.zeros(X_nor.shape[0], dtype=torch.float)
        
        self.X_cur = torch.cat([X_anor, X_nor], dim=0)
        self.y_cur = torch.cat([y_anor, y_nor], dim=0)
        
        indices = torch.randperm(len(self.X_cur))
        self.X_cur = self.X_cur[indices]
        self.y_cur = self.y_cur[indices]
        
        return False

    def update_anomaly_fn(self, X_new):
        if X_new is not None and len(X_new) > 0:
            if isinstance(X_new, list):
                X_new = torch.stack(X_new)
            
            self.X = torch.cat([self.X, X_new], dim=0)
            y_new = torch.ones(X_new.shape[0], dtype=torch.float).to(self.device)
            self.y = torch.cat([self.y, y_new], dim=0)
            self.balance_fn()
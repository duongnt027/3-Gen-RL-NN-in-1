from torch.utils.data import Dataset
import torch

class AnorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

        self.X_cur = None
        self.y_cur = None
        
        self.balance_fn()

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
        X_nor = X_nor[:X_anor.shape[0]]
        y_anor = torch.ones(X_anor.shape[0], dtype=torch.float)
        y_nor = torch.zeros(X_nor.shape[0], dtype=torch.float)
        self.X_cur = torch.cat([X_anor, X_nor], dim=0)
        self.y_cur = torch.cat([y_anor, y_nor], dim=0)

        return self.X_cur, self.y_cur

    def update_anormaly_fn(self, X_new=None):
        if X_new != None:
            self.X = torch.cat([self.X, X_new], dim=0)
            self.y = torch.cat([self.y, torch.ones(X_new.shape[0], dtype=torch.float)], dim=0)

            self.balance_fn()
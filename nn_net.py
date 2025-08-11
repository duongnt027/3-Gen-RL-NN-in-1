import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDetector(nn.Module):

    def __init__(self, in_dim:int, dropout:float=0.4, device:str=None):
        super().__init__()
        self.in_dim = in_dim
        self.dropout = nn.Dropout(dropout)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        h_dim = 128

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h_dim, 1)

        self.optimizer = self.optimizer_fn()

        self.to(self.device)

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.dropout(self.fc1(x)))))

    def optimizer_fn(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_fn(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y, reduction="sum")

    def train_fn(self, dataloader):
        self.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)

            y_pred = self.forward(x_batch)
            loss = self.loss_fn(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss/len(dataloader)

    def infer_fn(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
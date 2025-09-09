import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseNet(nn.Module):
    def __init__(self, in_dim, out_dim, clamp=None):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        self.clamp = clamp

    def forward(self, obs):
        obs = obs.to("cuda")
        activation1 = torch.relu(self.layer1(obs))
        activation2 = torch.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        if self.clamp:
            output = torch.clamp(output, min=-1.0*self.clamp, max=1.0*self.clamp)

        return output
    
class SimpleDetector(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2, device: str = None):
        super().__init__()
        self.in_dim = in_dim
        
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
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return torch.sigmoid(self.fc2(h))

    def optimizer_fn(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_fn(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y, reduction="mean")

    def train_fn(self, dataloader, num_epochs=100):
        best_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            self.train()
            total_loss = 0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1).float()
    
                y_pred = self.forward(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
            epoch_loss = total_loss / len(dataloader)
            if epoch % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}")
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                best_state = self.state_dict()  # save best weights in memory

        self.load_state_dict(best_state)
        return best_loss

    def infer_fn(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        

class PositionalEncoding(nn.Module):
    """
    Add positional encoding to the input for sequential modeling.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        """Add positional encoding to input tensor."""
        L = x.size(1)  # Sequence length
        return x + self.pe[:, :L, :].to(x.device)


class TransformerDetector(nn.Module):
    """
    Transformer-based detector model following SimpleDetector structure.
    """

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, device=None):
        super().__init__()
        # Device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Embedding + positional encoding
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Optimizer
        self.optimizer = self.optimizer_fn()
        self.to(self.device)

    def forward(self, x):
        # Input có thể là:
        # (B, input_size) -> thêm chiều sequence
        # (B, L, input_size) -> giữ nguyên
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_size)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)

        x = self.embedding(x)               # (B, L, d_model)
        x = self.positional_encoding(x)     # (B, L, d_model)
        x = self.transformer_encoder(x)     # (B, L, d_model)
        x = x.mean(dim=1)                   # (B, d_model)
        return self.fc(x)                   # (B, 1)
    def optimizer_fn(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_fn(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y, reduction="mean")

    def train_fn(self, dataloader, num_epochs=100):
        best_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            self.train()
            total_loss = 0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1).float()

                y_pred = self.forward(x_batch)
                loss = self.loss_fn(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            epoch_loss = total_loss / len(dataloader)

            if epoch % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}")
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                best_state = self.state_dict()

        self.load_state_dict(best_state)
        return best_loss

    def infer_fn(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x.to(self.device))

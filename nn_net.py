import torch
import torch.nn as nn
import torch.nn.functional as F

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
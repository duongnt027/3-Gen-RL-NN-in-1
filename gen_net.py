import torch
import torch.nn as nn
import torch.nn.functional as F

class CVae(nn.Module):

    def __init__(self, in_dim, h_dim:int=128, z_dim:int=64, beta:float=4.0, device:str=None):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.beta = beta
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Encoder
        # self.en_norm = nn.InstanceNorm1d(in_dim+1)
        self.fc1 = nn.Linear(in_dim+1, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, z_dim)
        self.fc3_logvar = nn.Linear(h_dim, z_dim)
    
        # Decoder
        self.fc4 = nn.Linear(z_dim+1, h_dim)
        self.fc5 = nn.Linear(h_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, in_dim)
        # self.de_norm = nn.InstanceNorm1d(in_dim)
        
        self.optimizer = self.optimizer_fn()

        self.to(self.device)

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        # xy = self.en_norm(xy)
        h = F.relu(self.fc1(xy))
        h = F.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = F.relu(self.fc4(zy))
        h = F.relu(self.fc5(h))
        # return self.de_norm(self.fc6(h))
        return self.fc6(h)

    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, y), mean, logvar

    def loss_fn(self, x, x_recon, mean, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def optimizer_fn(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_fn(self, data_loader, num_epochs=100):
        best_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            self.train()
            total_loss = 0
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
    
                x_recon, mean, logvar = self.forward(x_batch, y_batch)
                loss = self.loss_fn(x_batch, x_recon, mean, logvar)
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
            epoch_loss = total_loss/len(data_loader)
            if epoch % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}")
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                best_state = self.state_dict()  # save best weights in memory

        self.load_state_dict(best_state)
        return best_loss

    def infer_fn(self, z, y):
        self.eval()
        with torch.no_grad():
            return self.decode(z, y)
            
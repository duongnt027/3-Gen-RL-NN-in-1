import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import split_dataset, report_detector
from ad_dataset import AnorDataset
from rl_net import AnorEnv, GenPPO
from gen_net import CVae
from nn_net import SimpleDetector, BaseNet

def main_training_loop(X_train, y_train, X_test, y_test, episodes=10, k_new=100, device="cpu"):
    """Main training loop implementing the algorithm"""
    
    # Initialize models
    in_dim = X_train.shape[1]
    model_cvae = CVae(in_dim=in_dim, device=device)
    model_detector = SimpleDetector(in_dim=in_dim, device=device)
    model_ppo = None
    
    # Create datasets
    train_ds = AnorDataset(X_train, y_train)
    test_ds = AnorDataset(X_test, y_test)
    
    print("Starting training loop...")
    
    for ep in range(episodes):
        print(f"\n=== Episode {ep + 1}/{episodes} ===")
        
        # Balance dataset
        done = train_ds.balance_fn()
        if done:
            print("Dataset already balanced, breaking...")
            break
            
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        
        # if ep == 0:
        # First episode: train VAE
        print("Training VAE...")
        vae_loss = model_cvae.train_fn(train_loader)
        print(f"VAE Loss: {vae_loss:.4f}")
        
        # Evaluate VAE
        # report_generator(model_cvae, test_ds, device)
        
        # Initialize RL environment and agent
        rl_env = AnorEnv(train_ds, model_cvae, model_detector)
        
        # Create PPO networks
        actor_net = BaseNet(in_dim, 2, clamp=2.0)  # Output [mu, sigma]
        critic_net = BaseNet(in_dim, 1)
        model_ppo = GenPPO(actor_net, critic_net, rl_env)
        
        print("Generating new anomaly samples with RL...")
        # Generate new anomaly samples using RL
        x_news = []
        obs, _ = rl_env.reset()
        
        for k in range(k_new):
            action, _ = model_ppo.get_action(obs)
            new_obs, reward, terminated, truncated, info = rl_env.step(action)
            
            if info.get('added_to_buffer', False):
                x_news.append(new_obs.to(device))
            
            if terminated or truncated:
                obs, _ = rl_env.reset()
            else:
                obs = rl_env.current_state
        
        # Update dataset with new samples
        if x_news:
            print(f"Generated {len(x_news)} new anomaly samples")
            # x_news = torch.tensor(x_news).to(device)
            # print(x_news)
            train_ds.update_anomaly_fn(torch.stack(x_news))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        
        # Train detector
        print("Training detector...")
        detector_loss = model_detector.train_fn(train_loader)
        print(f"Detector Loss: {detector_loss:.4f}")
        
        # Evaluate detector
        detector_report = report_detector(model_detector, test_ds, device)
        
        # Update RL environment with new detector
        rl_env = AnorEnv(train_ds, model_cvae, model_detector)
        model_ppo.env = rl_env
        
        # Train RL agent
        print("Training RL agent...")
        model_ppo.learn(total_timesteps=60000)
        
        print(f"Dataset size after episode {ep + 1}: {len(train_ds)}")

    print("\nTraining completed!")
    return model_cvae, model_detector, model_ppo

# PREPARING DATASET
# Change it with the output of 0_download_dataset.py
DATASET_NPZ_DIR = "/usr/local/lib/python3.11/dist-packages/adbench/datasets"
DATASET_JSON_DIR = "./adbench_ds/json"

with open(f"./datasets_files_name.json", "r") as file:
    dataset_trees = json.load(file)

dataset_trees.pop("CV_by_ViT")
dataset_trees.pop("NLP_by_RoBERTa")

# ds_type = "CV_by_ResNet18"
# ds_name = "CIFAR10_1"
ds_type = "Classical"
ds_name = "7_Cardiotocography"

ds_npz = dict(np.load(f"{DATASET_NPZ_DIR}/{ds_type}/{ds_name}.npz"))

ds_X = torch.from_numpy(ds_npz['X']).float()
ds_y = torch.from_numpy(ds_npz['y']).float()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Example data creation (replace with your actual data)
N_train, N_test = 1000, 200
in_dim = 10

X, y = split_dataset(ds_X, ds_y, [80, 20])
X_train, X_test = X
y_train, y_test = y
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
# X_train, X_test = ds_X[:280000], ds_X[280001:]
# y_train, y_test = ds_y[:280000], ds_y[280001:]

# Run training
model_cvae, model_detector, model_ppo = main_training_loop(
    X_train, y_train, X_test, y_test,
    episodes=5,
    k_new=50,
    device=device
)


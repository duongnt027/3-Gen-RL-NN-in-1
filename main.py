from adbench.myutils import Utils
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import split_dataset, report_detector
from gen_net import CVae
from nn_net import SimpleDetector
from rl_net import RLEnv, PPO
from ad_dataset import AnorDataset

# I suggest you replace the dataset with adbench_ds/
DATASET_NPZ_DIR = "/usr/local/lib/python3.11/dist-packages/adbench/datasets"
DATASET_JSON_DIR = "/kaggle/working"

utils = Utils()

utils.root = DATASET_JSON_DIR
utils.download_datasets()

with open(f"{DATASET_JSON_DIR}/datasets_files_name.json", "r") as file:
    dataset_trees = json.load(file)

dataset_trees.pop("CV_by_ViT")
dataset_trees.pop("NLP_by_RoBERTa")


ds_type = "Classical"
ds_name = "10_cover"

ds_npz = dict(np.load(f"{DATASET_NPZ_DIR}/{ds_type}/{ds_name}.npz"))

ds_X = torch.from_numpy(ds_npz['X']).float()
ds_y = torch.from_numpy(ds_npz['y']).float()

X, y = split_dataset(ds_X, ds_y, [80, 20])
X_train, X_test = X
y_train, y_test = y

train_ds = AnorDataset(X_train, y_train)
test_ds = AnorDataset(X_test, y_test)

INPUT_DIM = X_train.shape[1]
k_new = 10 
EPISODES = 100

model_cvae = CVae(in_dim=INPUT_DIM)
model_detector = SimpleDetector(in_dim=INPUT_DIM)
model_ppo = PPO()

for ep in range(EPISODES):
    train_ds.balance_fn()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    if ep == 0:
        rl_env = RLEnv(train_ds)
    
        model_cvae.train_fn(train_loader)
        # report_generator(model_cvae, test_ds)
    
        x_news = []
        for k in range(k_new):
            mean, var = model_ppo.get_action(rl_env)
            z_new = model_cvae.reparameterize(mean, var)
            x_news.append(model_cvae.decode(z_new, torch.tensor([1], dtype=torch.float)))
        
        train_ds.update_anormaly_fn(torch.tensor(x_news, dtype=torch.float))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model_detector.train_fn(train_loader)
    report_detector(model_detector, test_ds)

    rl_env = RLEnv(train_ds, model_detector)

    model_ppo.train_fn(rl_env)
    # report_rl(model_ppo, test_ds)
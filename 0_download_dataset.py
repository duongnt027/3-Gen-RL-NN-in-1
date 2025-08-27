from adbench.myutils import Utils
import json
import os

# Change it with the output of this code
DATASET_NPZ_DIR = "/usr/local/lib/python3.11/dist-packages/adbench/datasets"
DATASET_JSON_DIR = "adbench_ds/json"

os.makedirs(DATASET_JSON_DIR, exist_ok=True)

utils = Utils()

utils.root = DATASET_JSON_DIR
utils.download_datasets()
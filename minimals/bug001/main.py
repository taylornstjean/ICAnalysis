# --- Imports ---

import os

from graphnet.data.dataset import Dataset  # GraphNeT dataset loader

from torch_geometric.loader import DataLoader
from graphnet.models import StandardModel
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)

# --- Constants ---

MODEL_CONFIG_PATH = "./config/model_config.yml"
DATASET_CONFIG_PATH = "./config/dataset_config.yml"

EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 16
NUM_WORKERS = 16
MAX_EPOCHS = 100
GPUS = None

OUTPUT_DIR = "./"

# --- Script ---

# Build model
model_config = ModelConfig.load(MODEL_CONFIG_PATH)
model = StandardModel.from_config(model_config, trust=True)

# Configuration
config = TrainingConfig(
    target=[
        target for task in model._tasks for target in task._target_labels
    ],
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    fit={
        "gpus": GPUS,
        "max_epochs": MAX_EPOCHS,
    },
    dataloader={"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS},
)

archive = os.path.join(OUTPUT_DIR, "train_model")
run_name = "dynedge_{}_example".format("_".join(config.target))

dataset_config = DatasetConfig.load(DATASET_CONFIG_PATH)

datasets = Dataset.from_config(dataset_config)

# Construct datasets from multiple selections
train_dataset = datasets["train"]
valid_dataset = datasets["validation"]
test_dataset = datasets["test"]

# Construct dataloaders
train_dataloaders = DataLoader(
    train_dataset, shuffle=True, **config.dataloader
)
valid_dataloaders = DataLoader(
    valid_dataset, shuffle=False, **config.dataloader
)
test_dataloaders = DataLoader(
    test_dataset, shuffle=False, **config.dataloader
)

# Training model
model.fit(
    train_dataloaders,
    valid_dataloaders,
    early_stopping_patience=config.early_stopping_patience,
    logger=None,
    **config.fit,
)

# Save model to file
db_name = dataset_config.path.split("/")[-1].split(".")[0]
path = os.path.join(archive, db_name, run_name)
os.makedirs(path, exist_ok=True)
model.save_state_dict(f"{path}/state_dict.pth")
model.save(f"{path}/model.pth")

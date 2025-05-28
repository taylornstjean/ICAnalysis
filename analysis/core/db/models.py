# --- Imports ---
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import yaml
import json
import sqlite3

from graphnet.data.dataset import Dataset  # GraphNeT dataset loader
from analysis import config  # Project-specific config file
from analysis.render import FeatureDistribution, FeatureDistribution3D

from pytorch_lightning.utilities import rank_zero_only
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)


# --- Class for handling multiple .db files in a directory ---
class DBFileGroup:

    def __init__(self, directory: str):
        self._directory = directory
        # List of all .db files in the directory
        self._files = [
            os.path.join(self._directory, file) for file in glob(os.path.join(self._directory, "*.db"))
        ]
        self._merge_path = os.path.join(self._directory, "merged")  # Where merged DB will go
        if not os.path.exists(self._merge_path):
            os.makedirs(self._merge_path)

        self._config_path = os.path.join(self._directory, "config.yml")  # Optional config path

    def __preprocess_for_merge(self):
        """Make event_no values unique across files by remapping them."""

        for db_file in tqdm(self._files):
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Use part of the filename to create a unique offset for event numbers
            identifier = int(int(os.path.basename(db_file)[-9:-3]) * 1e6)

            # Get all event numbers
            cursor.execute("SELECT event_no FROM truth ORDER BY event_no;")
            event_nos = [row[0] for row in cursor.fetchall()]

            if not event_nos:
                conn.close()
                continue  # Skip empty databases

            # Generate new unique event numbers by offsetting
            new_event_nos = [identifier + en for en in event_nos]
            mapping = dict(zip(event_nos, new_event_nos))

            # Check for potential conflicts
            cursor.execute("SELECT event_no FROM truth;")
            existing_events = {row[0] for row in cursor.fetchall()}
            if any(new_no in existing_events for new_no in new_event_nos):
                # Fallback: assign sequential new numbers
                event_counter = max(existing_events) + 1
                new_event_nos = list(range(event_counter, event_counter + len(event_nos)))
                mapping = dict(zip(event_nos, new_event_nos))

            # Apply remapping
            for old_no, new_no in mapping.items():
                cursor.execute("UPDATE truth SET event_no = ? WHERE event_no = ?;", (new_no, old_no))

            conn.commit()
            conn.close()

    def merge(self, batch_size=10 * 1024) -> str:
        """Merge all .db files in this group into a single SQLite database."""
        merged_file = os.path.join(self._merge_path, "merged.db")

        # Connect to the target merged DB
        target_conn = sqlite3.connect(merged_file)
        target_cursor = target_conn.cursor()

        # Optimize SQLite for large merge operations
        target_cursor.execute("PRAGMA journal_mode = WAL;")
        target_cursor.execute(f"PRAGMA cache_size = {100 * 1024};")
        target_cursor.execute("PRAGMA temp_store = MEMORY;")
        target_conn.execute('BEGIN TRANSACTION;')  # Speed up by deferring commits

        # Iterate over all source databases
        for db_file in tqdm(self._files):
            source_conn = sqlite3.connect(db_file)
            source_cursor = source_conn.cursor()

            # Find all tables in the source DB
            source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = source_cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # Create the table in the merged DB if it doesn't exist
                target_cursor.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
                )
                existing_table = target_cursor.fetchone()

                if not existing_table:
                    # Copy schema from source
                    source_cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = source_cursor.fetchall()
                    create_table_sql = f"CREATE TABLE {table_name} ("
                    create_table_sql += ", ".join([f"{column[1]} {column[2]}" for column in columns]) + ")"
                    target_cursor.execute(create_table_sql)

                # Copy rows in batches
                source_cursor.execute(f"SELECT * FROM {table_name}")
                while True:
                    rows = source_cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    target_cursor.executemany(
                        f"INSERT INTO {table_name} VALUES ({','.join(['?'] * len(rows[0]))})", rows
                    )

            source_conn.close()

        target_conn.commit()
        target_conn.close()

        return merged_file


# --- Class for handling a single .db file ---
class DBFile:

    def __init__(self, path: str):
        self._path = path
        self._config_path = os.path.join(
            os.path.dirname(self._path),
            f"{os.path.basename(self._path).strip('.db')}_config.yml"
        )
        self._dataset_config_path = os.path.join(
            os.path.dirname(self._path),
            f"{os.path.basename(self._path).strip('.db')}_training_config.yml"
        )
        self._model_config_path = config.VP_RECO_MODEL_PATH
        self._training_output_dir = os.path.join(
            config.OUTPUT_DIR,
            f"{os.path.basename(self._path).strip('.db')}/"
        )

    def generate_config(self):
        """Generate a YAML config file for this DB based on a base schema."""
        with open(config.YML_CONFIG_SCHEMA_FILE) as file:
            data = json.load(file)

        # Customize the base config with the current file path
        data["path"] = self._path

        # Save config
        with open(self._config_path, "w") as file:
            yaml.dump(data, file)

    def generate_training_config(self):
        """Generate a YAML training config file for this DB based on a base schema."""
        with open(config.YML_TRAINING_CONFIG_SCHEMA_FILE) as file:
            data = json.load(file)

        # Customize the base config with the current file path
        data["path"] = self._path

        # Save config
        with open(self._dataset_config_path, "w") as file:
            yaml.dump(data, file)

    def plot_feature_distribution(self):
        """Plot histograms of each feature's values using GraphNeT Dataset."""

        dataset = Dataset.from_config(self._config_path)
        features = dataset._features  # Feature names

        # Collect all feature tensors into a single array
        x_preprocessed_list = []
        for batch in tqdm(dataset, colour="green"):
            x_preprocessed_list.append(batch.x.numpy())

        x_preprocessed = np.concatenate(x_preprocessed_list, axis=0)

        # figure = FeatureDistribution(x_preprocessed, features, 100)
        # figure.populate()
        # figure.save(os.path.join(config.OUTPUT_DIR, "plots/feature_distribution.html"))

        figure = FeatureDistribution3D(x_preprocessed)
        figure.populate()
        figure.save(os.path.join(config.OUTPUT_DIR, "plots/feature_distribution_3d.html"))

    def train_model(
            self,
            gpus,
            max_epochs: int,
            early_stopping_patience: int,
            batch_size: int,
            num_workers: int,
            suffix
    ):
        # Build model
        model_config = ModelConfig.load(self._model_config_path)
        model: StandardModel = StandardModel.from_config(model_config, trust=True)

        # Configuration
        config = TrainingConfig(
            target=[
                target for task in model._tasks for target in task._target_labels
            ],
            early_stopping_patience=early_stopping_patience,
            fit={
                "gpus": gpus,
                "max_epochs": max_epochs,
            },
            dataloader={"batch_size": batch_size, "num_workers": num_workers},
        )

        if suffix is not None:
            archive = os.path.join(self._training_output_dir, f"train_model_{suffix}")
        else:
            archive = os.path.join(self._training_output_dir, "train_model")
        run_name = "dynedge_{}_example".format("_".join(config.target))

        # Construct dataloaders
        dataset_config = DatasetConfig.load(self._dataset_config_path)
        datasets = Dataset.from_config(
            dataset_config,
        )

        # Construct datasets from multiple selections
        train_dataset = EnsembleDataset(
            [datasets[key] for key in datasets if key.startswith("train")]
        )
        valid_dataset = EnsembleDataset(
            [datasets[key] for key in datasets if key.startswith("valid")]
        )
        test_dataset = EnsembleDataset(
            [datasets[key] for key in datasets if key.startswith("test")]
        )

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

        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Validation dataset length: {len(valid_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")

        # Peek into the first batch
        batch = next(iter(train_dataloaders))

        print("Keys in training batch:")
        print(batch.keys)

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

        # Get predictions
        if isinstance(config.target, str):
            additional_attributes = [config.target]
        else:
            additional_attributes = config.target

        results = model.predict_as_dataframe(
            test_dataloaders,
            additional_attributes=additional_attributes + ["event_no"],
            gpus=config.fit["gpus"],
        )
        results.to_csv(f"{path}/results.csv")

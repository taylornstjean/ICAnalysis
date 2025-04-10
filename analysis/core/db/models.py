# --- Imports ---
import os.path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob
import yaml
import json
import sqlite3

from graphnet.data.dataset import Dataset  # GraphNeT dataset loader
from analysis import config  # Project-specific config file


# --- Class for handling multiple .db files in a directory ---
class DBFileGroup:

    def __init__(self, directory: str):
        self._directory = directory
        # List of all .db files in the directory
        self._files = [
            os.path.join(self._directory, file) for file in glob(os.path.join(self._directory, "*.db"))
        ]
        self._merge_path = os.path.join(self._directory, "merged")  # Where merged DB will go
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

    def generate_config(self):
        """Generate a YAML config file for this DB based on a base schema."""
        with open(config.YML_CONFIG_SCHEMA_FILE) as file:
            data = json.load(file)

        # Customize the base config with the current file path
        data["path"] = self._path

        # Save config
        with open(self._config_path, "w") as file:
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

        # Layout setup for the grid of histograms
        nb_features_preprocessed = x_preprocessed.shape[1]
        dim = int(np.ceil(np.sqrt(nb_features_preprocessed)))
        axis_size = 4
        bins = 50

        # Create the figure
        fig, axes = plt.subplots(dim, dim, figsize=(dim * axis_size, dim * axis_size))

        for ix, ax in enumerate(axes.ravel()[:nb_features_preprocessed]):
            ax.hist(x_preprocessed[:, ix], bins=bins, color="orange")
            ax.set_xlabel(f"x{ix}: {features[ix] if ix <= len(features) else 'N/A'}")
            ax.set_yscale("log")  # Log scale helps visualize skewed distributions

        fig.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "plots/feature_distribution_preprocessed.png")
        fig.savefig(save_path)

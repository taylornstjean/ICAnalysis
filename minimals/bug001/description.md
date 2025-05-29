# Failure to Load Truth Info to Training Dataset

This bug appears when trying to load data from a database file (merged.db) to use in GNN training with GraphNeT. I am training to reconstruct vertices from pulse data, for this I need to tell GraphNeT to include the relevant truth information when loading the data from data.db into a `Dataset` object. In this case, I am pulling position_x, position_y, and position_z from its truth table. I have verified this data exists and is valid in merged.db.

To do this, I define the target labels within the model configuration file (model_config.yml):

```
target_labels: [position_x, position_y, position_z]
```

I then create a `StandardModel` object using this configuration file:

```
model_config = ModelConfig.load(MODEL_CONFIG_PATH)
model = StandardModel.from_config(model_config, trust=True)
```

At this point, `model_config` accurately includes the target labels as a task, and `model` includes them directly as target labels.

Next, I create a `Dataset` object using the dataset cofiguration file (dataset_config.yml):

```
dataset_config = DatasetConfig.load(DATASET_CONFIG_PATH)
datasets = Dataset.from_config(dataset_config)
```

Here, `datasets` includes all the specified truth parameters in the dataset config file.

When I go to load the datasets however, the truth information is not included. I am not sure where the truth info is lost, but it seems GraphNeT never loads it in the first place. This leads to a KeyError when calling `model.fit()`:

```
Traceback (most recent call last):
  File "/data/i3home/tstjean/icecube/main.py", line 46, in <module>
    main()
  File "/data/i3home/tstjean/icecube/main.py", line 42, in main
    mergefile.train_model(None, 100, 10, 16, 16, "run0")
  ...
  ...
  ...
  File "/data/i3home/tstjean/icecube/venv/lib/python3.11/site-packages/torch_geometric/data/data.py", line 577, in __getitem__
    return self._store[key]
           ~~~~~~~~~~~^^^^^
  File "/data/i3home/tstjean/icecube/venv/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 118, in __getitem__
    return self._mapping[key]
           ~~~~~~~~~~~~~^^^^^
KeyError: 'position_z'
```

The error is thrown while computing loss (/data/i3home/tstjean/icecube/graphnet/src/graphnet/models/standard_model.py, line 78):

```
def compute_loss(
    self, preds: Tensor, data: List[Data], verbose: bool = False
) -> Tensor:
    """Compute and sum losses across tasks."""
    data_merged = {}
    target_labels_merged = list(set(self.target_labels))
    for label in target_labels_merged:
        data_merged[label] = torch.cat([d[label] for d in data], dim=0)
    for task in self._tasks:
        if task._loss_weight is not None:
            data_merged[task._loss_weight] = torch.cat(
                [d[task._loss_weight] for d in data], dim=0
            )

    losses = [
        task.compute_loss(pred, data_merged)
        for task, pred in zip(self._tasks, preds)
    ]
    if verbose:
        self.info(f"{losses}")
    assert all(
        loss.dim() == 0 for loss in losses
    ), "Please reduce loss for each task separately"
    return torch.sum(torch.stack(losses))
```

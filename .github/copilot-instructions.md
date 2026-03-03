# Copilot instructions

## Project orientation
- This repo trains physics-informed GNNs for flood modeling; the entry points are `train.py`, `test.py`, and `hp_search.py` (see root README).
- Data flows through dataset factories in `data/__init__.py` and model/trainer factories in `models/__init__.py` and `training/__init__.py`.
- Training is split by task type: node-only, edge-only, or dual node+edge; autoregressive trainers use sliding windows and curriculum learning (see `training/README.md`).
- Physics-informed losses live in `loss/` and are wired via `training/physics_informed_trainer.py`; they depend on `global_mass_info`/`local_mass_info` attached to batches (see `loss/README.md`).

## Core architecture patterns
- Datasets: `FloodEventDataset`/`AutoregressiveFloodDataset` (and in-memory variants) are created via `dataset_factory(storage_mode, autoregressive)`.
- Models: `model_factory(model_name, ...)` switches between DUALFloodGNN, GAT/GCN variants, and edge/node variants.
- Trainers: `trainer_factory(model_name, autoregressive, isCluster)` selects regression vs autoregressive trainers; node+edge models use dual trainers.
- Testing mirrors training with tester classes in `testing/` (see `testing/README.md`).

## Workflows and commands
- Local setup is documented in the root README: PyTorch 2.5.1 + PyG wheels and `requirements.txt`.
- SLURM entry scripts exist for cluster runs: `run_train.sh`, `run_test.sh`, `run_hp_search.sh` (they call `train.py`, `test.py`, `hp_search.py` with config/model args).
- Configs are YAML files under `configs/`; hyperparameter configs live under `configs/hparam_config/`.

## Project-specific conventions
- Physics loss is applied only to node prediction trainers (see note in `training/README.md`).
- Autoregressive datasets contain multi-timestep node/edge features and labels; trainers extract sliding windows for target features.
- Loss scaling is handled by `utils/loss_scaler.py`; training stats are tracked in `utils/training_stats.py`.

## Data and external dependencies
- Raw data is expected in `data/datasets/raw` with HEC-RAS `.hdf`, shapefiles, DEM `.tif`, and event summary CSVs (see `data/README.md`).
- Geospatial tooling is used (`geopandas`, `rasterio`, `shapely`), alongside PyTorch and PyG.

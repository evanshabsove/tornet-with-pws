# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

TorNet is a benchmark dataset and ML software suite for tornado detection using polarimetric weather radar (WSR-88D). It provides Keras 3 CNN baseline models that support TensorFlow, PyTorch, and JAX backends. The dataset contains NetCDF radar samples labeled as tornadic (TOR), warning (WRN), or null (NUL).

## Environment Setup

```bash
pip install .
pip install -r requirements/tensorflow.txt  # or torch.txt, jax.txt, universal.txt
export TORNET_ROOT=/path/to/tornet_data     # Required at runtime
export KERAS_BACKEND=tensorflow             # or torch, jax
```

## Common Commands

```bash
# Run tests
pytest

# Train baseline model
python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params.json

# Evaluate trained model
python scripts/tornado_detection/test_tornado_keras.py --model_path /path/to/model.keras

# Download TorNet dataset from Zenodo
python download_tornet_data.py

# Download MADIS weather station data
python download_madis_data.py
```

## Architecture Overview

### Data Pipeline

`tornet/data/loader.py` is the core — `read_file()` reads a single NetCDF sample and returns radar arrays with shape `[time, azimuth, range, tilt]` plus metadata. `query_catalog()` filters `catalog.csv` by split and year. `TornadoDataLoader` wraps these into an iterable.

Backend-specific loaders in `data/keras/`, `data/torch/`, `data/tf/` extend the base loader. The Keras loader (`KerasDataLoader`) inherits from `keras.utils.PyDataset` for multiprocessing support and handles dimension reordering between backends.

MADIS atmospheric data (pressure, wind, humidity, temperature, dewpoint) is loaded from CSVs in `tornet_data/madis_data/` and matched to radar samples by storm_id within a 10-minute temporal window.

### Model Architecture

`tornet/models/keras/cnn_baseline.py` — VGG-style CNN with:
- Input: 6 radar variables (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH) + range-folded mask + optional coordinate tensors
- `CoordConv2D` layers (in `layers.py`) inject (r, r⁻¹, θ) coordinate channels before each convolution
- Optional MADIS branch: 7 weather features passed through an MLP and fused with CNN features
- Head: configurable maxpool or MLP, outputs single tornado probability

## MADIS Integration (Primary Research Focus)

### What MADIS Adds

MADIS provides surface weather station observations (APRSWXNET personal weather stations) that are not present in radar data. The 7 features are:

| Feature | Normalization Range |
|---|---|
| `pressure` (Pa) | 90,000–110,000 |
| `wind_direction` (°) | 0–360 |
| `wind_speed` (m/s) | 0–50 |
| `wind_gust` (m/s) | 0–50 |
| `relative_humidity` (%) | 0–100 |
| `temperature` (K) | 233.15–323.15 |
| `dewpoint` (K) | 233.15–323.15 |

### Data File

The expected file is `$TORNET_ROOT/madis_features_clean.csv` with columns: `storm_id`, `timestamp`, `pressure`, `wind_direction`, `wind_speed`, `wind_gust`, `relative_humidity`, `temperature`, `dewpoint`. The CSV is loaded once and cached as `_MADIS_DATA_CACHE` in `loader.py`.

### Sample Matching Logic (`loader.py:read_file`)

When `use_madis_data=True`, each radar sample is matched to MADIS observations by:
1. Extracting `storm_id` from `ds.attrs['storm_event_url']` (parsed as a URL query param via `get_id_from_storm_event_url`)
2. Looking up all rows for that `storm_id` in the cached DataFrame
3. Aggregating multiple simultaneous station readings by averaging (`groupby(...).mean()`)
4. Finding the temporally closest observation to the radar frame's timestamp
5. **Rejecting the sample entirely (returns `None`) if the closest observation is >10 minutes away, or if any of the 7 feature values is NaN**

This filtering means MADIS training sets are a strict subset of the full TorNet dataset — not all storms have coverage.

### CNN Fusion Architecture (`cnn_baseline.py:build_model`)

The MADIS branch runs in parallel with the CNN:

```
Radar inputs → [CoordConv VGG blocks × 4] → feature map (7×15×filters)
MADIS (7,) → normalize_madis → Dense(64, relu) → Dropout(0.3) → Dense(32, relu)
```

Fusion only happens with `head='mlp'`: the CNN feature map is flattened, then concatenated with the MADIS branch output before the Dense head layers. **With `head='maxpool'` (the default), MADIS features are ignored even if `use_madis=True`** — you must use `head='mlp'` for MADIS to actually influence predictions.

### Enabling MADIS in Training

Use `params_madis.json` which sets `"use_madis_data": true` and `"head": "mlp"`. The train/val years are set to 2013–2014/2015 (smaller window) due to MADIS coverage being sparser for earlier years:

```bash
python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params_madis.json
```

The `use_madis_data` flag propagates through: `params_madis.json` → `train_tornado_keras.py` → `KerasDataLoader(use_madis_data=True)` → `read_file(use_madis_data=True)`.

### Key Gotchas When Modifying MADIS Integration

- **Feature order matters**: `loader.py` extracts MADIS values in a hardcoded order (pressure, wind_direction, wind_speed, wind_gust, relative_humidity, temperature, dewpoint). `MADIS_MIN_MAX` in `constants.py` must stay in the same order.
- **`MAIDS_VARIABLES` in `constants.py` is a typo** (missing D) — it's `MAIDS_VARIABLES`, not `MADIS_VARIABLES`. It's not used in the model; `MADIS_MIN_MAX` is what matters.
- **Rejected samples propagate up**: `read_file` returns `None` for any sample without a MADIS match. `KerasDataLoader.__getitem__` filters these out, so effective batch sizes may be smaller than configured.
- **MADIS branch does nothing with `head='maxpool'`**: the `madis_branch` tensor is constructed but never concatenated into the output path in the maxpool head — it becomes a dangling node in the graph.

### Data Format

NetCDF files use naming convention: `{CATEGORY}_{YYMMDD}_{HHMMSS}_{RADAR}_{STORM_ID}_{TILT}.nc`

`constants.py` defines normalization ranges for all 6 radar variables (`CHANNEL_MIN_MAX`) and 7 MADIS features (`MADIS_MIN_MAX`). `preprocess.py` computes radar coordinate tensors used by CoordConv.

### Training Configuration

`scripts/tornado_detection/config/params.json` controls all hyperparameters. Key fields: `train_years`, `val_years`, `n_epochs`, `batch_size`, `start_lr`, `w0/w1/w2/wN/wW` (per-class loss weights). `params_madis.json` adds MADIS feature inputs.

### Custom Metrics and Losses

Metrics in `tornet/metrics/keras/metrics.py` use `FromLogitsMixin` so they apply sigmoid internally — models output raw logits, not probabilities. Losses in `tornet/models/keras/losses.py`: MAE, Jaccard (IoU), Dice.

### Pretrained Models

Available via HuggingFace Hub: `tornet-ml/tornado_detector_baseline_v1`. See `models/README.md` for loading instructions.

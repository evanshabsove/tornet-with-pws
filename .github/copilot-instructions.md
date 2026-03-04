# TorNet Development Guide

## Project Overview
TorNet is a research dataset and ML toolkit for tornado detection using polarimetric weather radar data. The codebase supports **Keras 3.0 with multiple backends** (TensorFlow, PyTorch, JAX) and provides backend-agnostic data loaders for training CNN models on weather radar samples.

## Critical Environment Setup

### Required Environment Variables
- `TORNET_ROOT`: Path to the TorNet dataset directory containing `catalog.csv` and `train/`/`test/` subdirectories
- `KERAS_BACKEND`: Set to `tensorflow`, `torch`, or `jax` before importing Keras (controls which deep learning framework Keras uses)

```bash
export TORNET_ROOT=/path/to/tornet_data
export KERAS_BACKEND=tensorflow  # or torch, jax
```

### Installation
Install base package: `pip install .` from repo root
For ML work, install backend-specific requirements: `pip install -r requirements/{tensorflow,torch,jax}.txt`

## Architecture Patterns

### Multi-Backend Support
The codebase is **backend-agnostic** by design. When writing preprocessing or model code:
- Use `backend` parameter in preprocessing functions (see [tornet/data/preprocess.py](tornet/data/preprocess.py))
- Pass the actual module (e.g., `backend=tf`, `backend=np`) for operations like `linspace`, `meshgrid`, `where`
- Keras 3.0 models work automatically across backends via `keras.ops`

### Data Loading Philosophy
Three dataloader implementations with identical interfaces:
- `"keras"`: Backend-agnostic using `keras.utils.PyDataset` ([tornet/data/keras/loader.py](tornet/data/keras/loader.py))
- `"tensorflow"`: Native `tf.data.Dataset` for TensorFlow-specific optimizations
- `"torch"`: Native `torch.utils.data.DataLoader` for PyTorch

Access via `get_dataloader(dataloader_type, data_root, years, data_type, batch_size, **kwargs)` in [tornet/data/loader.py](tornet/data/loader.py)

### Model Input Structure
Models expect **dictionary inputs**, not single tensors:
- Keys: Radar variables (`'DBZ'`, `'VEL'`, `'KDP'`, `'RHOHV'`, `'ZDR'`, `'WIDTH'`) + `'range_folded_mask'` + `'coordinates'`
- See [tornet/models/keras/cnn_baseline.py](tornet/models/keras/cnn_baseline.py#L28-L33) for input layer creation pattern
- Use `select_keys` parameter in dataloaders to specify which inputs to load

### CoordConv Architecture
Custom `CoordConv2D` layers inject spatial coordinate information:
- Takes tuple `(x, coords)` where `coords` contains range/azimuth/inverse-range
- See [tornet/models/keras/layers.py](tornet/models/keras/layers.py) for implementation
- Coordinates computed in [tornet/data/preprocess.py](tornet/data/preprocess.py) via `add_coordinates()`

## Data Conventions

### Radar Variables & Normalization
Constants defined in [tornet/data/constants.py](tornet/data/constants.py):
- 6 radar variables: `['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH']`
- `CHANNEL_MIN_MAX` dict provides normalization ranges for each variable
- Apply normalization in model's input processing, not in dataloader

### Dimension Ordering
- Default: `[batch, azimuth, range, tilt]` (tilt_last=True)
- Alternative: `[batch, tilt, azimuth, range]` (tilt_last=False)
- Control via `tilt_last` parameter in loaders/preprocessing

### Sample Categories
- `TOR=0`: Tornadic event
- `NUL=1`: Null/random sample
- `WRN=2`: Warning issued but no tornado

## Training Workflow

### Standard Training Command
```bash
python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params.json
```

Configuration in [scripts/tornado_detection/config/params.json](scripts/tornado_detection/config/params.json) includes:
- `train_years`/`val_years`: Year splits (dataset spans 2013-2022)
- `dataloader`: Choice of `"keras"`, `"tensorflow"`, or `"torch"`
- `input_variables`: Subset of radar variables to use
- Custom loss weights: `wN`, `w0`, `w1`, `w2`, `wW` for handling class imbalance

### Testing
Run tests with: `pytest` (requires `TORNET_ROOT` environment variable)
Tests verify: data loading, dataloaders, and training loops for each backend

## MADIS Weather Station Integration

### Overview
MADIS (Meteorological Assimilation Data Ingest System) provides surface weather station observations that augment radar data. The integration adds 7 atmospheric features as additional model inputs.

### Data Structure
**Variables** (defined in [tornet/data/constants.py](tornet/data/constants.py)):
- `madis_atmospheric_pressure` - Pa (90000-110000)
- `madis_wind_direction` - degrees (0-360)
- `madis_wind_speed` - m/s (0-50)
- `madis_wind_gust_speed` - m/s (0-50)
- `madis_relative_humidity` - % (0-100)
- `madis_temperature` - Kelvin (233.15-323.15)
- `madis_temperature_dew_point` - Kelvin (233.15-323.15)

**XML File Format**: Located in `tornet_data/madis_data/`, named as:
```
madis_data_{storm_id}_{timestamp}.xml
```
Structure: XML records with `var`, `data_value`, `lat`, `lon`, `ObTime` attributes

### Implementation Details

**Current Behavior** ([tornet/data/loader.py](tornet/data/loader.py#L66-L89)):
1. `use_madis_data=True` flag enables loading
2. Storm ID extracted from `storm_event_url` in netCDF attributes
3. MADIS file path constructed using **hardcoded absolute path** (needs refactoring to use `TORNET_ROOT`)
4. `extract_madis_features()` parses XML, maps variables via `var_map` dict (e.g., `'V-ALTSE'` → `'madis_atmospheric_pressure'`)
5. Returns **7-element float32 array** in `data['madis']` key

**Strict Validation**:
- If MADIS file missing → returns `None` (sample skipped)
- If any variable missing → returns `None`
- If any value is `0.0` → returns `None`
- This aggressive filtering reduces dataset size significantly

### Known Issues & Improvement Areas

1. **Hardcoded Path**: Line 69 uses absolute path instead of `TORNET_ROOT`:
   ```python
   madis_file = f"/Users/evanshabsove/Documents/.../madis_data_{storm_id}_{ds['time'].values[0]}.xml"
   ```
   Should be: `os.path.join(data_root, 'madis_data', f'madis_data_{storm_id}_{timestamp}.xml')`

2. **Zero Value Handling**: Current code treats `0.0` as missing, which may incorrectly filter valid measurements (e.g., wind speed can legitimately be 0)

3. **Missing Data Strategy**: All-or-nothing approach (return `None`) wastes samples. Consider:
   - Imputation strategies (mean/median/nearest neighbor)
   - Masking invalid values instead of dropping samples
   - Adding a validity mask as separate input

4. **Spatial Integration**: Current implementation doesn't account for:
   - Multiple weather stations per radar sample
   - Distance weighting from storm center
   - Temporal interpolation between observation times

5. **Model Integration**: When using MADIS data:
   - Must add `'madis'` to `select_keys` in dataloader
   - Model needs separate input branch for 1D features (currently not in baseline)
   - Normalization should use `MADIS_MIN_MAX` constants

### Enabling MADIS in Workflows

**In DataLoaders**:
```python
ds = KerasDataLoader(
    data_root=DATA_ROOT,
    years=[2018, 2019],
    batch_size=128,
    use_madis_data=True,  # Enable MADIS
    select_keys=ALL_VARIABLES + ['range_folded_mask', 'coordinates', 'madis']
)
```

**Model Architecture** (needs implementation):
- Current baseline doesn't consume `'madis'` input
- Need fusion architecture: CNN for radar + MLP for MADIS features
- Concatenate before final classification layer

### Debugging MADIS Loading
- Check `extract_madis_features()` return value for parsing issues
- Verify XML files exist: `ls tornet_data/madis_data/madis_data_{storm_id}_*.xml`
- Storm ID extraction: `get_id_from_storm_event_url()` parses URL query params
- Missing files will cause silent sample drops (check batch sizes)

## Key Gotchas

1. **Always set `KERAS_BACKEND`** before importing `keras` or `tornet` modules
2. **Pretrained models** available from HuggingFace: `tornet-ml/tornado_detector_baseline_v1` (see [models/README.md](models/README.md))
3. **License header**: MIT Lincoln Laboratory - all source files include distribution statement
4. **Coordinate scaling**: Range values scaled by `1e-5` for numerical stability (hardcoded in preprocessing)
5. **MADIS path hardcoding**: Currently uses absolute path instead of `TORNET_ROOT` - breaks portability

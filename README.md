# TorNet + MADIS Fusion

This started as a fork of [TorNet](https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0006.1.xml), MIT Lincoln Laboratory's benchmark dataset and Keras 3 CNN baseline for tornado detection from polarimetric WSR-88D radar. It has since grown into a research project on **fusing MADIS surface weather station data with radar** to see whether ground-truth atmospheric conditions (pressure, wind, humidity, temperature) improve tornado detection beyond what the radar alone sees, plus a Flask API that serves the trained models against both historical storms and a live NEXRAD feed.

The repo now contains three things:

1. **TorNet** — the original NetCDF radar dataset and VGG-style CoordConv CNN baseline (`tensorflow`/`torch`/`jax` via Keras 3).
2. **MADIS fusion research** — a pipeline that downloads MADIS personal weather station (PWS) observations for every storm in the TorNet catalog, joins them to radar samples, and fuses them into the CNN via either late (post-flatten concat) or FiLM (intermediate feature-map conditioning) fusion.
3. **Serving API** — a Flask app (`api/`) that runs historical storms through three model variants (no-MADIS, MADIS late-fusion, MADIS FiLM) side by side, and separately polls live NEXRAD data to produce real-time tornado-probability heatmaps.

![Alt text](tornet_image.png?raw=true "sample")

## Downloading the TorNet Dataset

TorNet is split across 10 files (one per year, 2013–2022) plus a catalog CSV, hosted on Zenodo:

* Tornet 2013 (3 GB) and catalog: https://doi.org/10.5281/zenodo.12636522
* Tornet 2014 (15 GB): https://doi.org/10.5281/zenodo.12637032
* Tornet 2015 (17 GB): https://doi.org/10.5281/zenodo.12655151
* Tornet 2016 (16 GB): https://doi.org/10.5281/zenodo.12655179
* Tornet 2017 (15 GB): https://doi.org/10.5281/zenodo.12655183
* Tornet 2018 (12 GB): https://doi.org/10.5281/zenodo.12655187
* Tornet 2019 (18 GB): https://doi.org/10.5281/zenodo.12655716
* Tornet 2020 (17 GB): https://doi.org/10.5281/zenodo.12655717
* Tornet 2021 (18 GB): https://doi.org/10.5281/zenodo.12655718
* Tornet 2022 (19 GB): https://doi.org/10.5281/zenodo.12655719

If downloading through your browser is slow, use `zenodo_get` (https://gitlab.com/dvolgyes/zenodo_get), or run `python download_tornet_data.py`.

After downloading, untar everything into one directory — this is `TORNET_ROOT` for the rest of this README. It should contain `catalog.csv` and `train/`/`test/` subdirectories full of `.nc` files.

## Setup

```bash
pip install .
pip install -r requirements/tensorflow.txt   # or torch.txt, jax.txt
export TORNET_ROOT=/path/to/tornet_data
export KERAS_BACKEND=tensorflow              # or torch, jax
```

`requirements/basic.txt` covers the base package; pick the backend-specific requirements file for the deep learning framework you want. `requirements/api.txt` is only needed for the Flask API (below).

### Conda

```bash
conda create -n tornet-{backend} python=3.10
conda activate tornet-{backend}
pip install -r requirements/{backend}.txt
```

Replace `{backend}` with `tensorflow`, `torch`, or `jax`.

## Loading and Visualizing TorNet

Start with `notebooks/DataLoaders.ipynb` for an overview of loading and visualizing radar samples. `notebooks/VisualizeSamples.ipynb` shows inference with a pretrained model.

## Generating the MADIS Dataset

MADIS surface station data isn't bundled with TorNet — it's pulled separately and joined to the radar catalog by `storm_id` and time. This only needs to be done once per `TORNET_ROOT`; the resulting CSVs are cached and reused on every training run.

1. **Generate download URLs** for each storm in the catalog, across three temporal windows (T0 — during the radar scan, T-2h — pre-storm setup, T-24h — a baseline control used for anomaly features):

   ```bash
   python generate_madis_urls.py --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022
   ```

2. **Download the raw XML station data.** This can be run on a different machine (e.g. one with VPN access to APRSWXNET) from the URL list generated above:

   ```bash
   python download_madis_from_urls.py madis_download_urls.csv --workers 10 --output-dir $TORNET_ROOT/madis_data
   ```

   (`download_madis_data.py` does both steps together if you don't need to split URL generation from downloading across machines.)

3. **Parse and aggregate into features.** `scripts/build_madis_features.py` parses the T0/T-2h/T-24h XML files, averages multiple simultaneous station readings per storm, computes 24-hour anomaly features (pressure/wind/instability deltas — the strongest predictors, see `MADIS_DATA_FINDINGS.md`), and writes `$TORNET_ROOT/madis_features_clean.csv`:

   ```bash
   python scripts/build_madis_features.py
   ```

4. **Build the MADIS-eligible catalog.** Not every storm has a nearby station — `scripts/build_madis_eligible_catalog.py` filters `catalog.csv` down to storms with a valid (non-NaN pressure + wind_gust) MADIS match, so MADIS and no-MADIS runs can be trained/evaluated on the identical storm population:

   ```bash
   python scripts/build_madis_eligible_catalog.py
   ```

   Roughly half the catalog has usable MADIS coverage (see `MADIS_DATA_FINDINGS.md` for the full coverage/quality analysis — station sparsity, temporal offsets, per-feature effect sizes, etc.).

5. **(API only)** Build the flat storm-id cache the Flask API reads at startup for `GET /storms/madis`:

   ```bash
   python scripts/build_madis_storm_ids_cache.py
   ```

`notebooks/madis_data_exploration.ipynb` and `notebooks/madis_weight_analysis.ipynb` cover the exploratory analysis behind this pipeline.

## Train a Model

### Backend selection (Keras 3)

```bash
export KERAS_BACKEND=tensorflow
# export KERAS_BACKEND=torch
# export KERAS_BACKEND=jax
```

### Radar-only baseline

```bash
export TORNET_ROOT=/path/to/tornet_data
python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params.json
```

If run out-of-the-box this will be slow, since it uses the basic dataloader — see `notebooks/DataLoaders.ipynb` for tips on speeding it up.

### MADIS fusion

`scripts/tornado_detection/config/` has one params file per fusion variant:

| Config | `use_madis_data` | `madis_fusion` | `head` | Notes |
|---|---|---|---|---|
| `params.json` | false | — | maxpool | radar-only baseline |
| `params_madis.json` | true | `late` | mlp | concat MADIS after flattening CNN features — **`late` fusion has no effect unless `head` is `mlp`**, since `maxpool` never flattens the feature map |
| `params_madis_film.json` | true | `film` | maxpool | FiLM-conditions the CNN feature map after the 3rd conv block, before the final block — works with either head |
| `params_madis_top3.json` | true | — | — | trains on only the 3 strongest raw MADIS features (pressure, wind_gust, wind_speed) instead of all 7 |

MADIS configs default to `train_years: [2013, 2014]` / `val_years: [2015]` — a smaller window than the radar-only baseline, since MADIS coverage is sparser in earlier years and downloads were prioritized there.

```bash
export TORNET_ROOT=/path/to/tornet_data
python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params_madis_film.json
```

`_quick_test` variants of each config (e.g. `params_madis_film_quick_test.json`) run a handful of steps on a single year for smoke-testing changes.

## Evaluate a Trained Model

```bash
export TORNET_ROOT=/path/to/tornet_data
python scripts/tornado_detection/test_tornado_keras.py --model_path /path/to/model.keras --params_path /path/to/params.json
```

This prints AUC and other metrics computed on the test set. Omitting `--model_path` downloads the pretrained radar-only baseline from HuggingFace (`tornet-ml/tornado_detector_baseline_v1`) instead — see `models/README.md`.

## Serving API

`api/app.py` is a Flask app that serves the three trained model variants (no-MADIS, MADIS late-fusion, MADIS FiLM — all from the same `run2` trial in `results/run_july_6th/results/` so comparisons aren't confounded by which random trial won) over HTTP, and separately runs a background thread that polls live NEXRAD data for real-time tornado-probability heatmaps. It's consumed cross-origin by a separate Rails frontend.

```bash
export TORNET_ROOT=/path/to/tornet_data
export KERAS_BACKEND=tensorflow
pip install -r requirements/api.txt

# One-time pre-build steps (skip if already built for this TORNET_ROOT)
python scripts/build_madis_eligible_catalog.py
python scripts/build_madis_storm_ids_cache.py

python api/app.py   # serves on :5050 (override with PORT)
```

Key routes:

* `GET /health` — liveness check
* `GET /predict/<storm_id>` — runs a historical storm through all three models, returns per-model probabilities plus a GeoJSON point and georeferenced radar PNG
* `GET /storms/madis` — storm IDs with confirmed MADIS coverage
* `GET /live/sites` — live NEXRAD polling status per site
* `GET /live/<site>/heatmap` — latest tornado-probability heatmap tiles for a live site

Full route/response documentation, environment variables, and error codes are in `api/README.md`.

## Repo Map

| Path | What's there |
|---|---|
| `tornet/` | Core package — data loaders, CNN model, metrics/losses |
| `scripts/tornado_detection/` | Train/test entrypoints and per-variant config JSON |
| `scripts/build_madis_*.py` | MADIS feature/catalog/cache build steps |
| `api/` | Flask serving layer + live NEXRAD pipeline (`live/`) |
| `notebooks/` | Data loading, visualization, MADIS exploration, and result-analysis notebooks |
| `results/` | Trained checkpoints, training histories, and params per experiment run |
| `MADIS_DATA_FINDINGS.md` | MADIS coverage, feature predictive power, and fusion-architecture analysis |

### Disclosure
```
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
```

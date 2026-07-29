"""
Loads the no_madis baseline checkpoint once and runs batched tile inference
for the live pipeline. Independent of api/app.py's own MODEL (which stays
on the MADIS-FiLM checkpoint for the historical /predict/<storm_id> path)
-- these are different graphs, not swappable.
"""
import json
import os
import threading

import keras
import numpy as np

from tornet.models.keras.cnn_baseline import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "results", "run_july_6th", "results")
MODEL_PATH = os.path.join(MODEL_DIR, "no_madis_run2_best.keras")
PARAMS_PATH = os.path.join(MODEL_DIR, "no_madis_run2_params.json")


def _load_model():
    with open(PARAMS_PATH) as f:
        raw = json.load(f)
    params = raw.get("config", raw)
    model = build_model(
        head=params.get("head", "mlp"),
        start_filters=params.get("start_filters", 48),
        l2_reg=params.get("l2_reg", 1e-5),
        use_madis=False,
    )
    model.load_weights(MODEL_PATH)
    return model


MODEL = _load_model()

# Guards concurrent calls to MODEL(...) when multiple sites are processed in
# parallel (live/scheduler.py's thread pool). Per-site tile counts vary
# slightly, so concurrent calls could trigger concurrent retracing for a new
# batch shape -- an unhardened, unverified-safe path not worth risking.
# Serializes ~4s of work per site; everything else (I/O, decode, tiling)
# still runs concurrently around this lock.
_MODEL_LOCK = threading.Lock()


def run_batch(tile_inputs):
    """
    tile_inputs: list of per-tile input dicts (from
    live.tiling.build_model_input), each already batch-of-1 shaped along
    every key. Stacks them into a single batch and runs one forward pass
    (NOT model.predict() -- see the documented deadlock gotcha in
    api/app.py -- calling the model directly avoids it here too, now
    validated at batch size >1).

    Returns a 1D numpy array of probabilities, same order as tile_inputs.
    """
    if not tile_inputs:
        return np.array([])

    keys = list(MODEL.input.keys())
    batched = {k: np.concatenate([t[k] for t in tile_inputs], axis=0) for k in keys}
    with _MODEL_LOCK:
        logits = keras.ops.convert_to_numpy(MODEL(batched, training=False))
    return 1.0 / (1.0 + np.exp(-logits[:, 0]))

"""
Flask API wrapping three trained tornado detectors (no-MADIS baseline,
MADIS late-fusion "hybrid", and MADIS FiLM), plus a live NEXRAD tiling
pipeline (no_madis baseline) for real-time storm data.

Given a storm_id (NetCDF event_id), loads the most recent matching radar
frame, runs it through all three models via the existing TorNet
preprocessing pipeline, and returns a tornado probability per model, a
GeoJSON point for the storm location, and a georeferenced reflectivity PNG
(Py-ART) for use with Leaflet's imageOverlay. All three share the
run_july_6th/results "run2" checkpoint for an apples-to-apples comparison
(same trial index across the no_madis / madis_hybrid / madis_film sweeps).

Separately, a background thread (live/scheduler.py) polls NOAA's live NEXRAD
S3 archive for a handful of configured sites, tiles each volume into
overlapping crops matching the model's fixed input shape, and serves a
probability heatmap per site -- see live/ and api/live_routes.py.

Run locally:
    export TORNET_ROOT=/path/to/tornet_data
    export KERAS_BACKEND=tensorflow
    pip install -r requirements/api.txt
    python scripts/build_madis_eligible_catalog.py       # one-time, if not already built
    python scripts/build_madis_storm_ids_cache.py        # one-time, builds the /storms/madis cache
    python api/app.py

Test:
    curl http://localhost:5000/health
    curl http://localhost:5000/predict/1000151
    curl http://localhost:5000/storms/madis
    curl http://localhost:5000/live/sites
    curl http://localhost:5000/live/KTLX/heatmap

Note: the two MADIS models require MADIS surface-weather coverage. For a
storm_id lacking nearby MADIS station data, /predict still 200s with the
no_madis probability, but "madis" and "madis_film" come back as
{"probability": None, "available": False} (see read_file's rejection logic
in tornet/data/loader.py) -- this is expected, not a bug. GET /storms/madis
lists exactly the storm_ids where all three predictions will be available.

Live sites are configured via TORNET_LIVE_SITES (default: KTLX,KFWS,KOUN)
and poll on TORNET_LIVE_POLL_INTERVAL seconds (default: 120). It can take
up to one poll interval after startup before /live/<site>/heatmap returns
data (503 until then).
"""
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import (headless rendering, no display)

import keras
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES, MADIS_MIN_MAX, MADIS_TOP3_MIN_MAX
from tornet.data.loader import read_file
from tornet.models.keras.cnn_baseline import build_model

from radar_image import render_radar_png

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("TORNET_ROOT", os.path.join(REPO_ROOT, "tornet_data"))

# `live/` is a repo-root package (not pip-installed like tornet), so make it
# importable regardless of this script's working directory or invocation style.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from live import scheduler as live_scheduler
import live_routes

MODEL_DIR = os.path.join(REPO_ROOT, "results", "run_july_6th", "results")

# Same trial index ("run2") across all three sweeps, so the three-way
# comparison isn't confounded by which random trial happened to win.
MODEL_VARIANTS = {
    "no_madis": "no_madis_run2",
    "madis": "madis_hybrid_run2",
    "madis_film": "madis_film_run2",
}

CATALOG_PATH = os.path.join(DATA_ROOT, "catalog.csv")
MADIS_STORM_IDS_PATH = os.path.join(DATA_ROOT, "madis_eligible_storm_ids.txt")


def _load_model(name):
    model_path = os.path.join(MODEL_DIR, f"{name}_best.keras")
    params_path = os.path.join(MODEL_DIR, f"{name}_params.json")
    with open(params_path) as f:
        raw = json.load(f)
    params = raw.get("config", raw)

    use_madis = params.get("use_madis_data", False)
    madis_feature_set = params.get("madis_feature_set", "full")
    madis_mm = MADIS_TOP3_MIN_MAX if madis_feature_set == "top3" else MADIS_MIN_MAX

    model = build_model(
        head=params.get("head", "maxpool"),
        head_units=(1024, 512),
        use_madis=use_madis,
        madis_min_max=madis_mm if use_madis else None,
        start_filters=params.get("start_filters", 48),
        l2_reg=params.get("l2_reg", 1e-5),
        madis_fusion=params.get("madis_fusion", "late"),
    )
    model.load_weights(model_path)
    return model, params


# {variant_key: (keras.Model, params_dict)}
MODELS = {key: _load_model(name) for key, name in MODEL_VARIANTS.items()}
CATALOG = pd.read_csv(CATALOG_PATH, parse_dates=["start_time", "end_time"])

with open(MADIS_STORM_IDS_PATH) as f:
    MADIS_STORM_IDS = [int(line) for line in f if line.strip()]


def predict_storm(storm_id_raw):
    try:
        storm_id = int(storm_id_raw)
    except (TypeError, ValueError):
        abort(400, description=f"storm_id must be an integer, got {storm_id_raw!r}")

    rows = CATALOG[CATALOG["event_id"] == storm_id]
    if rows.empty:
        abort(404, description=f"No storm found with storm_id {storm_id}")

    row = rows.sort_values("start_time").iloc[-1]
    filepath = os.path.join(DATA_ROOT, row["filename"])
    if not os.path.exists(filepath):
        abort(404, description=f"Radar sample file missing on disk for storm_id {storm_id}")

    # Try once with MADIS attached (serves all three models); only re-read
    # without it if this storm has no nearby station coverage, so the
    # no_madis model can still produce a prediction.
    data = read_file(
        filepath,
        variables=ALL_VARIABLES,
        n_frames=1,
        tilt_last=True,
        use_madis_data=True,
        madis_feature_set="full",
    )
    madis_available = data is not None
    if data is None:
        data = read_file(filepath, variables=ALL_VARIABLES, n_frames=1, tilt_last=True, use_madis_data=False)
    if data is None:
        abort(404, description=f"Radar sample could not be read for storm_id {storm_id}")

    pp.add_coordinates(data, include_az=False, backend=np, tilt_last=True)
    data["coordinates"] = data["coordinates"][None, ...]
    if "madis" in data:
        data["madis"] = data["madis"][None, ...]

    predictions = {}
    for key, (model, params) in MODELS.items():
        if params.get("use_madis_data", False) and not madis_available:
            predictions[key] = {"probability": None, "available": False}
            continue
        x = pp.select_keys(data, keys=list(model.input.keys()))
        # NOTE: model.predict() spins up Keras's threaded data-pipeline machinery
        # meant for large datasets, which deadlocks for single-sample inference in
        # some sandboxed environments. A direct call avoids that and is faster.
        logits = keras.ops.convert_to_numpy(model(x, training=False))
        probability = float(1.0 / (1.0 + np.exp(-logits[0, 0])))
        predictions[key] = {"probability": probability, "available": True}

    radar_image_url, radar_image_bounds = None, None
    try:
        radar_image_url, radar_image_bounds = render_radar_png(filepath, storm_id)
        radar_image_url = request.host_url.rstrip("/") + radar_image_url
    except Exception:
        logger.exception(f"Radar image rendering failed for storm_id {storm_id}")

    return {
        "storm_id": storm_id,
        "predictions": predictions,
        "geojson": {
            "type": "Point",
            "coordinates": [float(row["lon"]), float(row["lat"])],
        },
        "radar_image_url": radar_image_url,
        "radar_image_bounds": radar_image_bounds,
    }


app = Flask(__name__)
CORS(app)  # API is consumed cross-origin by a separate Rails frontend
app.register_blueprint(live_routes.bp)
live_scheduler.start()


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    response = jsonify({"error": e.description})
    response.status_code = e.code
    return response


@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    app.logger.exception("Unhandled error in request")
    response = jsonify({"error": "Internal server error"})
    response.status_code = 500
    return response


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/<storm_id>")
def predict(storm_id):
    return jsonify(predict_storm(storm_id))


@app.route("/storms/madis")
def storms_madis():
    return jsonify({"storm_ids": MADIS_STORM_IDS, "count": len(MADIS_STORM_IDS)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

"""
Flask API wrapping the trained MADIS-FiLM tornado detector, plus a live
NEXRAD tiling pipeline (no_madis baseline) for real-time storm data.

Given a storm_id (NetCDF event_id), loads the most recent matching radar
frame, runs it through the existing TorNet preprocessing pipeline, and
returns a tornado probability, a GeoJSON point for the storm location, and a
georeferenced reflectivity PNG (Py-ART) for use with Leaflet's imageOverlay.

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

Note: the served model requires MADIS surface-weather coverage, so any
storm_id lacking nearby MADIS station data will return a 404 even if its
NetCDF file exists on disk -- this is expected (see read_file's rejection
logic in tornet/data/loader.py), not a bug. GET /storms/madis lists exactly
the storm_ids that will succeed.

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
MODEL_PATH = os.path.join(MODEL_DIR, "madis_film_run2_best.keras")
PARAMS_PATH = os.path.join(MODEL_DIR, "madis_film_run2_params.json")

CATALOG_PATH = os.path.join(DATA_ROOT, "catalog.csv")
MADIS_STORM_IDS_PATH = os.path.join(DATA_ROOT, "madis_eligible_storm_ids.txt")


def _load_model():
    with open(PARAMS_PATH) as f:
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
    model.load_weights(MODEL_PATH)
    return model, params


MODEL, PARAMS = _load_model()
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

    data = read_file(
        filepath,
        variables=ALL_VARIABLES,
        n_frames=1,
        tilt_last=True,
        use_madis_data=PARAMS.get("use_madis_data", False),
        madis_feature_set=PARAMS.get("madis_feature_set", "full"),
    )
    if data is None:
        abort(404, description=f"No MADIS surface-weather coverage available for storm_id {storm_id}")

    pp.add_coordinates(data, include_az=False, backend=np, tilt_last=True)
    data["coordinates"] = data["coordinates"][None, ...]
    if "madis" in data:
        data["madis"] = data["madis"][None, ...]

    x = pp.select_keys(data, keys=list(MODEL.input.keys()))
    # NOTE: model.predict() spins up Keras's threaded data-pipeline machinery
    # meant for large datasets, which deadlocks for single-sample inference in
    # some sandboxed environments. A direct call avoids that and is faster.
    logits = keras.ops.convert_to_numpy(MODEL(x, training=False))
    probability = float(1.0 / (1.0 + np.exp(-logits[0, 0])))

    radar_image_url, radar_image_bounds = None, None
    try:
        radar_image_url, radar_image_bounds = render_radar_png(filepath, storm_id)
        radar_image_url = request.host_url.rstrip("/") + radar_image_url
    except Exception:
        logger.exception(f"Radar image rendering failed for storm_id {storm_id}")

    return {
        "storm_id": storm_id,
        "probability": probability,
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

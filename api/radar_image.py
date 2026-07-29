"""
Renders a georeferenced reflectivity PNG for a single radar sweep using Py-ART,
for display as a Leaflet imageOverlay on the Rails side.

TorNet NetCDF samples are pre-cropped range/azimuth windows, not full radar
volumes, so a minimal Py-ART Radar object is constructed here (from the sample's
azimuth/range/elevation coordinate arrays plus the radar site's lat/lon) purely
to reuse Py-ART's antenna-to-ground georeferencing (`gate_longitude`/`gate_latitude`).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyart
import xarray as xr

from tornet.display.display import get_refl_cmap

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "radar")
os.makedirs(STATIC_DIR, exist_ok=True)

# Matches the minimum realistic radar range used for CoordConv coordinates
# in tornet/data/preprocess.py's compute_coordinates.
MIN_RANGE_M = 2125.0


def _build_radar(filepath, sweep_idx):
    with xr.open_dataset(filepath) as ds:
        azimuth = ds["azimuth"].values.astype(np.float64)
        rng = np.clip(ds["range"].values.astype(np.float64), MIN_RANGE_M, None)
        elevation_val = float(ds["elevation"].values[sweep_idx])
        dbz = ds["DBZ"].values[-1, :, :, sweep_idx]
        site_lat = ds.attrs["site_lat"]
        site_lon = ds.attrs["site_lon"]

    n_az = len(azimuth)
    return pyart.core.Radar(
        time={"data": np.zeros(n_az), "units": "seconds since 1970-01-01T00:00:00Z"},
        _range={"data": rng, "units": "meters"},
        fields={"reflectivity": {"data": np.ma.masked_invalid(dbz)}},
        metadata={},
        scan_type="ppi",
        latitude={"data": np.array([site_lat])},
        longitude={"data": np.array([site_lon])},
        altitude={"data": np.array([0.0])},
        sweep_number={"data": np.array([0])},
        sweep_mode={"data": np.array(["ppi"])},
        fixed_angle={"data": np.array([elevation_val])},
        sweep_start_ray_index={"data": np.array([0])},
        sweep_end_ray_index={"data": np.array([n_az - 1])},
        azimuth={"data": azimuth},
        elevation={"data": np.full(n_az, elevation_val)},
    ), dbz


def render_radar_png(filepath, storm_id, sweep_idx=0):
    """
    Returns (image_url, bounds) where image_url is a Flask-servable static path
    and bounds is [[lat_min, lon_min], [lat_max, lon_max]] for Leaflet's
    imageOverlay. Renders once per storm_id and caches the PNG + bounds on disk.
    """
    png_path = os.path.join(STATIC_DIR, f"{storm_id}.png")
    bounds_path = os.path.join(STATIC_DIR, f"{storm_id}.bounds.json")

    if os.path.exists(png_path) and os.path.exists(bounds_path):
        with open(bounds_path) as f:
            return f"/static/radar/{storm_id}.png", json.load(f)

    radar, dbz = _build_radar(filepath, sweep_idx)
    lon = radar.gate_longitude["data"]
    lat = radar.gate_latitude["data"]

    cmap, norm = get_refl_cmap()

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_aspect("auto")
    ax.pcolormesh(lon, lat, dbz, cmap=cmap, norm=norm, shading="nearest")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    fig.savefig(png_path, transparent=True)
    plt.close(fig)

    bounds = [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]]
    with open(bounds_path, "w") as f:
        json.dump(bounds, f)

    return f"/static/radar/{storm_id}.png", bounds

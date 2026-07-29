"""
Renders a georeferenced reflectivity PNG for the live pipeline's lowest
elevation tilt, mirroring api/radar_image.py's historical rendering --
same colormap, same PNG/bounds caching pattern, same Leaflet imageOverlay
contract -- but sourced from live.decode's already-merged split-cut arrays
for a live volume instead of a historical NetCDF file.

Georeferences the FULL sweep (not just tile corners) using the same
antenna-to-ground math already validated in live/tiling.py
(`pyart.core.antenna_to_cartesian` + `cartesian_to_geographic` with the
`pyart_aeqd` projection) -- confirmed against Py-ART's own
gate_longitude/gate_latitude to ~1e-6 degree precision, and fast (~50ms for
a full 720x1832 sweep).
"""
import json
import os
import threading

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyart
from scipy import ndimage

from tornet.display.display import get_refl_cmap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(REPO_ROOT, "api", "static", "radar_live")
os.makedirs(STATIC_DIR, exist_ok=True)

_AEQD_R = 6370997.0  # matches Py-ART's pyart_aeqd default

# Below this correlation coefficient, a gate is much more likely to be ground
# clutter or biological scatter (birds/insects) than real precipitation --
# confirmed empirically against a live KTLX volume: near-radar (<40km) gates
# had a median RHOHV of 0.79 with 79% below 0.9 and only 9% >=0.95, versus a
# 100-200km control range where valid echo was far sparser (18% vs 98% gate
# coverage) and RHOHV ran notably higher (median 0.93). This only affects the
# rendered PNG for visual clarity -- it does NOT touch live/tiling.py's model
# inputs, which still see raw, unfiltered data.
MIN_RHOHV_FOR_DISPLAY = 0.9

# Isolated 1-4 gate specks that survive the RHOHV filter are still visually
# noisy ("salt and pepper") at tight zoom, and checking (empirically, this
# session) whether raising the RHOHV bar further would help revealed it
# doesn't cleanly separate clutter from real storm cores -- intense
# convection (esp. hail-bearing) can legitimately show reduced RHOHV too, so
# an aggressive RHOHV/DBZ combination risks erasing real severe weather.
# Spatial coherence is a safer signal: a real storm core spans many
# contiguous gates, while isolated clutter/biological targets are typically
# 1-4 gates alone. This removes only small, disconnected specks -- deliberately
# conservative so legitimate smaller precipitation features aren't erased too.
MIN_CONNECTED_GATES = 5

# WSR-88D's own onboard clutter filter reports how much power (dB) it
# removed at each gate (clutter_filter_power_removed) -- a real, per-gate
# clutter detection from the radar's own signal processor, not a heuristic.
# Confirmed empirically (this session): 78% of near-radar gates that survive
# the RHOHV filter above were ALREADY flagged by this onboard filter (median
# 10dB removed), meaning most of what's still getting through is clutter the
# radar itself identified. Any gate with a real (non-NaN) value here means
# the onboard filter detected and acted on likely clutter, so it's excluded
# from display outright.


def _despeckle(dbz):
    valid = ~np.isnan(dbz)
    labeled, n = ndimage.label(valid, structure=np.ones((3, 3)))
    if n == 0:
        return dbz
    sizes = ndimage.sum(valid, labeled, range(1, n + 1))
    too_small = np.isin(labeled, np.where(sizes < MIN_CONNECTED_GATES)[0] + 1)
    out = dbz.copy()
    out[too_small] = np.nan
    return out


def _full_sweep_latlon(azimuth, range_m, elevation_deg, site_lat, site_lon):
    az_grid, rng_grid = np.meshgrid(azimuth, range_m / 1000.0, indexing="ij")
    elev_grid = np.full_like(az_grid, elevation_deg)
    x, y, _ = pyart.core.antenna_to_cartesian(rng_grid, az_grid, elev_grid)
    projparams = {"proj": "pyart_aeqd", "lon_0": site_lon, "lat_0": site_lat, "R": _AEQD_R}
    lon, lat = pyart.core.cartesian_to_geographic(x, y, projparams)
    return lat, lon


def _safe_name(site, volume_key):
    return f"{site}_{volume_key.replace('/', '_')}"


# Tracks the currently-active cached filename per site, so cleanup can
# remove exactly the previous file for THIS site -- not a filesystem prefix
# scan, which is fragile (e.g. site="KTLX" would prefix-match a differently
# -named "KTLX_TEST_..." file belonging to something else entirely).
_current_name_by_site = {}

# matplotlib's pyplot module keeps global, process-wide state (the "current
# figure" stack) -- not safe for concurrent use from multiple threads, unlike
# the object-oriented Figure API. Since this is called from every worker
# thread in live/scheduler.py's pool, serialize just the pyplot section
# (~0.75-1s) rather than risk cross-talk between one thread's savefig and
# another's plt.close(fig).
_PYPLOT_LOCK = threading.Lock()


def render_live_radar_png(tilt_data, site, site_lat, site_lon, volume_key):
    """
    tilt_data: the lowest-elevation dict from live.decode.decode_lowest_tilts
    (has 'fields'['DBZ'], 'azimuth', 'range', 'elevation').

    Returns (relative_url, bounds). Cached per (site, volume_key); old
    cached images for the same site are removed once a newer volume is
    rendered, since only the latest is ever served.
    """
    name = _safe_name(site, volume_key)
    png_path = os.path.join(STATIC_DIR, f"{name}.png")
    bounds_path = os.path.join(STATIC_DIR, f"{name}.bounds.json")

    if os.path.exists(png_path) and os.path.exists(bounds_path):
        with open(bounds_path) as f:
            return f"/static/radar_live/{name}.png", json.load(f)

    dbz = np.ma.filled(tilt_data["fields"]["DBZ"], np.nan).copy()
    rhohv = np.ma.filled(tilt_data["fields"]["RHOHV"], np.nan)
    dbz[rhohv < MIN_RHOHV_FOR_DISPLAY] = np.nan

    cfp = tilt_data["fields"].get("CLUTTER_FILTER_POWER_REMOVED")
    if cfp is not None:
        cfp = np.ma.filled(cfp, np.nan)
        dbz[~np.isnan(cfp)] = np.nan

    dbz = _despeckle(dbz)

    lat, lon = _full_sweep_latlon(tilt_data["azimuth"], tilt_data["range"], tilt_data["elevation"], site_lat, site_lon)

    cmap, norm = get_refl_cmap()
    # The full sweep spans ~900km (out to the true unambiguous range, ~458km
    # radius) -- at 1200px that was ~0.76km/pixel, coarser than the radar's
    # actual 250m native gate spacing, so any tight zoom (e.g. a ~40km metro
    # area) blew individual pixels up into visible blocks. This resolution
    # was chosen to get close to native detail without an excessive file size.
    with _PYPLOT_LOCK:
        fig = plt.figure(figsize=(12, 12), dpi=250)
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

    previous_name = _current_name_by_site.get(site)
    if previous_name and previous_name != name:
        for ext in (".png", ".bounds.json"):
            try:
                os.remove(os.path.join(STATIC_DIR, f"{previous_name}{ext}"))
            except OSError:
                pass
    _current_name_by_site[site] = name

    return f"/static/radar_live/{name}.png", bounds

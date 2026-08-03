"""
Slices decoded NEXRAD sweeps (from live.decode) into overlapping 120x240
crops matching the no_madis model's fixed input shape, and georeferences
each tile's footprint to lat/lon.

Tile ray/gate index arithmetic is pure and independent of the model; the
CoordConv `coordinates` channel is computed by reusing
`tornet.data.preprocess.add_coordinates` unmodified, fed the tile's own
az_lower/az_upper (degrees) and rng_lower/rng_upper (meters) -- the same
keys/units the historical pipeline already uses.

Georeferencing is done directly via Py-ART's lower-level antenna-to-ground
math (`pyart.core.antenna_to_cartesian` + `cartesian_to_geographic`) rather
than a full Radar object's `gate_longitude`/`gate_latitude`, since after
live.decode's split-cut merge (nearest-neighbor resampling one sweep onto
another), tile ray indices no longer line up with the original radar
object's global ray indexing.
"""
import numpy as np
import pyart

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES

AZ_TILE = 120
RNG_TILE = 240
AZ_STRIDE = 60   # 50% overlap: 30 deg step at 0.5 deg/ray
RNG_STRIDE = 120  # 50% overlap: 30 km step at 250 m/gate

# Py-ART's default azimuthal-equidistant projection centered on the radar,
# matching what Radar.gate_longitude/gate_latitude use internally when no
# custom projection is configured -- avoids needing pyproj for this.
_AEQD_R = 6370997.0


def _tile_ray_starts(n_rays, tile_size=AZ_TILE, stride=AZ_STRIDE):
    return list(range(0, n_rays, stride))


def _tile_gate_starts(n_gates, tile_size=RNG_TILE, stride=RNG_STRIDE):
    return list(range(0, n_gates - tile_size + 1, stride))


def _wrap_rows(arr, ray_start, tile_size):
    n_rays = arr.shape[0]
    idx = np.arange(ray_start, ray_start + tile_size) % n_rays
    return arr[idx]


def _slice_variables(tilt_data_list, ray_start, gate_start, az_tile, rng_tile):
    """Slices+stacks all TorNet variables across tilts into a (az_tile,
    rng_tile,n_tilts) array per variable, for one tile window starting at
    (ray_start, gate_start) -- ray_start wraps circularly via _wrap_rows,
    gate_start must already be a valid (clamped) offset into range."""
    gate_end = gate_start + rng_tile
    variables = {}
    for var in ALL_VARIABLES:
        stacked = [
            np.ma.filled(_wrap_rows(tilt["fields"][var], ray_start, az_tile)[:, gate_start:gate_end], np.nan).astype(np.float32)
            for tilt in tilt_data_list
        ]
        variables[var] = np.stack(stacked, axis=-1)  # (az_tile,rng_tile,n_tilts)
    variables["range_folded_mask"] = np.zeros((az_tile, rng_tile, len(tilt_data_list)), dtype=np.float32)
    return variables


def _tile_bounds(base, ray_start, gate_start, az_tile, rng_tile):
    """Computes az/range lower/upper + center for a tile window, from the
    base (lowest-tilt) dict's azimuth/range grids."""
    n_rays = len(base["azimuth"])
    az_res = 360.0 / n_rays
    range_arr = base["range"]
    gate_spacing = float(range_arr[1] - range_arr[0])

    az_lower = float(base["azimuth"][ray_start % n_rays])
    rng_lower = float(range_arr[gate_start])

    return {
        "az_lower": az_lower,
        "az_upper": az_lower + az_tile * az_res,
        "rng_lower": rng_lower,
        "rng_upper": rng_lower + rng_tile * gate_spacing,
        "elevation_deg": base["elevation"],
        "center_azimuth_deg": az_lower + (az_tile * az_res) / 2.0,
        "center_range_m": rng_lower + (rng_tile * gate_spacing) / 2.0,
    }


def extract_tiles(tilt_data_list, az_tile=AZ_TILE, rng_tile=RNG_TILE,
                   az_stride=AZ_STRIDE, rng_stride=RNG_STRIDE):
    """
    tilt_data_list: ascending-elevation list of dicts from
    live.decode.merge_elevation (all sharing the same n_rays/gate spacing
    for one volume). Returns a list of tile dicts with per-variable
    (120,240,2) arrays plus az/range bounds and centroid info.
    """
    base = tilt_data_list[0]
    n_rays = len(base["azimuth"])
    n_gates = len(base["range"])

    tiles = []
    for ray_start in _tile_ray_starts(n_rays, az_tile, az_stride):
        for gate_start in _tile_gate_starts(n_gates, rng_tile, rng_stride):
            variables = _slice_variables(tilt_data_list, ray_start, gate_start, az_tile, rng_tile)
            tiles.append({
                "variables": variables,
                **_tile_bounds(base, ray_start, gate_start, az_tile, rng_tile),
            })
    return tiles


def build_model_input(tile):
    """Adds the CoordConv `coordinates` channel + a leading batch-of-1 dim,
    ready for tornet.data.preprocess.select_keys against the model's inputs."""
    d = dict(tile["variables"])
    d["az_lower"] = np.array([tile["az_lower"]])
    d["az_upper"] = np.array([tile["az_upper"]])
    d["rng_lower"] = np.array([tile["rng_lower"]])
    d["rng_upper"] = np.array([tile["rng_upper"]])
    pp.add_coordinates(d, include_az=False, backend=np, tilt_last=True)

    model_keys = list(tile["variables"].keys()) + ["coordinates"]
    return {k: d[k][None, ...] for k in model_keys}


def tile_footprint_latlon(tile, site_lat, site_lon):
    """
    Returns (center_lat, center_lon, bounds) for a tile, where bounds is
    [[lat_min, lon_min], [lat_max, lon_max]] -- computed from the tile's 4
    corners plus center, using the same antenna-to-ground math Py-ART's
    Radar.gate_longitude/gate_latitude use internally (validated against
    real gate_longitude/gate_latitude output for a non-resampled sweep to
    confirm this matches).
    """
    ranges = np.array([
        tile["rng_lower"], tile["rng_upper"], tile["rng_lower"], tile["rng_upper"],
        tile["center_range_m"],
    ])
    azimuths = np.array([
        tile["az_lower"], tile["az_lower"], tile["az_upper"], tile["az_upper"],
        tile["center_azimuth_deg"],
    ])
    elevations = np.full(5, tile["elevation_deg"])

    x, y, _ = pyart.core.antenna_to_cartesian(ranges / 1000.0, azimuths, elevations)
    projparams = {"proj": "pyart_aeqd", "lon_0": site_lon, "lat_0": site_lat, "R": _AEQD_R}
    lon, lat = pyart.core.cartesian_to_geographic(x, y, projparams)

    lats, lons = lat[:4], lon[:4]
    center_lat, center_lon = float(lat[4]), float(lon[4])
    bounds = [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
    return center_lat, center_lon, bounds

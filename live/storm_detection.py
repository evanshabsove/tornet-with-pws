"""
Detects discrete storm cells in a decoded NEXRAD volume via SCIT-inspired
multi-threshold connected-component labeling on lowest-tilt reflectivity,
then extracts one training-shaped (120x240) crop per detected storm --
replacing live.tiling's fixed sliding grid so live inference sees the same
kind of storm-centered single-object crop the model was trained on.

Multi-threshold design mirrors the WSR-88D SCIT algorithm (Johnson et al.
1998): a single fixed reflectivity threshold (as in the simpler TITAN
algorithm) either merges nearby cells together or fails to pinpoint an
intense core sitting inside a broader weak echo region. Processing several
thresholds from highest to lowest, and excluding gates already claimed by a
higher threshold's component, gives each storm a centroid at the most
precise (highest) threshold level it actually reaches, while still
catching weaker standalone storms that never reach a higher threshold.

Two distinct "wrap" problems appear across live/tiling.py and this module,
solved differently:
- live.tiling._wrap_rows wraps azimuth when *extracting* a tile window.
- _components (below) merges two labeled components that are really one
  storm split by scipy.ndimage.label at the 0/360 degree seam, since
  ndimage.label has no notion of the azimuth axis being circular.
"""
import os

import numpy as np
from scipy import ndimage

from live.tiling import _slice_variables, _tile_bounds

AZ_TILE = 120
RNG_TILE = 240

_STRUCTURE = np.ones((3, 3))


def _parse_thresholds():
    raw = os.environ.get("TORNET_LIVE_STORM_DBZ_THRESHOLDS", "30,40,50")
    return tuple(sorted((float(x) for x in raw.split(",")), reverse=True))


DBZ_THRESHOLDS = _parse_thresholds()
MIN_RHOHV = float(os.environ.get("TORNET_LIVE_STORM_MIN_RHOHV", "0.9"))
MIN_AREA_GATES = int(os.environ.get("TORNET_LIVE_STORM_MIN_AREA_GATES", "40"))
DILATION_ITERS = int(os.environ.get("TORNET_LIVE_STORM_DILATION_ITERS", "2"))


def _weighted_circular_mean_index(indices, weights, n_rays):
    """Weighted mean ray-index of a set of indices on a circular axis of
    length n_rays -- a plain arithmetic mean is wrong for a component that
    straddles the 0/360 degree seam (e.g. rows [718,719,0,1] would average
    to ~359.5 instead of correctly landing near 0)."""
    angles = 2 * np.pi * np.asarray(indices) / n_rays
    sin_mean = np.average(np.sin(angles), weights=weights)
    cos_mean = np.average(np.cos(angles), weights=weights)
    mean_angle = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
    return mean_angle * n_rays / (2 * np.pi)


def _components(mask, min_area_gates):
    """Labels mask with 8-connectivity, drops components smaller than
    min_area_gates, and merges any pair of surviving components that touch
    opposite azimuth-wrap edges (row 0 and the last row) into one -- since
    a real storm straddling the seam is otherwise split into two labels.
    Returns a list of boolean masks, one per surviving (possibly merged)
    component."""
    labeled, n = ndimage.label(mask, structure=_STRUCTURE)
    if n == 0:
        return []

    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    keep = {i + 1 for i, size in enumerate(sizes) if size >= min_area_gates}
    if not keep:
        return []

    touches_first = (set(np.unique(labeled[0, :]).tolist()) - {0}) & keep
    touches_last = (set(np.unique(labeled[-1, :]).tolist()) - {0}) & keep

    parent = {label_id: label_id for label_id in keep}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a in touches_first:
        for b in touches_last:
            if a != b:
                parent[find(a)] = find(b)

    groups = {}
    for label_id in keep:
        groups.setdefault(find(label_id), []).append(label_id)

    return [np.isin(labeled, label_ids) for label_ids in groups.values()]


def _component_centroid(component_mask, dbz, threshold, n_rays):
    """Reflectivity-intensity-weighted centroid: weights each component gate
    by how far above the detection threshold it is, so the resulting
    centroid is pulled toward the storm's actual reflectivity core rather
    than the geometric center of its outline. A plain unweighted mean can
    land in a low-reflectivity gap for non-convex shapes (bow echoes, hook
    echoes, squall lines) -- confirmed against a real live KTLX volume,
    where an unweighted centroid placed the tile center on ~7 dBZ
    background despite the component being detected at a 30 dBZ threshold.
    Falls back to an unweighted mean if every gate ties exactly at the
    threshold (weights would otherwise sum to zero)."""
    rows, cols = np.nonzero(component_mask)
    weights = np.maximum(dbz[component_mask] - threshold, 0.0)
    if not np.any(weights):
        weights = np.ones_like(weights)
    centroid_az_idx = _weighted_circular_mean_index(rows, weights, n_rays)
    centroid_rng_idx = float(np.average(cols, weights=weights))
    return centroid_az_idx, centroid_rng_idx


def detect_storms(tilt_data, dbz_thresholds=DBZ_THRESHOLDS, min_rhohv=MIN_RHOHV,
                   min_area_gates=MIN_AREA_GATES, dilation_iterations=DILATION_ITERS):
    """
    tilt_data: the ascending-elevation list from
    live.decode.decode_lowest_tilts. Detection runs on the lowest tilt's
    DBZ+RHOHV only.

    Returns a list of storm dicts: {"centroid_az_idx", "centroid_rng_idx",
    "area_gates", "max_dbz", "threshold"}, one per detected storm cell,
    processed from the highest threshold down so intense cores get their
    own precise centroid instead of being absorbed into a broader, less
    precise lower-threshold detection.
    """
    base = tilt_data[0]
    dbz = np.ma.filled(base["fields"]["DBZ"], -999.0)
    rhohv = np.ma.filled(base["fields"]["RHOHV"], 0.0)
    n_rays = dbz.shape[0]

    valid = rhohv >= min_rhohv
    claimed = np.zeros_like(dbz, dtype=bool)
    storms = []

    for threshold in sorted(dbz_thresholds, reverse=True):
        mask = (dbz >= threshold) & valid & ~claimed
        if dilation_iterations:
            # Bridges small interior gaps (e.g. a borderline-RHOHV gate
            # inside an otherwise-contiguous core) but must not swallow
            # territory already claimed by a higher threshold's storm.
            mask = ndimage.binary_dilation(mask, iterations=dilation_iterations) & ~claimed

        for component_mask in _components(mask, min_area_gates):
            centroid_az_idx, centroid_rng_idx = _component_centroid(component_mask, dbz, threshold, n_rays)
            storms.append({
                "centroid_az_idx": centroid_az_idx,
                "centroid_rng_idx": centroid_rng_idx,
                "area_gates": int(component_mask.sum()),
                "max_dbz": float(dbz[component_mask].max()),
                "threshold": threshold,
            })
            claimed |= component_mask

    return storms


def extract_storm_tiles(tilt_data, storms, az_tile=AZ_TILE, rng_tile=RNG_TILE):
    """
    Builds one tile dict per detected storm, centered on its centroid, in
    the same shape live.tiling.extract_tiles produces -- so
    live.inference.run_batch, live.tiling.build_model_input, and
    live.tiling.tile_footprint_latlon need no changes -- plus additive
    max_dbz/area_gates keys for observability.
    """
    if not storms:
        return []

    base = tilt_data[0]
    n_gates = len(base["range"])

    tiles = []
    for storm in storms:
        ray_start = round(storm["centroid_az_idx"]) - az_tile // 2
        gate_start = round(storm["centroid_rng_idx"]) - rng_tile // 2
        # Azimuth is circular (handled by _slice_variables' _wrap_rows), but
        # range is not -- clamp rather than wrap. A storm centroid within
        # rng_tile/2 gates (~30km) of the radar, or of the far range edge,
        # will not be perfectly centered in the resulting window; accepted
        # v1 approximation.
        gate_start = max(0, min(gate_start, n_gates - rng_tile))

        variables = _slice_variables(tilt_data, ray_start, gate_start, az_tile, rng_tile)
        tiles.append({
            "variables": variables,
            **_tile_bounds(base, ray_start, gate_start, az_tile, rng_tile),
            "max_dbz": storm["max_dbz"],
            "area_gates": storm["area_gates"],
        })
    return tiles

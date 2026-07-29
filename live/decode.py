"""
Decodes a raw NEXRAD Level II volume (via Py-ART) into the same 6
radar-variable arrays the historical TorNet pipeline expects, for the
lowest N nominal elevations.

Validated this session against a real live KTLX volume (see the plan doc
for details) -- three things worth knowing when reading this module:

1. Py-ART's `linear_interp=True` does NOT merge NEXRAD's "split-cut" sweeps
   (where reflectivity/dual-pol and velocity/width are scanned as separate
   sweeps at the same nominal elevation) into one combined sweep. Each
   split-cut sweep has the OTHER moment group fully masked (no real data).
   `merge_split_cut` below does this merge by hand: for each of the 6
   variables, it picks whichever sweep in the pair actually has real data,
   then nearest-neighbor-resamples it onto a shared azimuth grid (the two
   sweeps' same-ray-index azimuths are offset by ~10-15 degrees since they
   scan the full circle independently).

2. KDP is not a raw NEXRAD moment -- it's derived here from differential
   phase via `pyart.retrieve.kdp_vulpiani`, chosen over `kdp_maesaka` after
   empirical comparison: ~17x faster and (unlike maesaka) it naturally
   preserves the input's mask instead of extrapolating values into
   clear-air gates.

3. NEXRAD's range-folded ("RF") sentinel is not reliably distinguishable
   from ordinary missing/below-threshold data through Py-ART's standard
   masked-array field interface (confirmed via the underlying reader's own
   docstring). `range_folded_mask` is therefore always all-zero here --  a
   known, accepted MVP approximation, not a bug.
"""
import logging

import numpy as np
import pyart

logger = logging.getLogger(__name__)

FIELD_MAP = {
    "DBZ": "reflectivity",
    "VEL": "velocity",
    "WIDTH": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "RHOHV": "cross_correlation_ratio",
}
ELEVATION_GROUP_ROUND = 1  # decimal places for grouping sweeps into the same nominal elevation


def read_volume(local_path):
    return pyart.io.read_nexrad_archive(local_path, linear_interp=True)


def _sweep_groups(radar):
    """Groups sweep indices by rounded fixed_angle (nominal elevation)."""
    groups = {}
    for i, angle in enumerate(radar.fixed_angle["data"]):
        key = round(float(angle), ELEVATION_GROUP_ROUND)
        groups.setdefault(key, []).append(i)
    return groups


def lowest_elevation_groups(radar, n=2):
    """Returns the n lowest distinct nominal elevations as (elevation, [sweep_indices])."""
    groups = _sweep_groups(radar)
    return [(elev, groups[elev]) for elev in sorted(groups)[:n]]


def _ray_bounds(radar, sweep_idx):
    s, e = radar.get_start_end(sweep_idx)
    return s, e + 1  # end is exclusive here


def _is_fully_masked(radar, sweep_idx, field_name):
    if field_name not in radar.fields:
        return True
    s, e = _ray_bounds(radar, sweep_idx)
    return bool(np.ma.getmaskarray(radar.fields[field_name]["data"][s:e]).all())


def _nearest_neighbor_resample(src_azimuth, src_data, dst_azimuth):
    diffs = np.abs(src_azimuth[None, :] - dst_azimuth[:, None])
    diffs = np.minimum(diffs, 360 - diffs)
    nn_idx = diffs.argmin(axis=1)
    max_err = diffs[np.arange(len(dst_azimuth)), nn_idx].max()
    if max_err > 0.5:
        logger.warning(f"azimuth nearest-neighbor match error {max_err:.2f} deg exceeds one ray's resolution")
    return src_data[nn_idx]


def _sweep_field(radar, sweep_idx, field_name, dst_azimuth=None):
    s, e = _ray_bounds(radar, sweep_idx)
    data = radar.fields[field_name]["data"][s:e]
    if dst_azimuth is not None:
        az = radar.azimuth["data"][s:e]
        data = _nearest_neighbor_resample(az, data, dst_azimuth)
    return data


def merge_elevation(radar, sweep_indices):
    """
    Combines one or two same-elevation sweeps into a single set of
    (n_rays, n_gates) arrays for all 6 TorNet variables, plus the shared
    azimuth grid and the elevation angle. Picks whichever sweep in the pair
    actually has real data for each field; resamples onto the first sweep's
    azimuth grid if pulling from the other one.
    """
    base_idx = sweep_indices[0]
    s0, e0 = _ray_bounds(radar, base_idx)
    base_azimuth = radar.azimuth["data"][s0:e0]
    elevation = float(radar.fixed_angle["data"][base_idx])

    fields = {}
    for var, field_name in FIELD_MAP.items():
        chosen = next((i for i in sweep_indices if not _is_fully_masked(radar, i, field_name)), sweep_indices[0])
        fields[var] = _sweep_field(radar, chosen, field_name, dst_azimuth=base_azimuth if chosen != base_idx else None)

    phi_idx = next((i for i in sweep_indices if not _is_fully_masked(radar, i, "differential_phase")), sweep_indices[0])
    r_phi = radar.extract_sweeps([phi_idx])
    kdp = pyart.retrieve.kdp_vulpiani(r_phi, psidp_field="differential_phase")[0]["data"]
    if phi_idx != base_idx:
        s, e = _ray_bounds(radar, phi_idx)
        kdp = _nearest_neighbor_resample(radar.azimuth["data"][s:e], kdp, base_azimuth)
    fields["KDP"] = kdp

    fields["range_folded_mask"] = np.zeros(fields["DBZ"].shape, dtype=np.float32)

    # Display-only extra: how much power (dB) WSR-88D's own onboard clutter
    # filter removed at each gate. Not fed to the model (live/tiling.py only
    # reads the 6 TorNet variables + range_folded_mask), used purely to
    # further clean up live/radar_image.py's rendered PNG -- gates the radar
    # itself already flagged as clutter-contaminated are excluded from display.
    if not _is_fully_masked(radar, sweep_indices[0], "clutter_filter_power_removed") or (
        len(sweep_indices) > 1 and not _is_fully_masked(radar, sweep_indices[1], "clutter_filter_power_removed")
    ):
        cfp_idx = next(
            (i for i in sweep_indices if not _is_fully_masked(radar, i, "clutter_filter_power_removed")),
            sweep_indices[0],
        )
        fields["CLUTTER_FILTER_POWER_REMOVED"] = _sweep_field(
            radar, cfp_idx, "clutter_filter_power_removed", dst_azimuth=base_azimuth if cfp_idx != base_idx else None
        )

    return {
        "fields": fields,
        "azimuth": base_azimuth,
        "range": radar.range["data"],
        "elevation": elevation,
    }


def decode_lowest_tilts(local_path, n_tilts=2):
    """
    Returns a list of n_tilts dicts (one per elevation, ascending), each
    from merge_elevation -- ready for live.tiling to slice into model-input
    crops.
    """
    radar = read_volume(local_path)
    groups = lowest_elevation_groups(radar, n=n_tilts)
    return [merge_elevation(radar, idxs) for _, idxs in groups], radar

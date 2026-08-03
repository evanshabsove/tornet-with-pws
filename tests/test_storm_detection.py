"""
Synthetic-array unit tests for live.storm_detection -- no TORNET_ROOT or
real NEXRAD fixture data required, unlike tests/test_tornet.py, since
detection/centroid/windowing logic is pure array math independent of any
real dataset.
"""
import numpy as np

from live import storm_detection, tiling
from tornet.data.constants import ALL_VARIABLES

N_RAYS = 720  # 0.5 deg/ray -> full 360 deg circle, matching real NEXRAD
N_GATES = 1000  # 250 m/gate -> 250 km max range
AZ_RES = 360.0 / N_RAYS
GATE_SPACING = 250.0


def _blank_dbz():
    return np.full((N_RAYS, N_GATES), -999.0, dtype=np.float32)


def _blank_rhohv(value=1.0):
    return np.full((N_RAYS, N_GATES), value, dtype=np.float32)


def _make_tilt_data(dbz, rhohv, n_tilts=1):
    azimuth = np.arange(N_RAYS) * AZ_RES
    range_arr = np.arange(N_GATES) * GATE_SPACING
    fields = {var: np.full((N_RAYS, N_GATES), -999.0, dtype=np.float32) for var in ALL_VARIABLES}
    fields["DBZ"] = dbz
    fields["RHOHV"] = rhohv
    tilt = {"fields": fields, "azimuth": azimuth, "range": range_arr, "elevation": 0.5}
    return [tilt for _ in range(n_tilts)]


def test_single_isolated_blob_detected_and_centered():
    dbz = _blank_dbz()
    dbz[290:310, 490:510] = 45.0  # 20x20 blob, centroid (299.5, 499.5)
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 1
    storm = storms[0]
    assert storm["threshold"] == 40.0
    assert abs(storm["centroid_az_idx"] - 299.5) < 1.0
    assert abs(storm["centroid_rng_idx"] - 499.5) < 1.0
    # >= raw 20x20=400 blob area: binary_dilation (default 2 iterations)
    # grows the mask before labeling/sizing, so reported area exceeds the
    # undilated blob by design.
    assert storm["area_gates"] >= 400
    assert storm["max_dbz"] == 45.0

    tiles = storm_detection.extract_storm_tiles(tilt_data, storms)
    assert len(tiles) == 1
    tile_dbz = tiles[0]["variables"]["DBZ"][:, :, 0]
    # blob should land near the tile's own center (60,120), not off to one side
    assert tile_dbz[55:65, 115:125].max() >= 40.0


def test_subthreshold_area_speck_is_filtered():
    dbz = _blank_dbz()
    dbz[290:310, 490:510] = 45.0  # real storm, area 400
    dbz[600:603, 800:803] = 32.0  # 3x3=9 gates, below min_area_gates default (40)
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 1
    assert abs(storms[0]["centroid_az_idx"] - 299.5) < 1.0


def test_azimuth_wrap_boundary_storm_merges_into_one():
    dbz = _blank_dbz()
    # 20 rows total straddling the 0/720 seam: rows [710,720) and [0,10)
    dbz[710:720, 490:510] = 45.0
    dbz[0:10, 490:510] = 45.0
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 1
    centroid = storms[0]["centroid_az_idx"]
    wrap_distance = min(centroid, N_RAYS - centroid)
    assert wrap_distance < 2.0
    assert storms[0]["area_gates"] >= 400  # dilation grows area beyond the raw 20x20 blob


def test_near_radar_storm_clamps_gate_start_to_zero():
    dbz = _blank_dbz()
    dbz[300:320, 45:65] = 45.0  # centroid_rng_idx ~ 54.5, well within rng_tile/2=120 of gate 0
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 1
    tiles = storm_detection.extract_storm_tiles(tilt_data, storms)
    assert tiles[0]["rng_lower"] == 0.0


def test_far_range_storm_clamps_gate_start():
    dbz = _blank_dbz()
    dbz[300:320, N_GATES - 30:N_GATES - 10] = 45.0
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 1
    tiles = storm_detection.extract_storm_tiles(tilt_data, storms)
    expected_gate_start = N_GATES - storm_detection.RNG_TILE
    assert tiles[0]["rng_lower"] == expected_gate_start * GATE_SPACING


def test_two_separated_storms_detected_independently():
    dbz = _blank_dbz()
    dbz[100:120, 200:220] = 45.0
    dbz[500:520, 700:720] = 45.0
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 2
    centroids_az = sorted(s["centroid_az_idx"] for s in storms)
    assert abs(centroids_az[0] - 109.5) < 1.0
    assert abs(centroids_az[1] - 509.5) < 1.0


def test_close_storms_detected_distinctly_with_overlapping_windows():
    dbz = _blank_dbz()
    dbz[300:320, 400:420] = 45.0
    dbz[300:320, 440:460] = 45.0  # 20-gate gap keeps them non-contiguous
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 2  # distinct components -- NOT merged

    tiles = storm_detection.extract_storm_tiles(tilt_data, storms)
    lowers = sorted(t["rng_lower"] for t in tiles)
    uppers = sorted(t["rng_upper"] for t in tiles)
    # documented v1 tradeoff: close storms get overlapping 120x240 windows
    assert lowers[1] < uppers[0]


def test_empty_volume_returns_no_storms():
    dbz = _blank_dbz()
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert storms == []

    tiles = storm_detection.extract_storm_tiles(tilt_data, storms)
    assert tiles == []


def test_storm_tile_compatible_with_existing_tiling_helpers():
    dbz = _blank_dbz()
    dbz[290:310, 490:510] = 45.0
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    tiles = storm_detection.extract_storm_tiles(tilt_data, storms)

    model_input = tiling.build_model_input(tiles[0])
    expected_keys = set(ALL_VARIABLES) | {"range_folded_mask", "coordinates"}
    assert set(model_input.keys()) == expected_keys
    for arr in model_input.values():
        assert arr.shape[0] == 1  # batch-of-1

    center_lat, center_lon, bounds = tiling.tile_footprint_latlon(tiles[0], site_lat=35.0, site_lon=-97.5)
    assert isinstance(center_lat, float)
    assert isinstance(center_lon, float)
    assert len(bounds) == 2


def test_multi_threshold_nested_core_yields_one_storm():
    # 8x8 intense core (>=50 dBZ, area 64 >= min_area_gates) surrounded by a
    # thin 1-gate ring (35 dBZ, ring area 36 < min_area_gates) -- the ring
    # should be filtered as sub-threshold-area noise, leaving exactly the
    # core as a single storm rather than two nested detections.
    dbz = _blank_dbz()
    dbz[295:305, 495:505] = 35.0  # 10x10 envelope, area 100
    dbz[296:304, 496:504] = 55.0  # 8x8 core, area 64 -- ring left = 36
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 1
    assert storms[0]["threshold"] == 50.0
    # >= raw 8x8=64 core area: dilation grows the mask before sizing; the
    # key assertion is len(storms) == 1 -- the ring stayed filtered even
    # after dilation growth, i.e. wasn't inflated enough to survive as its
    # own component.
    assert storms[0]["area_gates"] >= 64


def test_weighted_centroid_avoids_empty_concave_center():
    # A square annulus ("ring") is a concave shape whose plain geometric
    # (unweighted) center falls in its own hollow middle -- not on the ring
    # itself. This reproduces a real bug found against a live KTLX volume:
    # an unweighted centroid for a non-convex storm shape landed the tile
    # window on ~7 dBZ background instead of the storm's own reflectivity.
    # Reflectivity-intensity weighting (via a brighter arc segment here)
    # should pull the centroid onto real signal instead.
    dbz = _blank_dbz()
    dbz[100:140, 500:540] = 31.0  # 40x40 outer square
    dbz[110:130, 510:530] = -999.0  # carve out a 20x20 hollow center -> ring shape
    dbz[100:110, 500:540] = 38.0  # bright top arc of the ring (stays < 40 so it isn't peeled off as its own higher-threshold storm)

    tilt_data = _make_tilt_data(dbz, _blank_rhohv())
    # dilation_iterations=0 keeps the detected mask identical to the raw
    # ring shape above, so the expected weighted centroid below can be
    # computed independently and compared exactly.
    storms = storm_detection.detect_storms(tilt_data, dilation_iterations=0)
    assert len(storms) == 1
    storm = storms[0]
    assert storm["threshold"] == 30.0

    ring_mask = dbz >= 30.0
    rows, cols = np.nonzero(ring_mask)
    weights = dbz[ring_mask] - 30.0
    expected_az = storm_detection._weighted_circular_mean_index(rows, weights, N_RAYS)
    expected_rng = float(np.average(cols, weights=weights))
    assert abs(storm["centroid_az_idx"] - expected_az) < 0.5
    assert abs(storm["centroid_rng_idx"] - expected_rng) < 0.5

    # The naive geometric center of the full 40x40 bounding square is
    # (120, 520) -- squarely in the carved-out hollow. The whole point of
    # weighting is that the real centroid lands well clear of that, on
    # actual reflectivity rather than the -999 hole.
    assert abs(storm["centroid_az_idx"] - 120) > 5
    landing_dbz = dbz[round(storm["centroid_az_idx"]), round(storm["centroid_rng_idx"])]
    assert landing_dbz >= 30.0


def test_multi_threshold_separate_weak_storm_not_suppressed():
    dbz = _blank_dbz()
    dbz[295:305, 495:505] = 35.0
    dbz[296:304, 496:504] = 55.0  # intense core storm, as above

    dbz[600:625, 800:825] = 32.0  # wholly separate, weak-only storm (25x25=625 >= min_area, never reaches 40 dBZ)
    tilt_data = _make_tilt_data(dbz, _blank_rhohv())

    storms = storm_detection.detect_storms(tilt_data)
    assert len(storms) == 2
    thresholds = sorted(s["threshold"] for s in storms)
    assert thresholds == [30.0, 50.0]

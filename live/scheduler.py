"""
Background thread pool: for each configured site, on a poll interval,
fetches the latest NEXRAD volume, decodes it, tiles it, runs batched
inference, and writes the result into live.cache.CACHE. This is the only
writer to that cache; Flask handlers only read it.

Sites are processed CONCURRENTLY via a ThreadPoolExecutor (not separate
processes -- avoids each worker needing its own ~625MB model copy, since
threads can share the single already-loaded MODEL in live/inference.py).
Two steps are intentionally serialized behind their own locks for
correctness (see live/inference.py's _MODEL_LOCK and live/radar_image.py's
_PYPLOT_LOCK) since concurrent-safety there was unverified, not because
threading itself is unsafe -- everything else (S3 download, decode, KDP,
tiling) runs genuinely concurrently.

In-process thread pool rather than a separate cron/worker process or
multiprocessing -- this repo has no task-queue infra, and separate
processes would each need their own model instance (extra memory/load
time). Known limitation: this only works cleanly for a single-process dev
server; a multi-worker WSGI deployment would run one redundant ingestion
pool per worker (see plan doc).
"""
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from live import decode, ingest, tiling, inference, storm_detection
from live.cache import CACHE
from live.radar_image import render_live_radar_png

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = int(os.environ.get("TORNET_LIVE_POLL_INTERVAL", "120"))
MAX_WORKERS = int(os.environ.get("TORNET_LIVE_MAX_WORKERS", "6"))

_DEMO_SITES = ["KTLX", "KFWS", "KVNX"]


def _resolve_sites():
    """Explicit env var override wins (fast local-iteration path);
    otherwise dynamically discover the real WSR-88D roster from S3, falling
    back to a small demo list if discovery fails at import time (so a
    transient network hiccup can't prevent the app from booting)."""
    override = os.environ.get("TORNET_LIVE_SITES")
    if override:
        return [s.strip() for s in override.split(",") if s.strip()]
    try:
        sites = ingest.discover_sites()
        if sites:
            return sites
        logger.warning("discover_sites() returned no sites, falling back to demo list")
    except Exception:
        logger.exception("discover_sites() failed, falling back to demo list")
    return _DEMO_SITES


SITES = _resolve_sites()

_last_processed_key = {}
_thread = None

# Sites currently being processed by a worker -- the scheduler must never
# resubmit a site whose previous cycle's future hasn't completed yet, or
# two concurrent workers on the SAME site can race on
# live/radar_image.py's _current_name_by_site bookkeeping (a straggler from
# an old volume finishing after a newer one completed could revert the
# cached filename back to stale data, deleting the fresh PNG).
_in_flight = set()
_in_flight_lock = threading.Lock()


def _volume_time_from_key(key):
    """SITE_YYYYMMDD_HHMMSS_V06 -> aware UTC datetime."""
    fname = key.split("/")[-1]
    date_part, time_part = fname.split("_")[0][-8:], fname.split("_")[1]
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def process_site(site):
    local_path = None
    try:
        local_path, key = ingest.fetch_latest(site)
        if key is None:
            CACHE.set_error(site, "no volume found in S3 for this site")
            return
        if _last_processed_key.get(site) == key:
            return  # already processed this exact volume

        tilt_data, radar = decode.decode_lowest_tilts(local_path)
        site_lat = float(radar.latitude["data"][0])
        site_lon = float(radar.longitude["data"][0])

        storms = storm_detection.detect_storms(tilt_data)
        tiles = storm_detection.extract_storm_tiles(tilt_data, storms)
        tile_inputs = [tiling.build_model_input(t) for t in tiles]
        probs = inference.run_batch(tile_inputs)

        # An empty `tiles` list (no storms detected) is a valid outcome --
        # e.g. clear air or stratiform-only weather -- not an error; it
        # still writes an OK cache entry with an empty FeatureCollection
        # below rather than leaving a stale/error cache entry.
        features = []
        for tile, prob in zip(tiles, probs):
            center_lat, center_lon, bounds = tiling.tile_footprint_latlon(tile, site_lat, site_lon)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]},
                "properties": {
                    "probability": float(prob),
                    "bounds": bounds,
                    "sweep_elevation_deg": tile["elevation_deg"],
                    "detection_max_dbz": tile.get("max_dbz"),
                    "detection_area_gates": tile.get("area_gates"),
                },
            })

        radar_image_url, radar_image_bounds = None, None
        try:
            radar_image_url, radar_image_bounds = render_live_radar_png(tilt_data[0], site, site_lat, site_lon, key)
        except Exception:
            logger.exception(f"live radar image rendering failed for site {site}")

        geojson = {"type": "FeatureCollection", "features": features}
        CACHE.set_ok(site, key, _volume_time_from_key(key).isoformat(), geojson, radar_image_url, radar_image_bounds)
        _last_processed_key[site] = key
        logger.info(f"live pipeline updated {site}: {len(features)} tiles from {key}")
    except Exception as e:
        logger.exception(f"live pipeline failed for site {site}")
        CACHE.set_error(site, str(e))
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


def _run_and_release(site):
    try:
        process_site(site)
    except Exception:
        logger.exception(f"unexpected top-level error processing site {site}")
    finally:
        with _in_flight_lock:
            _in_flight.discard(site)


def _loop(executor):
    while True:
        submitted = 0
        with _in_flight_lock:
            to_submit = [s for s in SITES if s not in _in_flight]
            _in_flight.update(to_submit)
        for site in to_submit:
            executor.submit(_run_and_release, site)
            submitted += 1
        skipped = len(SITES) - submitted
        if skipped:
            logger.info(f"scheduler cycle: submitted {submitted}, skipped {skipped} still in-flight from last cycle")
        time.sleep(POLL_INTERVAL_S)


def start():
    """Idempotent: starts the background scheduler thread + worker pool
    once, no-op if already running."""
    global _thread
    if _thread is not None:
        return
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="live-nexrad-worker")
    _thread = threading.Thread(target=_loop, args=(executor,), daemon=True, name="live-nexrad-scheduler")
    _thread.start()
    logger.info(f"live scheduler started: {len(SITES)} sites, max_workers={MAX_WORKERS}, poll_interval={POLL_INTERVAL_S}s")

"""
In-memory, thread-locked per-site cache. live/scheduler.py is the only
writer; Flask request handlers only read. This decouples "how often we
compute" (once per poll interval) from "how often the frontend polls."
"""
import threading
from datetime import datetime, timezone


class LiveCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._by_site = {}

    def set_ok(self, site, volume_key, volume_time, tiles_geojson, radar_image_url=None, radar_image_bounds=None):
        with self._lock:
            self._by_site[site] = {
                "status": "ok",
                "volume_key": volume_key,
                "volume_time": volume_time,
                "generated_at": datetime.now(timezone.utc),
                "tiles": tiles_geojson,
                "radar_image_url": radar_image_url,
                "radar_image_bounds": radar_image_bounds,
                "error": None,
            }

    def set_error(self, site, message):
        with self._lock:
            existing = self._by_site.get(site, {})
            self._by_site[site] = {
                **existing,
                "status": "error",
                "error": message,
                "generated_at": datetime.now(timezone.utc),
            }

    def get(self, site):
        with self._lock:
            entry = self._by_site.get(site)
            return dict(entry) if entry else None

    def sites_status(self, configured_sites):
        with self._lock:
            out = []
            for site in configured_sites:
                entry = self._by_site.get(site)
                out.append({
                    "site": site,
                    "status": entry.get("status") if entry else "pending",
                    "volume_time": entry.get("volume_time") if entry else None,
                })
            return out


CACHE = LiveCache()

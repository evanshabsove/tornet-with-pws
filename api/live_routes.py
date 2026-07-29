"""
Flask blueprint for the live NEXRAD tiling pipeline (no_madis baseline).
Read-only against live.cache.CACHE -- live/scheduler.py's background
thread is the only writer.
"""
from datetime import datetime, timezone

from flask import Blueprint, abort, jsonify, request

from live.cache import CACHE
from live.scheduler import SITES

bp = Blueprint("live", __name__)


@bp.route("/live/sites")
def live_sites():
    statuses = CACHE.sites_status(SITES)
    summary = {"total": len(statuses), "ok": 0, "error": 0, "pending": 0}
    for s in statuses:
        summary[s["status"]] = summary.get(s["status"], 0) + 1
    return jsonify({"sites": statuses, "summary": summary})


@bp.route("/live/<site>/heatmap")
def live_heatmap(site):
    if site not in SITES:
        # Configured-sites list can be ~155 entries at full scale -- not
        # dumped into the error message, see GET /live/sites for the roster.
        abort(404, description=f"Site {site} is not configured for live polling.")

    entry = CACHE.get(site)
    if entry is None or entry.get("status") != "ok":
        message = entry.get("error") if entry else "no ingestion has completed yet"
        abort(503, description=f"No live data yet for site {site}: {message}")

    staleness_seconds = (datetime.now(timezone.utc) - entry["generated_at"]).total_seconds()

    radar_image_url = entry.get("radar_image_url")
    if radar_image_url:
        radar_image_url = request.host_url.rstrip("/") + radar_image_url

    return jsonify({
        "site": site,
        "volume_time": entry["volume_time"],
        "generated_at": entry["generated_at"].isoformat(),
        "staleness_seconds": staleness_seconds,
        "tiles": entry["tiles"],
        "radar_image_url": radar_image_url,
        "radar_image_bounds": entry.get("radar_image_bounds"),
    })

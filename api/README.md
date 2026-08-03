# Tornado Detection API

Flask API serving three trained tornado detectors against historical TorNet radar data, plus a live NEXRAD tiling pipeline for real-time storm probability heatmaps.

## Setup

```bash
export TORNET_ROOT=/path/to/tornet_data
export KERAS_BACKEND=tensorflow
pip install -r requirements/api.txt

# One-time pre-build steps (only needed on first run or after re-downloading data)
python scripts/build_madis_eligible_catalog.py
python scripts/build_madis_storm_ids_cache.py

python api/app.py   # serves on :5050 by default
```

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `TORNET_ROOT` | `./tornet_data` | Root directory for the TorNet dataset |
| `KERAS_BACKEND` | — | `tensorflow`, `torch`, or `jax` |
| `PORT` | `5050` | Port the Flask server binds to |
| `TORNET_LIVE_SITES` | auto-discovered from S3 | Comma-separated NEXRAD site codes to poll (e.g. `KTLX,KFWS,KOUN`) |
| `TORNET_LIVE_POLL_INTERVAL` | `120` | Seconds between live NEXRAD polls |
| `TORNET_LIVE_MAX_WORKERS` | `6` | Concurrent site workers in the background thread pool |

---

## Routes

### `GET /health`

Liveness check. Returns `200` when the server is up.

**Response**
```json
{ "status": "ok" }
```

---

### `GET /predict/<storm_id>`

Runs a historical TorNet radar frame through all three model variants and returns a tornado probability per model.

**Path parameter**

| Parameter | Type | Description |
|---|---|---|
| `storm_id` | integer | The `event_id` from `catalog.csv` (e.g. `1000151`) |

**Response**
```json
{
  "storm_id": 1000151,
  "predictions": {
    "no_madis":   { "probability": 0.83, "available": true },
    "madis":      { "probability": 0.81, "available": true },
    "madis_film": { "probability": 0.86, "available": true }
  },
  "geojson": {
    "type": "Point",
    "coordinates": [-97.5, 35.2]
  },
  "radar_image_url": "http://localhost:5050/static/radar_1000151.png",
  "radar_image_bounds": [[-97.8, 34.9], [-97.2, 35.5]]
}
```

**Probability notes**
- All probabilities are in `[0.0, 1.0]`. Multiply by 100 for a percentage display — the API does not scale.
- `probability` is `null` and `available` is `false` for the two MADIS models when the storm has no nearby MADIS station coverage. The `no_madis` model always returns a result.
- `radar_image_url` / `radar_image_bounds` may be `null` if radar PNG rendering fails (non-fatal).

**Model variants**

| Key | Checkpoint | Uses MADIS | Fusion strategy |
|---|---|---|---|
| `no_madis` | `no_madis_run2_best.keras` | No | — |
| `madis` | `madis_hybrid_run2_best.keras` | Yes | Late fusion (concat after flatten) |
| `madis_film` | `madis_film_run2_best.keras` | Yes | FiLM conditioning on CNN features |

**Error responses**

| Code | Cause |
|---|---|
| `400` | `storm_id` is not an integer |
| `404` | No matching storm in catalog, file missing on disk, or sample unreadable |

---

### `GET /storms/madis`

Lists all storm IDs where MADIS surface-weather station data is available — i.e. the subset of storms where all three `/predict` variants will return `available: true`.

**Response**
```json
{
  "storm_ids": [1000151, 1000203, ...],
  "count": 412
}
```

---

### `GET /live/sites`

Returns the current polling status for every configured NEXRAD site.

**Response**
```json
{
  "sites": [
    {
      "site": "KTLX",
      "status": "ok",
      "last_updated": "2024-05-01T18:32:00Z"
    },
    {
      "site": "KFWS",
      "status": "pending",
      "last_updated": null
    }
  ],
  "summary": {
    "total": 3,
    "ok": 2,
    "error": 0,
    "pending": 1
  }
}
```

**Site status values**

| Value | Meaning |
|---|---|
| `ok` | At least one poll cycle has completed successfully |
| `pending` | Server just started; no poll cycle has finished yet |
| `error` | Last ingestion attempt failed (see site-level `error` field) |

---

### `GET /live/<site>/heatmap`

Returns the latest tiled tornado-probability heatmap for a single live NEXRAD site.

**Path parameter**

| Parameter | Type | Description |
|---|---|---|
| `site` | string | Four-letter NEXRAD site code (e.g. `KTLX`) |

**Response**
```json
{
  "site": "KTLX",
  "volume_time": "2024-05-01T18:30:00Z",
  "generated_at": "2024-05-01T18:32:05Z",
  "staleness_seconds": 87.3,
  "tiles": [
    {
      "lat": 35.33,
      "lon": -97.28,
      "probability": 0.12
    }
  ],
  "radar_image_url": "http://localhost:5050/static/live_KTLX.png",
  "radar_image_bounds": [[34.9, -97.8], [35.8, -96.7]]
}
```

**`tiles` array** — one entry per overlapping crop of the radar volume. Each tile covers a fixed spatial window centered at `(lat, lon)`, with `probability` in `[0.0, 1.0]` from the `no_madis_run2` model (MADIS data is not available for live frames).

**Error responses**

| Code | Cause |
|---|---|
| `404` | Site code is not in the configured polling list — see `GET /live/sites` for the roster |
| `503` | No ingestion has completed for this site yet (e.g. server just started), or the last ingestion failed |

---

## Quick test

```bash
curl http://localhost:5050/health
curl http://localhost:5050/predict/1000151
curl http://localhost:5050/storms/madis
curl http://localhost:5050/live/sites
curl http://localhost:5050/live/KTLX/heatmap
```

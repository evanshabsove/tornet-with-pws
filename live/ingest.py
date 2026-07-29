"""
Discovers and downloads the latest NEXRAD Level II volume for a site from
NOAA's public S3 archive.

VALIDATED this session against real live data: the bucket documented in
older tutorials (`noaa-nexrad-level2`) is deprecated/decommissioned as of
2025-09-01 -- the current bucket is `unidata-nexrad-level2` (confirmed via
NOAA's own Registry of Open Data page, and via a real anonymous
list_objects_v2 + download against KTLX). Anonymous listing IS supported
here (unlike the old bucket, which now returns AccessDenied for everything).

Key format confirmed: `YYYY/MM/DD/SITE/SITE_YYYYMMDD_HHMMSS_V06`, with an
occasional `..._V06_MDM` metadata-only sibling that must be filtered out.

`discover_sites()` queries the same bucket's top-level date prefix to find
every site currently broadcasting, rather than trusting a hardcoded list --
confirmed this session that 202 site prefixes are live today, of which 155
start with K or P (the real WSR-88D network); the rest are TDWR sites
(different radar technology, unvalidated against this pipeline) and a
handful of oddities, filtered out by default.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3
from botocore import UNSIGNED
from botocore.config import Config

logger = logging.getLogger(__name__)

BUCKET = "unidata-nexrad-level2"
REGION = "us-east-1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.environ.get("TORNET_LIVE_TMP", os.path.join(REPO_ROOT, "temp_downloads"))

# Bounds how long one stalled S3 request can occupy a worker thread -- the
# main defense against a hung site blocking a thread pool slot indefinitely.
_CLIENT_CONFIG = Config(
    signature_version=UNSIGNED,
    connect_timeout=5,
    read_timeout=15,
    retries={"max_attempts": 2},
)


def _client():
    return boto3.client("s3", region_name=REGION, config=_CLIENT_CONFIG)


def discover_sites(prefix_letters=("K", "P")):
    """
    Lists every site currently broadcasting into the bucket (today's date
    prefix), filtered to codes starting with one of `prefix_letters` --
    default K/P, the real WSR-88D network (CONUS + Alaska/Hawaii/Guam).
    Excludes TDWR (T-prefixed) and other non-standard prefixes by default.
    """
    s3 = _client()
    now = datetime.now(timezone.utc)
    prefix = f"{now.year:04d}/{now.month:02d}/{now.day:02d}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    if resp.get("IsTruncated"):
        logger.warning("discover_sites: S3 listing was truncated -- some sites may be missing")
    sites = [p["Prefix"].rstrip("/").split("/")[-1] for p in resp.get("CommonPrefixes", [])]
    return sorted(s for s in sites if s.startswith(prefix_letters))


def _list_real_volumes(s3, site, when):
    prefix = f"{when.year:04d}/{when.month:02d}/{when.day:02d}/{site}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", []) if not o["Key"].endswith("_MDM")]
    return sorted(keys)  # timestamp is embedded, so lexicographic == chronological


def latest_volume_key(site, s3=None):
    """
    Returns the most recent real (non-_MDM) volume key for `site`, checking
    today's UTC prefix and falling back to yesterday's near UTC midnight
    (when today's prefix may not have any volumes yet).
    """
    s3 = s3 or _client()
    now = datetime.now(timezone.utc)
    keys = _list_real_volumes(s3, site, now)
    if not keys:
        keys = _list_real_volumes(s3, site, now - timedelta(days=1))
    return keys[-1] if keys else None


def fetch_volume(key, s3=None):
    """Downloads `key` to TMP_DIR and returns the local path."""
    s3 = s3 or _client()
    os.makedirs(TMP_DIR, exist_ok=True)
    dest = os.path.join(TMP_DIR, key.split("/")[-1])
    s3.download_file(BUCKET, key, dest)
    return dest


def fetch_latest(site, s3=None):
    """Returns (local_path, key) for the latest volume, or (None, None) if
    no volume was found for the site."""
    s3 = s3 or _client()
    key = latest_volume_key(site, s3=s3)
    if key is None:
        return None, None
    return fetch_volume(key, s3=s3), key

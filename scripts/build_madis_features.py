"""
Build the MADIS features CSV used for CNN training.

Pipeline:
  1. Parse T0, T-2h, T-24h XML files from madis_data/ into per-scan DataFrames
  2. Aggregate T0/T-2h/T-24h to one row per storm
  3. Compute temporal anomaly features
  4. Join catalog metadata (ef_number, category, type)
  5. Produce one row per (storm_id, timestamp) with raw T0 features + anomalies
  6. Save to $TORNET_ROOT/madis_features_clean.csv (and .pkl)

All intermediate parse results are cached to CSV so re-runs are instant.

Usage:
    python scripts/build_madis_features.py
"""

import os
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(os.environ["TORNET_ROOT"])
MADIS_DIR = DATA_ROOT / "madis_data"
CATALOG_PATH = DATA_ROOT / "catalog.csv"

CACHE_T0   = DATA_ROOT / "madis_temporal_T0.csv"
CACHE_T2H  = DATA_ROOT / "madis_temporal_T2h.csv"
CACHE_T24H = DATA_ROOT / "madis_temporal_T24h.csv"

OUTPUT_CSV = DATA_ROOT / "madis_features_clean.csv"
OUTPUT_PKL = DATA_ROOT / "madis_features_clean.pkl"

# ── Feature map: XML variable name → clean column name ───────────────────────

_FEAT_MAP = {
    "V-T":      "temperature",
    "V-TD":     "dewpoint",
    "V-RH":     "relative_humidity",
    "V-FF":     "wind_speed",
    "V-DD":     "wind_direction",
    "V-ALTSE":  "pressure",
    "V-FFGUST": "wind_gust",
}
FEAT_COLS = list(_FEAT_MAP.values())

# Tier-1 anomaly features selected by Cohen's d analysis in the notebook
TIER1_FEATURES = [
    "pressure_anomaly_24h",   # |d| = 1.388
    "wind_anomaly_24h",       # |d| = 0.665
    "instability_proxy_T2h",  # |d| = 0.598
    "instability_proxy_T0",   # |d| = 0.485
]

# Tier-2 raw T0 features selected by Cohen's d analysis
TIER2_FEATURES = ["pressure", "wind_gust"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _storm_id_from_path(p: Path) -> int | None:
    try:
        return int(p.stem.split("_")[2])
    except (IndexError, ValueError):
        return None


def _timestamp_from_path(p: Path) -> str | None:
    """Extract timestamp string from T0 filename: madis_data_{id}_{timestamp}_T0.xml"""
    try:
        parts = p.stem.split("_")
        # stem example: "madis_data_1073298_2013-09-02 17:43:30_T0"
        # split on "_" gives: ['madis', 'data', '1073298', '2013-09-02 17:43:30', 'T0']
        # but spaces in timestamp mean we need to rejoin from index 3 onward, dropping last part
        ts = "_".join(parts[3:-1])  # drops 'T0' suffix
        return ts if ts else None
    except Exception:
        return None


def extract_features_from_xml(xml_file: Path, storm_id: int, window: str) -> dict:
    """Parse one MADIS XML file, return mean of each feature across all station records."""
    result = {"storm_id": storm_id, "window": window}
    try:
        root = ET.parse(xml_file).getroot()
        records = root.findall(".//record")
        var_data: dict[str, list[float]] = defaultdict(list)
        for rec in records:
            var = rec.attrib.get("var")
            val = rec.attrib.get("data_value")
            if var and val:
                try:
                    var_data[var].append(float(val))
                except ValueError:
                    pass
        for xml_var, col in _FEAT_MAP.items():
            vals = var_data.get(xml_var, [])
            result[col] = float(np.mean(vals)) if vals else np.nan
        result["num_stations"] = len(
            {r.attrib.get("shef_id") for r in records if r.attrib.get("shef_id")}
        )
    except Exception:
        for col in FEAT_COLS:
            result[col] = np.nan
        result["num_stations"] = 0
    return result


def parse_or_load(xml_files: list[Path], window_label: str, cache_path: Path) -> pd.DataFrame:
    """Load from cache CSV if it exists, otherwise parse XML files and cache."""
    if cache_path.exists():
        print(f"  Loading {window_label} from cache: {cache_path.name}")
        return pd.read_csv(cache_path)
    print(f"  Parsing {len(xml_files)} {window_label} XML files...")
    records = [
        extract_features_from_xml(f, _storm_id_from_path(f), window_label)
        for f in tqdm(xml_files, desc=f"Parsing {window_label}")
    ]
    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {window_label} cache → {cache_path.name}")
    return df


# ── Step 1: Parse XML files ───────────────────────────────────────────────────

def load_temporal_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0_files   = sorted(MADIS_DIR.glob("*_T0.xml"))
    t2h_files  = sorted(MADIS_DIR.glob("*_Tminus2h.xml"))
    t24h_files = sorted(MADIS_DIR.glob("*_Tminus24h.xml"))

    print(f"Found {len(t0_files)} T0, {len(t2h_files)} T-2h, {len(t24h_files)} T-24h XML files")

    df_t0_raw  = parse_or_load(t0_files,   "T0",       CACHE_T0)
    df_t2h_raw = parse_or_load(t2h_files,  "Tminus2h", CACHE_T2H)
    df_t24h_raw= parse_or_load(t24h_files, "Tminus24h",CACHE_T24H)

    return df_t0_raw, df_t2h_raw, df_t24h_raw


# ── Step 2: Aggregate to one row per storm ────────────────────────────────────

def aggregate_per_storm(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of weather features per storm_id, dropping rows with null storm_id."""
    df = df.dropna(subset=["storm_id"]).copy()
    df["storm_id"] = df["storm_id"].astype(int)
    available = [c for c in FEAT_COLS if c in df.columns]
    return df.groupby("storm_id")[available].mean().reset_index()


# ── Step 3: Compute anomaly features ─────────────────────────────────────────

def compute_anomalies(t0_agg: pd.DataFrame, t2h_agg: pd.DataFrame, t24h_agg: pd.DataFrame) -> pd.DataFrame:
    """Join T0/T-2h/T-24h and compute temporal anomaly features per storm."""
    df = (
        t0_agg.rename(columns={c: f"{c}_T0" for c in FEAT_COLS})
        .merge(t2h_agg.rename(columns={c: f"{c}_T2h" for c in FEAT_COLS}),
               on="storm_id", how="inner")
        .merge(t24h_agg.rename(columns={c: f"{c}_T24h" for c in FEAT_COLS}),
               on="storm_id", how="inner")
    )

    df["pressure_anomaly_24h"]   = df["pressure_T0"]          - df["pressure_T24h"]
    df["wind_anomaly_24h"]       = df["wind_speed_T0"]        - df["wind_speed_T24h"]
    df["instability_proxy_T2h"]  = df["temperature_T2h"]      - df["dewpoint_T2h"]
    df["instability_proxy_T0"]   = df["temperature_T0"]       - df["dewpoint_T0"]

    print(f"Anomaly features computed for {len(df)} storms with all 3 time windows")
    return df


# ── Step 4: Load catalog metadata ─────────────────────────────────────────────

def load_catalog_metadata() -> pd.DataFrame:
    """Return storm_id → {ef_number, category, type} from TorNet catalog."""
    catalog = pd.read_csv(CATALOG_PATH)

    def _get_event_id(url):
        if pd.isna(url):
            return None
        try:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            return query.get("id", [None])[0]
        except Exception:
            return None

    url_col = next((c for c in ["storm_event_url", "event_url", "url"] if c in catalog.columns), None)
    if url_col:
        catalog["storm_id"] = catalog[url_col].apply(_get_event_id)
    elif "event_id" in catalog.columns:
        catalog["storm_id"] = catalog["event_id"].astype(str)
    else:
        raise RuntimeError("Catalog has no storm_event_url or event_id column")

    catalog["storm_id"] = pd.to_numeric(catalog["storm_id"], errors="coerce")
    catalog = catalog.dropna(subset=["storm_id"])
    catalog["storm_id"] = catalog["storm_id"].astype(int)

    keep = ["storm_id"] + [c for c in ["ef_number", "category", "type"] if c in catalog.columns]
    return catalog[keep].drop_duplicates(subset=["storm_id"])


# ── Step 5: Build final per-(storm_id, timestamp) DataFrame ──────────────────

def build_final(df_t0_raw: pd.DataFrame, df_anom: pd.DataFrame, catalog_meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (storm_id, timestamp) with:
      - Tier-2 raw T0 features (pressure, wind_gust) at that scan time
      - Tier-1 storm-level anomaly features
      - Catalog metadata
    """
    df_t0 = df_t0_raw.dropna(subset=["storm_id"]).copy()
    df_t0["storm_id"] = df_t0["storm_id"].astype(int)

    # Keep only columns we need from T0
    t0_keep = ["storm_id"] + [c for c in TIER2_FEATURES if c in df_t0.columns]
    if "window" in df_t0.columns:
        t0_keep = ["storm_id", "window"] + [c for c in TIER2_FEATURES if c in df_t0.columns]

    # Reconstruct timestamp from file listing (T0 cache has no timestamp column from extract_features_from_xml)
    # Add timestamps by re-reading filenames
    t0_files = sorted(MADIS_DIR.glob("*_T0.xml"))
    ts_records = []
    for f in t0_files:
        sid = _storm_id_from_path(f)
        ts  = _timestamp_from_path(f)
        if sid is not None and ts is not None:
            ts_records.append({"storm_id": sid, "timestamp": ts})
    df_timestamps = pd.DataFrame(ts_records).drop_duplicates()

    # Merge timestamps onto raw T0 features (match by storm_id row order isn't safe — use file list)
    # Rebuild T0 with timestamps included
    print("  Rebuilding T0 with timestamps...")
    t0_with_ts = []
    for f in tqdm(t0_files, desc="T0 + timestamps"):
        sid = _storm_id_from_path(f)
        ts  = _timestamp_from_path(f)
        row = extract_features_from_xml(f, sid, "T0")
        row["timestamp"] = ts
        t0_with_ts.append(row)
    df_t0_ts = pd.DataFrame(t0_with_ts)
    df_t0_ts = df_t0_ts.dropna(subset=["storm_id"])
    df_t0_ts["storm_id"] = df_t0_ts["storm_id"].astype(int)

    tier2_cols = [c for c in TIER2_FEATURES if c in df_t0_ts.columns]
    df_final = (
        df_t0_ts[["storm_id", "timestamp"] + tier2_cols]
        .drop_duplicates(subset=["storm_id", "timestamp"])
        .copy()
    )

    # Join tier-1 anomaly features (storm-level)
    anom_cols = [c for c in TIER1_FEATURES if c in df_anom.columns]
    if anom_cols:
        df_anom_slim = df_anom[["storm_id"] + anom_cols].copy()
        df_final = df_final.merge(df_anom_slim, on="storm_id", how="left")
        n_matched = df_final[anom_cols[0]].notna().sum()
        print(f"  Anomaly features merged — {n_matched:,} / {len(df_final):,} records have Tier-1 data")

    # Join catalog metadata
    if not catalog_meta.empty:
        df_final = df_final.merge(catalog_meta, on="storm_id", how="left")

    df_final["has_madis_data"] = True
    return df_final


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"TORNET_ROOT = {DATA_ROOT}")
    print(f"MADIS dir   = {MADIS_DIR}\n")

    print("=== Step 1: Load temporal XML data ===")
    df_t0_raw, df_t2h_raw, df_t24h_raw = load_temporal_data()

    print("\n=== Step 2: Aggregate per storm ===")
    t0_agg  = aggregate_per_storm(df_t0_raw)
    t2h_agg = aggregate_per_storm(df_t2h_raw)
    t24h_agg= aggregate_per_storm(df_t24h_raw)
    print(f"  T0: {len(t0_agg)}, T-2h: {len(t2h_agg)}, T-24h: {len(t24h_agg)} storms")

    print("\n=== Step 3: Compute anomaly features ===")
    df_anom = compute_anomalies(t0_agg, t2h_agg, t24h_agg)

    print("\n=== Step 4: Load catalog metadata ===")
    catalog_meta = load_catalog_metadata()
    print(f"  Catalog storms: {len(catalog_meta)}")

    print("\n=== Step 5: Build final dataset ===")
    df_final = build_final(df_t0_raw, df_anom, catalog_meta)

    print(f"\nFinal shape: {df_final.shape}")
    print(f"Columns: {df_final.columns.tolist()}")
    missing = df_final.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"Missing values:\n{missing.to_string()}")

    print(f"\n=== Saving ===")
    df_final.to_csv(OUTPUT_CSV, index=False)
    df_final.to_pickle(OUTPUT_PKL)
    print(f"  CSV  → {OUTPUT_CSV}")
    print(f"  pkl  → {OUTPUT_PKL}")
    print(f"  {len(df_final):,} rows  |  {df_final['storm_id'].nunique():,} unique storms")


if __name__ == "__main__":
    main()

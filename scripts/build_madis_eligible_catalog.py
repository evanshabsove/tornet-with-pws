"""
Build a pre-filtered catalog containing only entries with valid MADIS coverage.

For each row in catalog.csv, checks whether the storm has a MADIS observation
(non-NaN pressure + wind_gust) within 15 minutes of the catalog row's start_time.
Saves the filtered catalog to $TORNET_ROOT/catalog_madis_eligible.csv.

Both MADIS and no-MADIS training runs should use this catalog so they train and
validate on identical storm populations — the only difference being whether
MADIS features are fed to the model.

Usage:
    python scripts/build_madis_eligible_catalog.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT    = Path(os.environ["TORNET_ROOT"])
CATALOG_PATH = DATA_ROOT / "catalog.csv"
MADIS_PATH   = DATA_ROOT / "madis_features_clean.csv"
OUTPUT_PATH  = DATA_ROOT / "catalog_madis_eligible.csv"

MATCH_WINDOW_SECONDS = 900  # 15 minutes


def main():
    print(f"TORNET_ROOT = {DATA_ROOT}\n")

    print("Loading catalog...")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["start_time", "end_time"])
    print(f"  Full catalog: {len(catalog):,} rows, {catalog['event_id'].nunique():,} unique storms")

    print("Loading MADIS features...")
    madis = pd.read_csv(MADIS_PATH)
    madis["timestamp"] = pd.to_datetime(madis["timestamp"], errors="coerce")
    madis_valid = madis[madis["pressure"].notna() & madis["wind_gust"].notna()].copy()
    madis_valid["storm_id"] = madis_valid["storm_id"].astype(str)
    print(f"  MADIS rows with valid pressure + wind_gust: {len(madis_valid):,}")

    # Group MADIS timestamps by storm_id for fast per-storm lookup
    madis_by_storm = madis_valid.groupby("storm_id")["timestamp"].apply(list).to_dict()
    madis_storm_ids = set(madis_by_storm.keys())

    print(f"\nChecking {len(catalog):,} catalog rows for MADIS coverage...")
    eligible_mask = np.zeros(len(catalog), dtype=bool)

    for i, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Filtering"):
        storm_id = str(row["event_id"])
        if storm_id not in madis_storm_ids:
            continue
        start_time = row["start_time"]
        if pd.isna(start_time):
            continue
        madis_timestamps = madis_by_storm[storm_id]
        diffs = [abs((ts - start_time).total_seconds()) for ts in madis_timestamps]
        if min(diffs) <= MATCH_WINDOW_SECONDS:
            eligible_mask[i] = True

    catalog_eligible = catalog[eligible_mask].copy()

    print(f"\nEligible rows: {len(catalog_eligible):,} / {len(catalog):,} "
          f"({100 * len(catalog_eligible) / len(catalog):.1f}%)")
    print(f"Unique storms: {catalog_eligible['event_id'].nunique():,}")

    print("\nBreakdown by year:")
    year_counts = catalog_eligible.groupby(catalog_eligible["start_time"].dt.year).size()
    print(year_counts.to_string())

    print("\nBreakdown by category:")
    if "category" in catalog_eligible.columns:
        print(catalog_eligible["category"].value_counts().to_string())

    catalog_eligible.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

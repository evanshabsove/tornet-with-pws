"""
Build a pre-filtered catalog containing only entries with valid MADIS coverage.

For each row in catalog.csv, checks whether the storm has any MADIS observation
with non-NaN pressure + wind_gust (no temporal cutoff — MADIS was downloaded for
±60 min around each radar frame, so the closest observation is always usable).
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


def main():
    print(f"TORNET_ROOT = {DATA_ROOT}\n")

    print("Loading catalog...")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["start_time", "end_time"])
    print(f"  Full catalog: {len(catalog):,} rows, {catalog['event_id'].nunique():,} unique storms")

    print("Loading MADIS features...")
    madis = pd.read_csv(MADIS_PATH)
    madis_valid = madis[madis["pressure"].notna() & madis["wind_gust"].notna()].copy()
    print(f"  MADIS rows with valid pressure + wind_gust: {len(madis_valid):,}")

    # Any storm with at least one valid (pressure, wind_gust) observation is eligible.
    # No temporal cutoff — loader.py picks the closest observation within the ±60 min download window.
    eligible_storm_ids = set(madis_valid["storm_id"].astype(str))

    print(f"\nChecking {len(catalog):,} catalog rows for MADIS coverage...")
    eligible_mask = catalog["event_id"].astype(str).isin(eligible_storm_ids).values

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

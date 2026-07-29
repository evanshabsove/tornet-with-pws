"""
Builds a flat text cache of storm_ids (event_ids) with valid MADIS coverage,
one per line, for the Flask API's GET /storms/madis route to read at startup
instead of recomputing the catalog/MADIS join on every run.

Reuses $TORNET_ROOT/catalog_madis_eligible.csv (see build_madis_eligible_catalog.py)
as the source of truth, since that's the exact same eligibility criterion
tornet/data/loader.py's read_file() enforces (non-NaN pressure + wind_gust).

Usage:
    python scripts/build_madis_storm_ids_cache.py
"""
import os
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ["TORNET_ROOT"])
ELIGIBLE_CATALOG_PATH = DATA_ROOT / "catalog_madis_eligible.csv"
OUTPUT_PATH = DATA_ROOT / "madis_eligible_storm_ids.txt"


def main():
    catalog_eligible = pd.read_csv(ELIGIBLE_CATALOG_PATH)
    storm_ids = sorted(catalog_eligible["event_id"].unique().tolist())

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(str(s) for s in storm_ids))

    print(f"Wrote {len(storm_ids):,} storm_ids to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate a list of MADIS download URLs from TorNet catalog.

This script reads the TorNet catalog and generates a CSV file containing
all the MADIS URLs that need to be downloaded. This file can be transferred
to another computer for downloading via VPN.

Usage:
    python generate_madis_urls.py
    python generate_madis_urls.py --years 2013 2014
    python generate_madis_urls.py --output madis_urls_2013_2014.csv
"""

import os
import sys
import argparse
import math
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm


def get_bounding_box(lat: float, lon: float, distance_km: float = 20) -> Tuple[float, float, float, float]:
    """
    Returns the bounding box corners (latll, lonll, latur, lonur) for a box centered at (lat, lon)
    with the given distance_km as the box width/height.
    
    Args:
        lat: Center latitude
        lon: Center longitude
        distance_km: Size of bounding box (width and height) in kilometers
    
    Returns:
        Tuple of (latll, lonll, latur, lonur)
    """
    # 1 degree latitude is ~111 km
    delta_lat = (distance_km / 2) / 111.0

    # 1 degree longitude is ~111 km * cos(latitude)
    delta_lon = (distance_km / 2) / (111.0 * math.cos(math.radians(lat)))

    latll = lat - delta_lat
    lonll = lon - delta_lon
    latur = lat + delta_lat
    lonur = lon + delta_lon

    return latll, lonll, latur, lonur


def convert_to_timestamp(time) -> str:
    """
    Converts a datetime object or numpy.datetime64 to a string in the format 'YYYYMMDD_HHMM'.
    
    Args:
        time: datetime object, numpy.datetime64, or pandas Timestamp
    
    Returns:
        Formatted timestamp string
    """
    if isinstance(time, np.datetime64):
        # Convert numpy.datetime64 to python datetime
        time = pd.to_datetime(str(time)).to_pydatetime()
    elif hasattr(time, 'to_pydatetime'):
        time = time.to_pydatetime()
    return time.strftime('%Y%m%d_%H%M')


def get_id_from_storm_event_url(storm_event_url: str) -> Optional[str]:
    """
    Extracts the 'id' parameter value from the storm_event_url.
    
    Args:
        storm_event_url: URL containing storm event ID
    
    Returns:
        Storm ID as string, or None if not found
    """
    if storm_event_url:
        parsed = urllib.parse.urlparse(storm_event_url)
        query = urllib.parse.parse_qs(parsed.query)
        return query.get('id', [None])[0]
    return None


def set_madis_url(latll: float, lonll: float, latur: float, lonur: float, time: str) -> str:
    """
    Constructs the MADIS API URL for downloading data.
    
    Args:
        latll: Lower-left latitude
        lonll: Lower-left longitude
        latur: Upper-right latitude
        lonur: Upper-right longitude
        time: Timestamp in YYYYMMDD_HHMM format
    
    Returns:
        Complete MADIS API URL
    """
    return (f"https://madis-data.ncep.noaa.gov/madisPublic1/cgi-bin/madisXmlPublicDir?"
            f"rdr=&time={time}&minbck=-59&minfwd=0&recwin=3&dfltrsel=1&state=&"
            f"latll={latll}&lonll={lonll}&latur={latur}&lonur={lonur}&"
            f"stanam=&stasel=0&pvdrsel=1&varsel=2&qctype=0&qcsel=1&xml=1&csvmiss=0&"
            f"pvd=APRSWXNET")


def generate_url_list(catalog: pd.DataFrame, data_root: Path, 
                      years: Optional[List[int]] = None,
                      distance_km: float = 20,
                      skip_existing: bool = True,
                      existing_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Generate list of MADIS URLs from TorNet catalog.
    
    Args:
        catalog: TorNet catalog DataFrame
        data_root: Root directory containing TorNet data
        years: Optional list of years to filter by
        distance_km: Bounding box size in kilometers
        skip_existing: Whether to skip URLs for files that already exist
        existing_dir: Directory to check for existing files
    
    Returns:
        DataFrame with columns: storm_id, start_time, url, output_filename
    """
    # Filter by years if specified
    if years:
        catalog = catalog[catalog.start_time.dt.year.isin(years)].copy()
    
    print(f"Generating URLs from {len(catalog)} catalog entries...")
    
    url_data = []
    failed_files = []
    skipped_existing = 0
    
    for idx, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Processing storms"):
        try:
            file_path = data_root / row['filename']
            
            if not file_path.exists():
                failed_files.append((row['filename'], "File not found"))
                continue
            
            # Open NetCDF file to get attributes
            ds = xr.open_dataset(file_path, engine='netcdf4')
            
            # Extract storm ID from URL
            storm_id = get_id_from_storm_event_url(ds.attrs.get('storm_event_url', ''))
            
            if not storm_id:
                failed_files.append((row['filename'], "No storm ID found"))
                ds.close()
                continue
            
            # Get location and time
            site_lat = ds.attrs.get('site_lat')
            site_lon = ds.attrs.get('site_lon')
            
            if site_lat is None or site_lon is None:
                failed_files.append((row['filename'], "Missing lat/lon"))
                ds.close()
                continue
            
            # Create output filename
            start_time = row['start_time']
            output_filename = f"madis_data_{storm_id}_{start_time}.xml"
            
            # Check if file already exists
            if skip_existing and existing_dir:
                output_file = existing_dir / output_filename
                if output_file.exists():
                    skipped_existing += 1
                    ds.close()
                    continue
            
            # Get bounding box
            latll, lonll, latur, lonur = get_bounding_box(site_lat, site_lon, distance_km)
            
            # Convert time to MADIS format
            timestamp = convert_to_timestamp(start_time)
            
            # Construct URL
            url = set_madis_url(latll, lonll, latur, lonur, timestamp)
            
            # Add to list
            url_data.append({
                'storm_id': storm_id,
                'start_time': str(start_time),
                'url': url,
                'output_filename': output_filename
            })
            
            ds.close()
            
        except Exception as e:
            failed_files.append((row['filename'], str(e)))
            continue
    
    # Create DataFrame
    url_df = pd.DataFrame(url_data)
    
    print(f"\n{'='*60}")
    print(f"Successfully generated {len(url_df)} URLs")
    if skip_existing and existing_dir:
        print(f"Skipped {skipped_existing} files that already exist")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
        if len(failed_files) <= 10:
            for fname, error in failed_files:
                print(f"  - {fname}: {error}")
    print(f"{'='*60}\n")
    
    return url_df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate MADIS download URLs from TorNet catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate URLs for all years
  python generate_madis_urls.py
  
  # Generate URLs for specific years
  python generate_madis_urls.py --years 2013 2014 2015
  
  # Specify custom output file
  python generate_madis_urls.py --years 2020 2021 --output madis_urls_2020_2021.csv
  
  # Include URLs for files that already exist
  python generate_madis_urls.py --no-skip-existing
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='tornet_data',
        help='Path to TorNet data directory (default: tornet_data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='madis_download_urls.csv',
        help='Output CSV file path (default: madis_download_urls.csv)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Years to process (default: all years in catalog)'
    )
    
    parser.add_argument(
        '--distance-km',
        type=float,
        default=20,
        help='Bounding box size in kilometers (default: 20)'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Include URLs for files that already exist (default: skip existing)'
    )
    
    parser.add_argument(
        '--existing-dir',
        type=str,
        default=None,
        help='Directory to check for existing files (default: DATA_ROOT/madis_data)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        sys.exit(1)
    
    catalog_path = data_root / 'catalog.csv'
    if not catalog_path.exists():
        print(f"Error: Catalog file not found: {catalog_path}")
        sys.exit(1)
    
    # Determine existing files directory
    if args.existing_dir:
        existing_dir = Path(args.existing_dir)
    else:
        existing_dir = data_root / 'madis_data'
    
    skip_existing = not args.no_skip_existing
    
    print(f"{'='*60}")
    print("MADIS URL Generation Configuration")
    print(f"{'='*60}")
    print(f"Data root:       {data_root}")
    print(f"Output file:     {args.output}")
    print(f"Years:           {args.years if args.years else 'All'}")
    print(f"Bounding box:    {args.distance_km} km")
    print(f"Skip existing:   {skip_existing}")
    if skip_existing:
        print(f"Existing dir:    {existing_dir}")
    print(f"{'='*60}\n")
    
    # Load catalog
    print("Loading TorNet catalog...")
    try:
        catalog = pd.read_csv(catalog_path, parse_dates=['start_time', 'end_time'])
        print(f"Loaded {len(catalog)} entries from catalog\n")
    except Exception as e:
        print(f"Error loading catalog: {e}")
        sys.exit(1)
    
    # Generate URL list
    url_df = generate_url_list(
        catalog, 
        data_root, 
        args.years,
        args.distance_km,
        skip_existing,
        existing_dir if skip_existing else None
    )
    
    if len(url_df) == 0:
        print("No URLs to generate. Exiting.")
        sys.exit(0)
    
    # Save to CSV
    output_path = Path(args.output)
    url_df.to_csv(output_path, index=False)
    
    print(f"{'='*60}")
    print(f"URLs saved to: {output_path}")
    print(f"Total URLs: {len(url_df)}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print(f"1. Transfer {output_path} to your personal computer")
    print(f"2. Run: python download_madis_from_urls.py {output_path.name}")
    print()


if __name__ == '__main__':
    main()

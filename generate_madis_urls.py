#!/usr/bin/env python3
"""
Generate a list of MADIS download URLs from TorNet catalog.

This script reads the TorNet catalog and generates a CSV file containing
all the MADIS URLs that need to be downloaded. This file can be transferred
to another computer for downloading via VPN.

Supports multiple temporal windows:
- T0: At-storm time (during radar scan)
- T-2h: Pre-storm (2 hours before radar scan) - atmospheric setup
- T-24h: Control (24 hours before) - baseline/normal conditions

Usage:
    python generate_madis_urls.py
    python generate_madis_urls.py --years 2013 2014
    python generate_madis_urls.py --output madis_urls_2013_2014.csv
    python generate_madis_urls.py --time-windows T0 Tminus2h Tminus24h
"""

import os
import sys
import argparse
import math
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import timedelta

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


def get_time_offset(base_time, window: str):
    """
    Apply time offset based on temporal window.
    
    Args:
        base_time: Base datetime (storm time)
        window: Time window identifier ('T0', 'Tminus2h', or 'Tminus24h')
    
    Returns:
        Adjusted datetime
    """
    # Convert to pandas Timestamp for easier manipulation
    if isinstance(base_time, np.datetime64):
        base_time = pd.to_datetime(str(base_time))
    elif not isinstance(base_time, pd.Timestamp):
        base_time = pd.Timestamp(base_time)
    
    if window == 'T0':
        return base_time
    elif window == 'Tminus2h':
        return base_time - timedelta(hours=2)
    elif window == 'Tminus24h':
        return base_time - timedelta(hours=24)
    else:
        raise ValueError(f"Unknown time window: {window}")


def get_window_description(window: str) -> str:
    """Get human-readable description of time window."""
    descriptions = {
        'T0': 'At-storm (during radar scan)',
        'Tminus2h': 'Pre-storm (2 hours before)',
        'Tminus24h': 'Control (24 hours before)'
    }
    return descriptions.get(window, window)


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
                      existing_dir: Optional[Path] = None,
                      time_windows: List[str] = ['T0']) -> pd.DataFrame:
    """
    Generate list of MADIS URLs from TorNet catalog.
    
    Args:
        catalog: TorNet catalog DataFrame
        data_root: Root directory containing TorNet data
        years: Optional list of years to filter by
        distance_km: Bounding box size in kilometers
        skip_existing: Whether to skip URLs for files that already exist
        existing_dir: Directory to check for existing files
        time_windows: List of time windows to generate ('T0', 'Tminus2h', 'Tminus24h')
    
    Returns:
        DataFrame with columns: storm_id, start_time, time_window, url, output_filename
    """
    # Filter by years if specified
    if years:
        catalog = catalog[catalog.start_time.dt.year.isin(years)].copy()
    
    print(f"Generating URLs from {len(catalog)} catalog entries...")
    print(f"Time windows: {', '.join([get_window_description(w) for w in time_windows])}")
    
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
            
            # Get base time
            base_time = row['start_time']
            
            # Generate URLs for each time window
            for window in time_windows:
                # Apply time offset
                adjusted_time = get_time_offset(base_time, window)
                
                # Create output filename with time window suffix
                output_filename = f"madis_data_{storm_id}_{base_time}_{window}.xml"
                
                # Check if file already exists
                if skip_existing and existing_dir:
                    output_file = existing_dir / output_filename
                    if output_file.exists():
                        skipped_existing += 1
                        continue
                
                # Get bounding box
                latll, lonll, latur, lonur = get_bounding_box(site_lat, site_lon, distance_km)
                
                # Convert time to MADIS format
                timestamp = convert_to_timestamp(adjusted_time)
                
                # Construct URL
                url = set_madis_url(latll, lonll, latur, lonur, timestamp)
                
                # Add to list
                url_data.append({
                    'storm_id': storm_id,
                    'base_time': str(base_time),
                    'time_window': window,
                    'window_description': get_window_description(window),
                    'adjusted_time': str(adjusted_time),
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
    print(f"  - {len(url_df) // len(time_windows)} storms")
    print(f"  - {len(time_windows)} time windows per storm")
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
        description="Generate MADIS download URLs from TorNet catalog with multiple temporal windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate URLs for all years, all time windows
  python generate_madis_urls.py
  
  # Generate URLs for specific years
  python generate_madis_urls.py --years 2013 2014 2015
  
  # Generate only pre-storm and control windows (not at-storm)
  python generate_madis_urls.py --time-windows Tminus2h Tminus24h
  
  # Specify custom output file
  python generate_madis_urls.py --years 2020 2021 --output madis_urls_2020_2021.csv
  
  # Include URLs for files that already exist
  python generate_madis_urls.py --no-skip-existing

Time Windows:
  T0        - At-storm time (during radar scan) - in-storm conditions
  Tminus2h  - Pre-storm (2 hours before) - atmospheric setup
  Tminus24h - Control (24 hours before) - baseline/normal conditions
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
        '--time-windows',
        type=str,
        nargs='+',
        default=['T0', 'Tminus2h', 'Tminus24h'],
        choices=['T0', 'Tminus2h', 'Tminus24h'],
        help='Time windows to generate URLs for (default: all three)'
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
    print(f"Time windows:    {args.time_windows}")
    for window in args.time_windows:
        print(f"  - {window}: {get_window_description(window)}")
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
        existing_dir if skip_existing else None,
        args.time_windows
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
    print(f"  - Storms: {url_df['storm_id'].nunique()}")
    print(f"  - Time windows: {url_df['time_window'].nunique()}")
    for window in args.time_windows:
        count = (url_df['time_window'] == window).sum()
        print(f"    • {window}: {count} URLs")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print(f"1. Transfer {output_path} to your personal computer")
    print(f"2. Run: python download_madis_from_urls.py {output_path.name}")
    print(f"\nNote: Generated {len(url_df)} total URLs ({len(args.time_windows)} per storm)")
    print(f"      This enables temporal analysis: pre-storm setup, at-storm conditions, and baseline")
    print()


if __name__ == '__main__':
    main()

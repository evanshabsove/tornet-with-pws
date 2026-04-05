#!/usr/bin/env python3
"""
Download MADIS Personal Weather Station data for TorNet storm events.

This script:
1. Reads the TorNet catalog to identify storm events
2. Downloads MADIS APRSWXNET data for each storm's time/location in parallel
3. Saves XML files with weather station observations
4. Provides progress tracking and error reporting

Usage:
    python download_madis_data.py
    python download_madis_data.py --years 2013 2014 --workers 20
    python download_madis_data.py --data-root /path/to/tornet_data
"""

import os
import sys
import argparse
import math
import time
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import xarray as xr
import requests
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
        latll: latitude of lower left (SW) corner
        lonll: longitude of lower left (SW) corner
        latur: latitude of upper right (NE) corner
        lonur: longitude of upper right (NE) corner
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


def download_madis_for_storm(storm_info: Tuple, output_dir: Path, distance_km: float = 20, 
                              max_retries: int = 3, verbose: bool = False) -> Tuple[str, bool, Optional[str], int]:
    """
    Download MADIS data for a single storm event.
    
    Args:
        storm_info: Tuple of (storm_id, start_time, site_lat, site_lon, filename)
        output_dir: Directory to save XML files
        distance_km: Bounding box size in kilometers
        max_retries: Number of retry attempts for failed downloads
        verbose: Whether to print detailed progress messages
    
    Returns:
        Tuple of (storm_id, success, error_message, num_records)
    """
    storm_id, start_time, site_lat, site_lon, filename = storm_info
    
    # Create output filename
    output_file = output_dir / f"madis_data_{storm_id}_{start_time}.xml"
    
    # Skip if file already exists
    if output_file.exists():
        if verbose:
            print(f"File already exists: {output_file.name}, skipping")
        return (storm_id, True, "Already exists", 0)
    
    # Get bounding box
    latll, lonll, latur, lonur = get_bounding_box(site_lat, site_lon, distance_km)
    
    # Convert time to MADIS format
    timestamp = convert_to_timestamp(start_time)
    
    # Construct URL
    url = set_madis_url(latll, lonll, latur, lonur, timestamp)
    
    # Attempt download with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse XML to check if there's actual data
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            records = root.findall('.//record')
            num_records = len(records)
            
            # Save the file
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            if verbose:
                if num_records > 0:
                    print(f"✅ {storm_id}: Found {num_records} records")
                else:
                    print(f"⚠️  {storm_id}: No data records found")
            
            return (storm_id, True, None, num_records)
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Wait before retrying (exponential backoff)
                time.sleep(2 ** attempt)
                continue
            else:
                return (storm_id, False, f"HTTP error: {str(e)}", 0)
        except Exception as e:
            return (storm_id, False, f"Error: {str(e)}", 0)
    
    return (storm_id, False, "Max retries exceeded", 0)


def prepare_storm_list(catalog: pd.DataFrame, data_root: Path, 
                       years: Optional[List[int]] = None) -> List[Tuple]:
    """
    Extract storm metadata from catalog and NetCDF files.
    
    Args:
        catalog: TorNet catalog DataFrame
        data_root: Root directory containing TorNet data
        years: Optional list of years to filter by
    
    Returns:
        List of tuples containing storm information
    """
    # Filter by years if specified
    if years:
        catalog = catalog[catalog.start_time.dt.year.isin(years)].copy()
    
    print(f"Preparing storm list from {len(catalog)} catalog entries...")
    
    storm_list = []
    failed_files = []
    
    for idx, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Reading NetCDF metadata"):
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
            
            # Add to storm list
            storm_list.append((
                storm_id,
                row['start_time'],
                site_lat,
                site_lon,
                row['filename']
            ))
            
            ds.close()
            
        except Exception as e:
            failed_files.append((row['filename'], str(e)))
            continue
    
    print(f"\n{'='*60}")
    print(f"Successfully prepared {len(storm_list)} storms")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
        if len(failed_files) <= 10:
            for fname, error in failed_files:
                print(f"  - {fname}: {error}")
    print(f"{'='*60}\n")
    
    return storm_list


def parallel_download(storm_list: List[Tuple], output_dir: Path, max_workers: int = 20,
                      distance_km: float = 20, max_retries: int = 3) -> Tuple[int, int, List]:
    """
    Download MADIS data for multiple storms in parallel.
    
    Args:
        storm_list: List of storm information tuples
        output_dir: Directory to save XML files
        max_workers: Number of parallel workers
        distance_km: Bounding box size in kilometers
        max_retries: Number of retry attempts for failed downloads
    
    Returns:
        Tuple of (successful_count, failed_count, failed_storms)
    """
    print(f"Downloading MADIS data for {len(storm_list)} storms using {max_workers} workers...")
    print(f"Output directory: {output_dir}")
    print(f"Bounding box size: {distance_km} km")
    print(f"Max retries per storm: {max_retries}\n")
    
    successful = 0
    failed = 0
    already_exist = 0
    no_data = 0
    failed_storms = []
    total_records = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_storm = {
            executor.submit(download_madis_for_storm, storm, output_dir, 
                          distance_km, max_retries, False): storm 
            for storm in storm_list
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_storm), total=len(storm_list), 
                          desc="Downloading", unit="storm"):
            storm = future_to_storm[future]
            try:
                storm_id, success, error, num_records = future.result()
                
                if success:
                    if error == "Already exists":
                        already_exist += 1
                    else:
                        successful += 1
                        total_records += num_records
                        if num_records == 0:
                            no_data += 1
                else:
                    failed += 1
                    failed_storms.append((storm_id, error))
                    
            except Exception as e:
                failed += 1
                failed_storms.append((storm[0], f"Exception: {str(e)}"))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    print(f"Total storms:        {len(storm_list)}")
    print(f"Already existed:     {already_exist}")
    print(f"Successfully downloaded: {successful}")
    print(f"  - With data:       {successful - no_data}")
    print(f"  - Without data:    {no_data}")
    print(f"Failed:              {failed}")
    print(f"Total weather records: {total_records}")
    print(f"{'='*60}\n")
    
    if failed_storms:
        print("Failed storms:")
        for storm_id, error in failed_storms[:20]:  # Show first 20
            print(f"  - Storm {storm_id}: {error}")
        if len(failed_storms) > 20:
            print(f"  ... and {len(failed_storms) - 20} more")
        print()
    
    return successful, failed, failed_storms


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download MADIS Personal Weather Station data for TorNet storm events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for all years with default settings
  python download_madis_data.py
  
  # Download specific years with more workers
  python download_madis_data.py --years 2013 2014 2015 --workers 40
  
  # Custom directories and bounding box size
  python download_madis_data.py --data-root /path/to/tornet_data \\
                                --output-dir /path/to/output \\
                                --distance-km 30
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='tornet_data',
        help='Path to TorNet data directory (default: tornet_data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save MADIS XML files (default: DATA_ROOT/madis_data)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Years to process (default: all years in catalog)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=20,
        help='Number of parallel download workers (default: 20)'
    )
    
    parser.add_argument(
        '--retry',
        type=int,
        default=3,
        help='Number of retry attempts for failed downloads (default: 3)'
    )
    
    parser.add_argument(
        '--distance-km',
        type=float,
        default=20,
        help='Bounding box size in kilometers (default: 20)'
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
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / 'madis_data'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print("MADIS Data Download Configuration")
    print(f"{'='*60}")
    print(f"Data root:       {data_root}")
    print(f"Output dir:      {output_dir}")
    print(f"Years:           {args.years if args.years else 'All'}")
    print(f"Workers:         {args.workers}")
    print(f"Max retries:     {args.retry}")
    print(f"Bounding box:    {args.distance_km} km")
    print(f"{'='*60}\n")
    
    # Load catalog
    print("Loading TorNet catalog...")
    try:
        catalog = pd.read_csv(catalog_path, parse_dates=['start_time', 'end_time'])
        print(f"Loaded {len(catalog)} entries from catalog\n")
    except Exception as e:
        print(f"Error loading catalog: {e}")
        sys.exit(1)
    
    # Prepare storm list
    storm_list = prepare_storm_list(catalog, data_root, args.years)
    
    if not storm_list:
        print("No storms to process. Exiting.")
        sys.exit(0)
    
    # Download data
    start_time = time.time()
    successful, failed, failed_storms = parallel_download(
        storm_list,
        output_dir,
        max_workers=args.workers,
        distance_km=args.distance_km,
        max_retries=args.retry
    )
    elapsed_time = time.time() - start_time
    
    # Final summary
    print(f"{'='*60}")
    print("Final Summary:")
    print(f"{'='*60}")
    print(f"Total time:      {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Avg per storm:   {elapsed_time/len(storm_list):.2f} seconds")
    print(f"Success rate:    {successful/(successful+failed)*100:.1f}%")
    print(f"{'='*60}\n")
    
    if failed_storms:
        # Save failed storms to file
        failed_file = output_dir / 'failed_downloads.txt'
        with open(failed_file, 'w') as f:
            f.write("Storm ID\tError\n")
            for storm_id, error in failed_storms:
                f.write(f"{storm_id}\t{error}\n")
        print(f"Failed storm list saved to: {failed_file}\n")
    
    print("Download complete!")


if __name__ == '__main__':
    main()

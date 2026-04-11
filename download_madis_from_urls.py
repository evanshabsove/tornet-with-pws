#!/usr/bin/env python3
"""
Download MADIS data from a pre-generated URL list.

This script downloads MADIS weather station data using URLs from a CSV file.
It's designed to run on a different computer (with VPN) than where the URLs
were generated.

Usage:
    python download_madis_from_urls.py madis_download_urls.csv
    python download_madis_from_urls.py madis_urls_2013_2014.csv --workers 10
    python download_madis_from_urls.py urls.csv --output-dir ./madis_data
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional

import pandas as pd
import requests
from tqdm import tqdm


def download_madis_from_url(url: str, output_file: Path, storm_id: str,
                            max_retries: int = 3, verbose: bool = False) -> Tuple[str, bool, Optional[str], int]:
    """
    Download MADIS data from a single URL.
    
    Args:
        url: MADIS API URL to download from
        output_file: Path to save XML file
        storm_id: Storm ID for tracking
        max_retries: Number of retry attempts for failed downloads
        verbose: Whether to print detailed progress messages
    
    Returns:
        Tuple of (storm_id, success, error_message, num_records)
    """
    # Skip if file already exists
    if output_file.exists():
        if verbose:
            print(f"File already exists: {output_file.name}, skipping")
        return (storm_id, True, "Already exists", 0)
    
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


def parallel_download_from_urls(url_df: pd.DataFrame, output_dir: Path, 
                                max_workers: int = 20, max_retries: int = 3) -> Tuple[int, int, list]:
    """
    Download MADIS data from multiple URLs in parallel.
    
    Args:
        url_df: DataFrame with columns: storm_id, start_time, url, output_filename
        output_dir: Directory to save XML files
        max_workers: Number of parallel workers
        max_retries: Number of retry attempts for failed downloads
    
    Returns:
        Tuple of (successful_count, failed_count, failed_storms)
    """
    print(f"Downloading MADIS data for {len(url_df)} storms using {max_workers} workers...")
    print(f"Output directory: {output_dir}")
    print(f"Max retries per storm: {max_retries}\n")
    
    successful = 0
    failed = 0
    already_exist = 0
    no_data = 0
    failed_storms = []
    total_records = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_storm = {}
        for idx, row in url_df.iterrows():
            output_file = output_dir / row['output_filename']
            future = executor.submit(
                download_madis_from_url,
                row['url'],
                output_file,
                row['storm_id'],
                max_retries,
                False
            )
            future_to_storm[future] = row['storm_id']
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_storm), total=len(url_df), 
                          desc="Downloading", unit="storm"):
            storm_id = future_to_storm[future]
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
                failed_storms.append((storm_id, f"Exception: {str(e)}"))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    print(f"Total storms:        {len(url_df)}")
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
        description="Download MADIS data from pre-generated URL list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with default settings
  python download_madis_from_urls.py madis_download_urls.csv
  
  # Use more workers for faster downloads
  python download_madis_from_urls.py urls.csv --workers 40
  
  # Custom output directory
  python download_madis_from_urls.py urls.csv --output-dir /path/to/output
  
  # Reduce workers if you're getting rate-limited
  python download_madis_from_urls.py urls.csv --workers 5 --retry 5
        """
    )
    
    parser.add_argument(
        'url_file',
        type=str,
        help='CSV file containing URLs to download (generated by generate_madis_urls.py)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='madis_data',
        help='Directory to save MADIS XML files (default: madis_data)'
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
    
    args = parser.parse_args()
    
    # Check if URL file exists
    url_file = Path(args.url_file)
    if not url_file.exists():
        print(f"Error: URL file not found: {url_file}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print("MADIS Download Configuration")
    print(f"{'='*60}")
    print(f"URL file:        {url_file}")
    print(f"Output dir:      {output_dir}")
    print(f"Workers:         {args.workers}")
    print(f"Max retries:     {args.retry}")
    print(f"{'='*60}\n")
    
    # Load URL file
    print("Loading URL list...")
    try:
        url_df = pd.read_csv(url_file)
        # Required columns (only those actually used by the download function)
        required_cols = ['storm_id', 'url', 'output_filename']
        if not all(col in url_df.columns for col in required_cols):
            print(f"Error: URL file must contain columns: {required_cols}")
            print(f"Found columns: {list(url_df.columns)}")
            sys.exit(1)
        print(f"Loaded {len(url_df)} URLs from {url_file}\n")
    except Exception as e:
        print(f"Error loading URL file: {e}")
        sys.exit(1)
    
    # Download data
    start_time = time.time()
    successful, failed, failed_storms = parallel_download_from_urls(
        url_df,
        output_dir,
        max_workers=args.workers,
        max_retries=args.retry
    )
    elapsed_time = time.time() - start_time
    
    # Final summary
    print(f"{'='*60}")
    print("Final Summary:")
    print(f"{'='*60}")
    print(f"Total time:      {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Avg per storm:   {elapsed_time/len(url_df):.2f} seconds")
    if successful + failed > 0:
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

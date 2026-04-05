#!/usr/bin/env python3
"""
Script to download and organize TorNet dataset files from Zenodo.

This script:
1. Downloads TorNet data for years 2015-2022 using zenodo_get
2. Extracts the compressed files
3. Moves train/test data to the appropriate tornet_data directory
4. Cleans up temporary files

Usage:
    python download_tornet_data.py
"""

import os
import subprocess
import tarfile
import shutil
import argparse
from pathlib import Path


# Zenodo DOIs for each year
TORNET_URLS = {
    2015: "https://doi.org/10.5281/zenodo.12655151",
    2016: "https://doi.org/10.5281/zenodo.12655179",
    2017: "https://doi.org/10.5281/zenodo.12655183",
    2018: "https://doi.org/10.5281/zenodo.12655187",
    2019: "https://doi.org/10.5281/zenodo.12655716",
    2020: "https://doi.org/10.5281/zenodo.12655717",
    2021: "https://doi.org/10.5281/zenodo.12655718",
    2022: "https://doi.org/10.5281/zenodo.12655719",
}


def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def download_zenodo(doi_url, download_dir):
    """Download data from Zenodo using zenodo_get."""
    print(f"\n{'='*60}")
    print(f"Downloading from {doi_url}")
    print(f"{'='*60}")
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Run zenodo_get
    try:
        run_command(["zenodo_get", doi_url], cwd=download_dir)
    except RuntimeError as e:
        print(f"Failed to download: {e}")
        return None
    
    return download_dir


def find_tar_file(directory, year):
    """Find the tar.gz file for the given year in the directory."""
    tar_files = list(Path(directory).glob(f"*{year}*.tar.gz"))
    if not tar_files:
        tar_files = list(Path(directory).glob("*.tar.gz"))
    
    if not tar_files:
        print(f"Warning: No .tar.gz file found in {directory}")
        return None
    
    if len(tar_files) > 1:
        print(f"Warning: Multiple .tar.gz files found, using {tar_files[0]}")
    
    return tar_files[0]


def extract_tar_file(tar_path, extract_dir):
    """Extract a tar.gz file."""
    print(f"Extracting {tar_path}...")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    
    print(f"Extracted to {extract_dir}")


def move_data_to_tornet(extracted_dir, year, tornet_data_dir):
    """Move train and test folders to the tornet_data directory."""
    print(f"Moving data for year {year} to {tornet_data_dir}...")
    
    # Find the extracted folder (might be named tornet_YEAR or just have train/test)
    extracted_path = Path(extracted_dir)
    
    # Look for train and test directories
    possible_roots = [
        extracted_path,
        extracted_path / f"tornet_{year}",
        extracted_path / f"tornet-{year}",
    ]
    
    train_source = None
    test_source = None
    
    for root in possible_roots:
        if (root / "train" / str(year)).exists():
            train_source = root / "train" / str(year)
        if (root / "test" / str(year)).exists():
            test_source = root / "test" / str(year)
        
        if train_source and test_source:
            break
    
    if not train_source and not test_source:
        # Try to find any train/test folders
        for root in possible_roots:
            if (root / "train").exists():
                train_source = root / "train"
            if (root / "test").exists():
                test_source = root / "test"
    
    # Move train data
    if train_source and train_source.exists():
        train_dest = Path(tornet_data_dir) / "train" / str(year)
        print(f"  Moving {train_source} -> {train_dest}")
        if train_dest.exists():
            print(f"  Warning: {train_dest} already exists, removing it first")
            shutil.rmtree(train_dest)
        shutil.move(str(train_source), str(train_dest))
    else:
        print(f"  Warning: Train data not found for year {year}")
    
    # Move test data
    if test_source and test_source.exists():
        test_dest = Path(tornet_data_dir) / "test" / str(year)
        print(f"  Moving {test_source} -> {test_dest}")
        if test_dest.exists():
            print(f"  Warning: {test_dest} already exists, removing it first")
            shutil.rmtree(test_dest)
        shutil.move(str(test_source), str(test_dest))
    else:
        print(f"  Warning: Test data not found for year {year}")


def cleanup(download_dir):
    """Remove downloaded and extracted files."""
    print(f"Cleaning up {download_dir}...")
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    print("Cleanup complete")


def process_year(year, doi_url, tornet_data_dir, temp_dir, skip_cleanup=False):
    """Process a single year: download, extract, move, and cleanup."""
    print(f"\n{'#'*60}")
    print(f"# Processing Year {year}")
    print(f"{'#'*60}")
    
    # Create temporary directory for this year
    year_temp_dir = Path(temp_dir) / f"tornet_{year}_download"
    
    try:
        # Download
        download_dir = download_zenodo(doi_url, year_temp_dir)
        if not download_dir:
            print(f"Skipping year {year} due to download failure")
            return False
        
        # Find tar file
        tar_file = find_tar_file(year_temp_dir, year)
        if not tar_file:
            print(f"Skipping year {year} - no tar file found")
            return False
        
        # Extract
        extract_dir = year_temp_dir / "extracted"
        extract_tar_file(tar_file, extract_dir)
        
        # Move data to tornet_data
        move_data_to_tornet(extract_dir, year, tornet_data_dir)
        
        # Cleanup
        if not skip_cleanup:
            cleanup(year_temp_dir)
        else:
            print(f"Skipping cleanup for {year_temp_dir}")
        
        print(f"\n✓ Year {year} completed successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing year {year}: {e}")
        if not skip_cleanup and year_temp_dir.exists():
            cleanup(year_temp_dir)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize TorNet dataset from Zenodo"
    )
    parser.add_argument(
        "--tornet-data-dir",
        type=str,
        default="./tornet_data",
        help="Path to tornet_data directory (default: ./tornet_data)"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="./temp_downloads",
        help="Temporary directory for downloads (default: ./temp_downloads)"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(TORNET_URLS.keys()),
        help="Specific years to download (default: all years 2015-2022)"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep temporary files after processing"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    tornet_data_dir = Path(args.tornet_data_dir).resolve()
    temp_dir = Path(args.temp_dir).resolve()
    
    print("TorNet Data Download Script")
    print(f"Target directory: {tornet_data_dir}")
    print(f"Temporary directory: {temp_dir}")
    print(f"Years to process: {args.years}")
    print()
    
    # Verify tornet_data directory exists
    if not tornet_data_dir.exists():
        print(f"Error: tornet_data directory not found at {tornet_data_dir}")
        print("Please run this script from the project root or specify --tornet-data-dir")
        return
    
    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    results = {}
    for year in args.years:
        if year not in TORNET_URLS:
            print(f"Warning: No URL defined for year {year}, skipping")
            continue
        
        success = process_year(
            year,
            TORNET_URLS[year],
            tornet_data_dir,
            temp_dir,
            skip_cleanup=args.skip_cleanup
        )
        results[year] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = [year for year, success in results.items() if success]
    failed = [year for year, success in results.items() if not success]
    
    print(f"Successful: {len(successful)} years - {successful}")
    if failed:
        print(f"Failed: {len(failed)} years - {failed}")
    
    print("\nAll done!")


if __name__ == "__main__":
    main()

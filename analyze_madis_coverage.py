#!/usr/bin/env python3
"""
Analyze MADIS data coverage for TorNet storms.

This script analyzes the downloaded MADIS XML files to determine:
- How many storms have weather station data
- Coverage rates by year
- Distribution of weather stations per storm
- Data quality metrics

Usage:
    python analyze_madis_coverage.py --years 2013 2014 2015
    python analyze_madis_coverage.py --all
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import pandas as pd


def parse_storm_id_from_filename(filename):
    """Extract storm ID and timestamp from MADIS XML filename."""
    # Format: madis_data_{storm_id}_{timestamp}.xml
    try:
        parts = filename.replace('.xml', '').split('_')
        if len(parts) >= 3:
            storm_id = parts[2]
            # Extract year from timestamp if present
            if len(parts) >= 4:
                timestamp = parts[3]
                year = int(timestamp[:4]) if len(timestamp) >= 4 else None
                return storm_id, year
            return storm_id, None
        return None, None
    except:
        return None, None


def analyze_madis_coverage(madis_dir, years=None):
    """
    Analyze MADIS data coverage.
    
    Args:
        madis_dir: Directory containing MADIS XML files
        years: List of years to filter by, or None for all years
    
    Returns:
        Dictionary with coverage statistics
    """
    stats = {
        'by_year': defaultdict(lambda: {
            'total_files': 0,
            'with_data': 0,
            'without_data': 0,
            'total_records': 0,
            'storms_by_record_count': defaultdict(int)
        }),
        'overall': {
            'total_files': 0,
            'with_data': 0,
            'without_data': 0,
            'total_records': 0,
            'storms_by_record_count': defaultdict(int),
            'years_found': set()
        }
    }
    
    madis_path = Path(madis_dir)
    if not madis_path.exists():
        print(f"Error: MADIS directory not found: {madis_dir}")
        return None
    
    xml_files = list(madis_path.glob('*.xml'))
    if not xml_files:
        print(f"Warning: No XML files found in {madis_dir}")
        return stats
    
    print(f"Found {len(xml_files)} XML files in {madis_dir}")
    print("Analyzing files...\n")
    
    for xml_file in xml_files:
        storm_id, year = parse_storm_id_from_filename(xml_file.name)
        
        # Skip if filtering by years and this year doesn't match
        if years and (year is None or year not in years):
            continue
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            records = root.findall('.//record')
            num_records = len(records)
            
            # Update overall stats
            stats['overall']['total_files'] += 1
            stats['overall']['total_records'] += num_records
            if year:
                stats['overall']['years_found'].add(year)
            
            if num_records > 0:
                stats['overall']['with_data'] += 1
                stats['overall']['storms_by_record_count'][num_records] += 1
            else:
                stats['overall']['without_data'] += 1
            
            # Update year-specific stats
            if year:
                year_stats = stats['by_year'][year]
                year_stats['total_files'] += 1
                year_stats['total_records'] += num_records
                
                if num_records > 0:
                    year_stats['with_data'] += 1
                    year_stats['storms_by_record_count'][num_records] += 1
                else:
                    year_stats['without_data'] += 1
                    
        except Exception as e:
            print(f"Warning: Could not parse {xml_file.name}: {e}")
            continue
    
    return stats


def print_statistics(stats, years_requested=None):
    """Print formatted statistics."""
    
    overall = stats['overall']
    
    print("=" * 70)
    print("MADIS DATA COVERAGE ANALYSIS")
    print("=" * 70)
    
    if years_requested:
        print(f"Analyzing years: {', '.join(map(str, sorted(years_requested)))}")
    else:
        print("Analyzing all years")
    
    print(f"Years found in data: {', '.join(map(str, sorted(overall['years_found'])))}")
    print()
    
    # Overall statistics
    print("-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total storm files analyzed:          {overall['total_files']}")
    print(f"Storms WITH weather station data:    {overall['with_data']}")
    print(f"Storms WITHOUT data:                 {overall['without_data']}")
    
    if overall['total_files'] > 0:
        coverage_rate = overall['with_data'] / overall['total_files'] * 100
        print(f"Coverage rate:                       {coverage_rate:.1f}%")
    
    print(f"\nTotal weather station observations:  {overall['total_records']}")
    
    if overall['with_data'] > 0:
        avg_per_storm = overall['total_records'] / overall['with_data']
        print(f"Avg observations per storm (with data): {avg_per_storm:.1f}")
    
    # Distribution
    if overall['storms_by_record_count']:
        print(f"\nDistribution of weather stations per storm:")
        record_counts = sorted(overall['storms_by_record_count'].keys())
        for count in record_counts:
            num_storms = overall['storms_by_record_count'][count]
            print(f"  {count:3d} stations: {num_storms:4d} storms")
    
    # Year-by-year breakdown
    if stats['by_year']:
        print()
        print("-" * 70)
        print("YEAR-BY-YEAR BREAKDOWN")
        print("-" * 70)
        
        for year in sorted(stats['by_year'].keys()):
            year_stats = stats['by_year'][year]
            print(f"\n{year}:")
            print(f"  Total storms:        {year_stats['total_files']}")
            print(f"  With data:           {year_stats['with_data']}")
            print(f"  Without data:        {year_stats['without_data']}")
            
            if year_stats['total_files'] > 0:
                coverage = year_stats['with_data'] / year_stats['total_files'] * 100
                print(f"  Coverage rate:       {coverage:.1f}%")
            
            print(f"  Total observations:  {year_stats['total_records']}")
            
            if year_stats['with_data'] > 0:
                avg = year_stats['total_records'] / year_stats['with_data']
                print(f"  Avg per storm:       {avg:.1f}")
    
    print()
    print("=" * 70)
    
    # Assessment
    print("\nASSESSMENT:")
    if overall['total_files'] == 0:
        print("❌ No data found to analyze")
    elif overall['with_data'] == 0:
        print("❌ No storms have weather station data")
    elif overall['with_data'] < overall['total_files'] * 0.3:
        print(f"⚠️  Low coverage ({coverage_rate:.1f}%) - PWS data may not significantly help")
        print("   Consider focusing on urban storm subsets or supplementing with other data")
    elif overall['with_data'] < overall['total_files'] * 0.5:
        print(f"⚠️  Moderate coverage ({coverage_rate:.1f}%) - PWS data could help but may be limited")
        print("   Recommended: Use as auxiliary features with proper handling of missing data")
    else:
        print(f"✅ Good coverage ({coverage_rate:.1f}%) - PWS data should be useful for modeling")
        print("   Recommended: Incorporate as auxiliary features in hybrid CNN architecture")
    
    if overall['total_records'] > 0:
        print(f"\n💡 You have {overall['total_records']} total weather observations")
        print(f"   This is enough data to potentially improve model performance")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MADIS data coverage for TorNet storms",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--madis-dir',
        type=str,
        default='tornet_data/madis_data',
        help='Directory containing MADIS XML files (default: tornet_data/madis_data)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Specific years to analyze (e.g., --years 2013 2014 2015)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all years (default if no --years specified)'
    )
    
    args = parser.parse_args()
    
    # Determine years to analyze
    years = None
    if args.years:
        years = args.years
        print(f"Filtering for years: {years}")
    elif not args.all:
        # Default to common years if nothing specified
        years = None
        print("Analyzing all available years")
    
    # Run analysis
    stats = analyze_madis_coverage(args.madis_dir, years)
    
    if stats is None:
        sys.exit(1)
    
    # Print results
    print_statistics(stats, years)


if __name__ == '__main__':
    main()

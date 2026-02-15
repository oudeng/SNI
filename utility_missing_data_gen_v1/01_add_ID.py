#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add ID column to CSV file.

Usage:
    python 01_add_ID.py --input <input_csv_path> --output <output_csv_path>

Example:
    python utility_missing_data_gen_v1/01_add_ID.py --input data/raw.csv --output data/with_id.csv

Arguments:
    --input, -i   : Path to the input CSV file
    --output, -o  : Path to the output CSV file with ID column added

Description:
    This script reads a CSV file, adds an "ID" column with sequential integers
    starting from 1, and saves the result to the specified output file.
    The ID column is inserted as the first column of the dataframe.
"""

import argparse
import pandas as pd
import sys


def add_id_column(input_path: str, output_path: str) -> None:
    """
    Read a CSV file, add an ID column, and save to output path.
    
    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path to save the output CSV file with ID column.
    """
    # Read the input CSV file
    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    original_cols = len(df.columns)
    print(f"Original shape: {original_rows} rows × {original_cols} columns")
    
    # Add ID column (1 to N)
    df.insert(0, "ID", range(1, len(df) + 1))
    
    # Save to output file
    df.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")
    print(f"New shape: {len(df)} rows × {len(df.columns)} columns (ID column added)")


def main():
    parser = argparse.ArgumentParser(
        description="Add ID column to CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python 01_add_ID.py --input data/raw.csv --output data/with_id.csv
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        add_id_column(args.input, args.output)
        print("Done!")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
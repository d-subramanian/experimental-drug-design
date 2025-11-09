#!/usr/bin/env python3
"""
Convert TSV file with molecule-binding data to JSON format.
Takes a TSV file with SMILES strings and binding affinities,
and outputs a JSON file with SMILES as keys and affinities as values.
"""

import argparse
import json
import sys
import csv
from pathlib import Path


def find_columns(headers):
    """
    Find SMILES and binding affinity columns in the header.
    Returns (smiles_col_idx, affinity_col_idx) or (None, None) if not found.
    """
    smiles_col = None
    affinity_col = None
    
    # Common column name variations
    smiles_names = ['smiles', 'smi', 'SMILES', 'SMI', 'molecule', 'Molecule']
    affinity_names = ['affinity', 'binding_affinity', 'binding', 'Binding', 
                     'Binding_Affinity', 'ki', 'Ki', 'IC50', 'ic50', 'kd', 'Kd',
                     'pIC50', 'pKi', 'pKd', 'logKi', 'logIC50']
    
    for i, header in enumerate(headers):
        header_lower = header.lower().strip()
        # Check for SMILES column
        if any(name.lower() in header_lower for name in smiles_names) or header_lower == 'smiles':
            smiles_col = i
        # Check for binding affinity column
        if any(name.lower() in header_lower for name in affinity_names):
            affinity_col = i
    
    return smiles_col, affinity_col


def tsv_to_json(input_tsv, output_json, smiles_col=None, affinity_col=None):
    """
    Convert TSV file to JSON format.
    
    Args:
        input_tsv: Path to input TSV file
        output_json: Path to output JSON file
        smiles_col: Column index or name for SMILES (None for auto-detect)
        affinity_col: Column index or name for binding affinity (None for auto-detect)
    """
    data_dict = {}
    skipped_rows = []
    
    try:
        with open(input_tsv, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            first_line = f.readline()
            f.seek(0)
            
            # Check if it's tab-separated or comma-separated
            delimiter = '\t' if '\t' in first_line else ','
            
            reader = csv.reader(f, delimiter=delimiter)
            headers = next(reader)
            
            # Find columns if not specified
            smiles_idx = smiles_col
            affinity_idx = affinity_col
            
            if smiles_idx is None or affinity_idx is None:
                detected_smiles, detected_affinity = find_columns(headers)
                if smiles_idx is None:
                    smiles_idx = detected_smiles
                if affinity_idx is None:
                    affinity_idx = detected_affinity
            
            # If column indices were provided as strings (column names), convert them
            if isinstance(smiles_idx, str):
                try:
                    smiles_idx = headers.index(smiles_idx)
                except ValueError:
                    print(f"Error: Column '{smiles_idx}' not found in TSV file.")
                    print(f"Available columns: {', '.join(headers)}")
                    sys.exit(1)
            
            if isinstance(affinity_idx, str):
                try:
                    affinity_idx = headers.index(affinity_idx)
                except ValueError:
                    print(f"Error: Column '{affinity_idx}' not found in TSV file.")
                    print(f"Available columns: {', '.join(headers)}")
                    sys.exit(1)
            
            # Validate column indices
            if smiles_idx is None or affinity_idx is None:
                print("Error: Could not identify SMILES and/or binding affinity columns.")
                print(f"Headers found: {', '.join(headers)}")
                print("\nPlease specify columns using --smiles-col and --affinity-col options.")
                sys.exit(1)
            
            if smiles_idx >= len(headers) or affinity_idx >= len(headers):
                print(f"Error: Column indices out of range. File has {len(headers)} columns.")
                sys.exit(1)
            
            print(f"Using SMILES column: {headers[smiles_idx]} (index {smiles_idx})")
            print(f"Using affinity column: {headers[affinity_idx]} (index {affinity_idx})")
            
            # Process rows
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because header is row 1
                if len(row) <= max(smiles_idx, affinity_idx):
                    skipped_rows.append((row_num, "Not enough columns"))
                    continue
                
                smiles = row[smiles_idx].strip()
                affinity = row[affinity_idx].strip()
                
                # Skip empty rows
                if not smiles or not affinity:
                    skipped_rows.append((row_num, "Empty SMILES or affinity"))
                    continue
                
                # Try to convert affinity to number
                try:
                    affinity_value = float(affinity)
                except ValueError:
                    # Keep as string if it's not a number
                    affinity_value = affinity
                
                # Handle duplicate SMILES (use the last value or warn)
                if smiles in data_dict:
                    print(f"Warning: Duplicate SMILES found at row {row_num}: {smiles}")
                    print(f"  Previous value: {data_dict[smiles]}, New value: {affinity_value}")
                
                data_dict[smiles] = affinity_value
            
            print(f"\nProcessed {len(data_dict)} molecules")
            if skipped_rows:
                print(f"Skipped {len(skipped_rows)} rows")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_tsv}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        sys.exit(1)
    
    # Write JSON file
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote JSON file: {output_json}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert TSV file with molecule-binding data to JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect columns
  python tsv_to_json.py input.tsv output.json
  
  # Specify columns by name
  python tsv_to_json.py input.tsv output.json --smiles-col "SMILES" --affinity-col "Binding_Affinity"
  
  # Specify columns by index (0-based)
  python tsv_to_json.py input.tsv output.json --smiles-col 0 --affinity-col 1
        """
    )
    parser.add_argument('input_tsv', type=str, help='Input TSV file path')
    parser.add_argument('output_json', type=str, help='Output JSON file path')
    parser.add_argument('--smiles-col', type=str, default=None,
                       help='SMILES column name or index (0-based). Auto-detected if not specified.')
    parser.add_argument('--affinity-col', type=str, default=None,
                       help='Binding affinity column name or index (0-based). Auto-detected if not specified.')
    
    args = parser.parse_args()
    
    # Convert column arguments to int if they're numeric strings
    smiles_col = args.smiles_col
    affinity_col = args.affinity_col
    
    if smiles_col is not None:
        try:
            smiles_col = int(smiles_col)
        except ValueError:
            pass  # Keep as string if it's not a number
    
    if affinity_col is not None:
        try:
            affinity_col = int(affinity_col)
        except ValueError:
            pass  # Keep as string if it's not a number
    
    tsv_to_json(args.input_tsv, args.output_json, smiles_col, affinity_col)


if __name__ == '__main__':
    main()


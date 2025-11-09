# experimental-drug-design
Drug design method based on learning from experimental results

## TSV to JSON Converter

This repository includes a Python script (`tsv_to_json.py`) that converts TSV files containing molecule-binding data into JSON format.

### Usage

```bash
python3 tsv_to_json.py <input_tsv> <output_json> [options]
```

### Examples

```bash
# Auto-detect SMILES and binding affinity columns
python3 tsv_to_json.py data.tsv output.json

# Specify columns by name
python3 tsv_to_json.py data.tsv output.json --smiles-col "SMILES" --affinity-col "Binding_Affinity"

# Specify columns by index (0-based)
python3 tsv_to_json.py data.tsv output.json --smiles-col 0 --affinity-col 1
```

### Output Format

The script generates a JSON file where:
- **Keys**: SMILES strings (molecule structures)
- **Values**: Binding affinities (as numbers or strings)

Example output:
```json
{
  "CCO": -5.2,
  "CC(=O)O": -4.8,
  "c1ccccc1": -3.5
}
```

### Features

- Automatic column detection for SMILES and binding affinity
- Supports various column name formats (SMILES, Ki, IC50, Kd, etc.)
- Handles duplicate SMILES (warns and uses the last value)
- Converts numeric affinities automatically
- Skips empty rows and invalid data

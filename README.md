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

## Molecular Property Prediction

This repository includes a neural network training script (`train_model.py`) that predicts molecular properties from SMILES strings using RDKit fingerprints.

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### As a Python Function

```python
from train_model import train_molecular_property_model

# Train a model on your JSON data
model, test_error = train_molecular_property_model('data.json')

# Use the trained model for predictions
# (model is a PyTorch model that can be used for inference)
```

#### As a Command-Line Script

```bash
python3 train_model.py <json_file> [options]
```

### Examples

```bash
# Train with default settings (70% train, 10% val, 20% test)
python3 train_model.py data.json

# Train with custom parameters
python3 train_model.py data.json --num-epochs 200 --batch-size 64 --learning-rate 0.0001

# Save the trained model
python3 train_model.py data.json --save-model model.pth
```

### Function Details

The `train_molecular_property_model` function:

1. **Loads JSON data**: Reads a JSON file with SMILES strings as keys and affinity data as values
2. **Generates fingerprints**: Creates RDKit Morgan fingerprints (2048 bits, radius 2) for each molecule
3. **Splits data**: Creates train/validation/test datasets (70%/10%/20% by default)
4. **Trains neural network**: Trains a multi-layer perceptron to predict properties from fingerprints
5. **Returns**: The trained model and test error (RMSE)

### Model Architecture

- Input: 2048-bit Morgan fingerprints
- Hidden layers: 512 → 256 → 128 neurons (default)
- Activation: ReLU with dropout (0.3)
- Output: Single value (regression)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

### Features

- Automatic fingerprint generation from SMILES strings
- Handles invalid SMILES gracefully (skips with warning)
- Early stopping based on validation loss
- GPU support (automatically uses CUDA if available)
- Reproducible results (configurable random seed)
- Comprehensive training progress reporting

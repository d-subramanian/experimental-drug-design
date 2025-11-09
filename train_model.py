#!/usr/bin/env python3
"""
Train a neural network to predict molecular properties from SMILES strings.
Takes a JSON file with SMILES strings as keys and affinity data as values,
generates RDKit fingerprints, and trains a neural network.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, Tuple, Optional
import warnings


class MolecularDataset(Dataset):
    """PyTorch Dataset for molecular fingerprints and properties."""
    
    def __init__(self, fingerprints, properties):
        """
        Args:
            fingerprints: Tensor of molecular fingerprints
            properties: Tensor of property values (affinities)
        """
        self.fingerprints = fingerprints
        self.properties = properties
    
    def __len__(self):
        return len(self.fingerprints)
    
    def __getitem__(self, idx):
        return self.fingerprints[idx], self.properties[idx]


class MolecularPropertyPredictor(nn.Module):
    """Neural network for predicting molecular properties from fingerprints."""
    
    def __init__(self, input_size: int, hidden_sizes: list = [512, 256, 128], dropout: float = 0.3):
        """
        Args:
            input_size: Size of input fingerprint vector
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(MolecularPropertyPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)


def generate_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Generate Morgan fingerprint for a SMILES string.
    
    Args:
        smiles: SMILES string
        radius: Radius for Morgan fingerprint (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)
    
    Returns:
        numpy array of fingerprint bits, or None if SMILES is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fingerprint, dtype=np.float32)
    except Exception as e:
        warnings.warn(f"Error generating fingerprint for SMILES {smiles}: {e}")
        return None


def load_json_data(json_file: str) -> Dict[str, float]:
    """
    Load JSON file with SMILES strings and affinity data.
    
    Args:
        json_file: Path to JSON file
    
    Returns:
        Dictionary with SMILES as keys and affinities as values
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def prepare_data(data: Dict[str, float], fingerprint_radius: int = 2, 
                 fingerprint_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fingerprints for all molecules and prepare data arrays.
    
    Args:
        data: Dictionary with SMILES as keys and affinities as values
        fingerprint_radius: Radius for Morgan fingerprint
        fingerprint_bits: Number of bits in fingerprint
    
    Returns:
        Tuple of (fingerprints, properties) as numpy arrays
    """
    fingerprints = []
    properties = []
    skipped = 0
    
    for smiles, affinity in data.items():
        fingerprint = generate_fingerprint(smiles, fingerprint_radius, fingerprint_bits)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
            # Convert affinity to float if it's not already
            try:
                properties.append(float(affinity))
            except (ValueError, TypeError):
                warnings.warn(f"Could not convert affinity '{affinity}' to float for SMILES {smiles}")
                skipped += 1
                continue
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} invalid molecules")
    
    if len(fingerprints) == 0:
        raise ValueError("No valid molecules found in the data")
    
    return np.array(fingerprints), np.array(properties)


def split_data(fingerprints: np.ndarray, properties: np.ndarray, 
               train_ratio: float = 0.7, val_ratio: float = 0.1, 
               test_ratio: float = 0.2, random_seed: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        fingerprints: Array of fingerprints
        properties: Array of property values
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.1)
        test_ratio: Fraction of data for testing (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_fp, train_prop, val_fp, val_prop, test_fp, test_prop)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    np.random.seed(random_seed)
    n_samples = len(fingerprints)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_fp = fingerprints[train_indices]
    train_prop = properties[train_indices]
    val_fp = fingerprints[val_indices]
    val_prop = properties[val_indices]
    test_fp = fingerprints[test_indices]
    test_prop = properties[test_indices]
    
    return train_fp, train_prop, val_fp, val_prop, test_fp, test_prop


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 100, learning_rate: float = 0.001, 
                device: str = 'cpu', patience: int = 10) -> Tuple[nn.Module, list, list]:
    """
    Train the neural network model.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        patience: Early stopping patience (epochs without improvement)
    
    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for fingerprints, properties in train_loader:
            fingerprints = fingerprints.to(device)
            properties = properties.to(device)
            
            optimizer.zero_grad()
            outputs = model(fingerprints)
            loss = criterion(outputs, properties)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fingerprints, properties in val_loader:
                fingerprints = fingerprints.to(device)
                properties = properties.to(device)
                
                outputs = model(fingerprints)
                loss = criterion(outputs, properties)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def calculate_test_error(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> float:
    """
    Calculate test error (RMSE) on test set.
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cpu' or 'cuda')
    
    Returns:
        Test RMSE (Root Mean Squared Error)
    """
    model.eval()
    model = model.to(device)
    test_loss = 0.0
    
    with torch.no_grad():
        for fingerprints, properties in test_loader:
            fingerprints = fingerprints.to(device)
            properties = properties.to(device)
            
            outputs = model(fingerprints)
            loss = nn.MSELoss()(outputs, properties)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_rmse = np.sqrt(test_loss)
    
    return test_rmse


def train_molecular_property_model(json_file: str, fingerprint_radius: int = 2,
                                  fingerprint_bits: int = 2048, train_ratio: float = 0.7,
                                  val_ratio: float = 0.1, test_ratio: float = 0.2,
                                  hidden_sizes: list = [512, 256, 128], dropout: float = 0.3,
                                  batch_size: int = 32, num_epochs: int = 100,
                                  learning_rate: float = 0.001, random_seed: int = 42,
                                  device: Optional[str] = None, patience: int = 10) -> Tuple[nn.Module, float]:
    """
    Main function to train a neural network on molecular property data.
    
    This function:
    1. Loads JSON file with SMILES strings and affinity data
    2. Generates RDKit fingerprints for each molecule
    3. Splits data into train/validation/test sets (70/10/20 by default)
    4. Trains a neural network to predict properties from fingerprints
    5. Returns the trained model and test error
    
    Args:
        json_file: Path to JSON file with SMILES as keys and affinities as values
        fingerprint_radius: Radius for Morgan fingerprint (default: 2)
        fingerprint_bits: Number of bits in fingerprint (default: 2048)
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.1)
        test_ratio: Fraction of data for testing (default: 0.2)
        hidden_sizes: List of hidden layer sizes (default: [512, 256, 128])
        dropout: Dropout probability (default: 0.3)
        batch_size: Batch size for training (default: 32)
        num_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate for optimizer (default: 0.001)
        random_seed: Random seed for reproducibility (default: 42)
        device: Device to train on ('cpu' or 'cuda'). Auto-detects if None.
        patience: Early stopping patience (default: 10)
    
    Returns:
        Tuple of (trained_model, test_rmse)
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {json_file}...")
    data = load_json_data(json_file)
    print(f"Loaded {len(data)} molecules")
    
    # Generate fingerprints
    print(f"Generating fingerprints (radius={fingerprint_radius}, bits={fingerprint_bits})...")
    fingerprints, properties = prepare_data(data, fingerprint_radius, fingerprint_bits)
    print(f"Generated fingerprints for {len(fingerprints)} molecules")
    print(f"Fingerprint shape: {fingerprints.shape}")
    print(f"Property range: [{properties.min():.4f}, {properties.max():.4f}]")
    
    # Split data
    print(f"Splitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    train_fp, train_prop, val_fp, val_prop, test_fp, test_prop = split_data(
        fingerprints, properties, train_ratio, val_ratio, test_ratio, random_seed
    )
    print(f"Train set: {len(train_fp)} samples")
    print(f"Validation set: {len(val_fp)} samples")
    print(f"Test set: {len(test_fp)} samples")
    
    # Convert to tensors and create datasets
    train_fp_tensor = torch.FloatTensor(train_fp)
    train_prop_tensor = torch.FloatTensor(train_prop)
    val_fp_tensor = torch.FloatTensor(val_fp)
    val_prop_tensor = torch.FloatTensor(val_prop)
    test_fp_tensor = torch.FloatTensor(test_fp)
    test_prop_tensor = torch.FloatTensor(test_prop)
    
    train_dataset = TensorDataset(train_fp_tensor, train_prop_tensor)
    val_dataset = TensorDataset(val_fp_tensor, val_prop_tensor)
    test_dataset = TensorDataset(test_fp_tensor, test_prop_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = fingerprint_bits
    model = MolecularPropertyPredictor(input_size, hidden_sizes, dropout)
    print(f"Model architecture: {model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"\nTraining model for up to {num_epochs} epochs...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate, device, patience
    )
    
    # Calculate test error
    print("\nEvaluating on test set...")
    test_rmse = calculate_test_error(model, test_loader, device)
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return model, test_rmse


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train a neural network to predict molecular properties from SMILES strings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('json_file', type=str, help='Path to JSON file with SMILES and affinities')
    parser.add_argument('--fingerprint-radius', type=int, default=2,
                       help='Radius for Morgan fingerprint (default: 2)')
    parser.add_argument('--fingerprint-bits', type=int, default=2048,
                       help='Number of bits in fingerprint (default: 2048)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Fraction of data for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Fraction of data for validation (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer sizes (default: 512 256 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability (default: 0.3)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cpu or cuda). Auto-detects if not specified.')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save the trained model (optional)')
    
    args = parser.parse_args()
    
    # Train model
    model, test_rmse = train_molecular_property_model(
        json_file=args.json_file,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        device=args.device,
        patience=args.patience
    )
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"\nModel saved to {args.save_model}")
    
    print(f"\nTraining completed. Final test RMSE: {test_rmse:.4f}")


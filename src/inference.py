"""
Inference Script for Temporal Graph Models

This script loads trained models (LSTM, GAT-GRU, Graph WaveNet) and performs inference
on the test set. It generates predictions for t+1 given sequences up to time t, and
compares predictions against ground truth across the entire time period.

Output:
- Time series of predictions vs ground truth for each model
- Comparison metrics and visualizations
- Results saved to specified path in config
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
import json
import argparse
from pathlib import Path

from modules.lstm import TemporalGraphLSTM
from modules.gatgru import TemporalGraphGATGRU
from modules.wavenet import TemporalGraphWaveNet

import warnings
warnings.filterwarnings('ignore')


def create_sequences(graphs, seq_length):
    """
    Create input-target sequence pairs for temporal prediction.
    
    Args:
        graphs: List of graph snapshots
        seq_length: Length of input sequences
        
    Returns:
        List of (input_sequence, target, timestep) tuples
    """
    sequences = []
    for i in range(len(graphs) - seq_length):
        input_seq = graphs[i:i + seq_length]
        target = graphs[i + seq_length]
        timestep = i + seq_length  # The timestep being predicted
        sequences.append((input_seq, target, timestep))
    return sequences


def predict_sequence(model, sequences, device, model_name):
    """
    Generate predictions for all sequences.
    
    Args:
        model: Trained model
        sequences: List of (input_sequence, target, timestep) tuples
        device: Device to run inference on
        model_name: Name of the model for display
        
    Returns:
        Dictionary containing predictions, ground truth, and timesteps
    """
    model.eval()
    
    predictions = []
    ground_truths = []
    timesteps = []
    
    with torch.no_grad():
        for input_seq, target, timestep in tqdm(sequences, desc=f"  {model_name} inference", leave=False):
            # Move data to device
            input_seq = [graph.to(device) for graph in input_seq]
            target = target.to(device)
            
            # Ensure target has proper batch tensor for single graph
            if not isinstance(target, Batch):
                # Manually add batch tensor for single graph
                target.batch = torch.zeros(target.num_nodes, dtype=torch.long, device=device)
                # Create a simple Batch-like wrapper
                class SingleGraphBatch:
                    def __init__(self, data):
                        self.x = data.x
                        self.edge_index = data.edge_index
                        self.edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                        self.batch = data.batch
                        self.num_graphs = 1
                        
                target = SingleGraphBatch(target)
            
            # Forward pass
            output, _ = model(input_seq)
            
            # Get prediction for last timestep
            prediction = output[:, -1, :]  # [batch_size, output_dim]
            
            # Compute target representation
            target_repr = model.graph_encoder(target)
            
            # Project target to output dimension if needed
            if target_repr.shape[1] != prediction.shape[1]:
                target_repr = target_repr[:, :prediction.shape[1]]
            
            # Store results
            predictions.append(prediction.cpu().numpy())
            ground_truths.append(target_repr.cpu().numpy())
            timesteps.append(timestep)
    
    # Convert to numpy arrays
    predictions = np.concatenate(predictions, axis=0)  # [num_sequences, output_dim]
    ground_truths = np.concatenate(ground_truths, axis=0)  # [num_sequences, output_dim]
    timesteps = np.array(timesteps)  # [num_sequences]
    
    # Normalize embeddings (z-score normalization)
    # Normalize across the temporal dimension (each embedding dimension separately)
    predictions_mean = predictions.mean(axis=0, keepdims=True)
    predictions_std = predictions.std(axis=0, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
    predictions_normalized = (predictions - predictions_mean) / predictions_std
    
    ground_truths_mean = ground_truths.mean(axis=0, keepdims=True)
    ground_truths_std = ground_truths.std(axis=0, keepdims=True) + 1e-8
    ground_truths_normalized = (ground_truths - ground_truths_mean) / ground_truths_std
    
    return {
        'predictions': predictions_normalized,
        'ground_truth': ground_truths_normalized,
        'predictions_raw': predictions,
        'ground_truth_raw': ground_truths,
        'timesteps': timesteps
    }


def compute_metrics(predictions, ground_truth):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values [num_sequences, output_dim]
        ground_truth: Ground truth values [num_sequences, output_dim]
        
    Returns:
        Dictionary of metrics
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - ground_truth))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    # Mean Absolute Percentage Error
    epsilon = 1e-8  # To avoid division by zero
    mape = np.mean(np.abs((ground_truth - predictions) / (ground_truth + epsilon))) * 100
    
    # R-squared
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    # Standardized RMSE
    std_gt = np.std(ground_truth)
    standardized_rmse = rmse / std_gt if std_gt > 0 else 0.0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'standardized_rmse': float(standardized_rmse)
    }


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run inference on trained temporal graph models')
parser.add_argument('--config', type=str, default='src/config.yaml',
                    help='Path to configuration file (default: config.yaml)')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

print("="*80)
print("Temporal Graph Models - Inference and Comparison")
print("="*80)

# Load data
print("\n[1/5] Loading temporal graph sequence...")
graph_sequence_path = config['paths']['graph_sequence_path']

if config['test'] == 1:
    print('TEST CONFIG')
    temporal_graphs_data = torch.load(graph_sequence_path, weights_only=False)[:100]
else:
    temporal_graphs_data = torch.load(graph_sequence_path, weights_only=False)

# Split data (use same split as training)
print("\n[2/5] Splitting data chronologically...")
train_ratio = config['training']['train_split']
val_ratio = config['training']['val_split']

num_snapshots = len(temporal_graphs_data)
train_size = int(num_snapshots * train_ratio)
val_size = int(num_snapshots * val_ratio)

# Use test set for inference
test_graphs = temporal_graphs_data[train_size + val_size:]

print(f"  Total snapshots: {num_snapshots}")
print(f"  Test snapshots: {len(test_graphs)}")

# Store metadata
num_node_features = test_graphs[0].x.shape[1]
num_edge_features = test_graphs[0].edge_attr.shape[1] if test_graphs[0].edge_attr.numel() > 0 else 0

print(f"\n  Graph statistics:")
print(f"    Node features: {num_node_features}")
print(f"    Edge features: {num_edge_features}")

# Create sequences
print("\n[3/5] Creating temporal sequences...")
sequence_length = config['training']['input_sequence_length']
test_sequences = create_sequences(test_graphs, sequence_length)

print(f"  Sequence length: {sequence_length}")
print(f"  Test sequences: {len(test_sequences)}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")

# Initialize results dictionary
results = {
    'metadata': {
        'num_test_snapshots': len(test_graphs),
        'num_test_sequences': len(test_sequences),
        'sequence_length': sequence_length,
        'node_features': num_node_features,
        'edge_features': num_edge_features
    },
    'models': {}
}

# Load and run inference for each model
print("\n[4/5] Loading models and generating predictions...")
print("="*80)

models_config = [
    {
        'name': 'LSTM',
        'class': TemporalGraphLSTM,
        'checkpoint_path': config['paths']['lstm_best_model_path'],
        'config_key': 'lstm',
        'init_params': lambda cfg: {
            'node_feature_dim': num_node_features,
            'edge_feature_dim': num_edge_features,
            'hidden_dim': cfg['hidden_dim'],
            'num_layers': cfg['num_layers'],
            'output_dim': cfg['output_dim'],
            'aggregation': cfg['aggregation'],
            'dropout': cfg['dropout'],
            'bidirectional': cfg['bidirectional']
        }
    },
    {
        'name': 'GAT-GRU',
        'class': TemporalGraphGATGRU,
        'checkpoint_path': config['paths']['gatgru_best_model_path'],
        'config_key': 'gatgru',
        'init_params': lambda cfg: {
            'node_feature_dim': num_node_features,
            'edge_feature_dim': num_edge_features,
            'hidden_dim': cfg['hidden_dim'],
            'num_gat_layers': cfg['num_gat_layers'],
            'num_gru_layers': cfg['num_gru_layers'],
            'num_heads': cfg['num_heads'],
            'output_dim': cfg['output_dim'],
            'aggregation': cfg['aggregation'],
            'dropout': cfg['dropout'],
            'bidirectional': cfg['bidirectional'],
            'use_edge_attr_in_gat': cfg['use_edge_attr_in_gat']
        }
    },
    {
        'name': 'Graph WaveNet',
        'class': TemporalGraphWaveNet,
        'checkpoint_path': config['paths']['wavenet_best_model_path'],
        'config_key': 'wavenet',
        'init_params': lambda cfg: {
            'node_feature_dim': num_node_features,
            'edge_feature_dim': num_edge_features,
            'hidden_dim': cfg['hidden_dim'],
            'num_gcn_layers': cfg['num_gcn_layers'],
            'num_wavenet_blocks': cfg['num_wavenet_blocks'],
            'num_layers_per_block': cfg['num_layers_per_block'],
            'kernel_size': cfg['kernel_size'],
            'output_dim': cfg['output_dim'],
            'aggregation': cfg['aggregation'],
            'dropout': cfg['dropout']
        }
    }
]

for model_config in models_config:
    print(f"\n{model_config['name']}:")
    print("-" * 40)
    
    try:
        # Load model configuration
        model_cfg = config[model_config['config_key']]
        
        # Initialize model
        model = model_config['class'](**model_config['init_params'](model_cfg)).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_config['checkpoint_path'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"  ✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
        print(f"  ✓ Training val_loss: {checkpoint['val_loss']:.6f}")
        
        # Generate predictions
        inference_results = predict_sequence(model, test_sequences, device, model_config['name'])
        
        # Compute metrics
        metrics = compute_metrics(
            inference_results['predictions'],
            inference_results['ground_truth']
        )
        
        print(f"\n  Inference Metrics:")
        print(f"    MAE: {metrics['mae']:.6f}")
        print(f"    RMSE: {metrics['rmse']:.6f}")
        print(f"    Standardized RMSE: {metrics['standardized_rmse']:.6f}")
        print(f"    MAPE: {metrics['mape']:.2f}%")
        print(f"    R²: {metrics['r2']:.6f}")
        
        # Store results
        results['models'][model_config['name']] = {
            'predictions': inference_results['predictions'].tolist(),
            'ground_truth': inference_results['ground_truth'].tolist(),            'predictions_raw': inference_results['predictions_raw'].tolist(),
            'ground_truth_raw': inference_results['ground_truth_raw'].tolist(),            'timesteps': inference_results['timesteps'].tolist(),
            'metrics': metrics,
            'checkpoint_info': {
                'epoch': checkpoint['epoch'] + 1,
                'train_loss': float(checkpoint['train_loss']),
                'val_loss': float(checkpoint['val_loss'])            },
            'normalization_info': {
                'note': 'predictions and ground_truth are z-score normalized',
                'raw_versions': 'predictions_raw and ground_truth_raw contain unnormalized values'            }
        }
        
    except FileNotFoundError:
        print(f"  ✗ Model checkpoint not found: {model_config['checkpoint_path']}")
        print(f"  Skipping {model_config['name']}...")
    except Exception as e:
        print(f"  ✗ Error loading {model_config['name']}: {str(e)}")
        print(f"  Skipping {model_config['name']}...")

# Save results
print("\n[5/5] Saving inference results...")
print("="*80)

inference_results_path = config['paths'].get('inference_results_path', '../outputs/inference_results.json')

# Create output directory if it doesn't exist
output_dir = Path(inference_results_path).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Save to JSON
with open(inference_results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Results saved to {inference_results_path}")

# Print summary comparison
print("\n" + "="*80)
print("Summary Comparison")
print("="*80)

if len(results['models']) > 0:
    print("\nModel Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'R²':<12}")
    print("-" * 80)
    
    for model_name, model_results in results['models'].items():
        metrics = model_results['metrics']
        print(f"{model_name:<20} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
              f"{metrics['mape']:<12.2f} {metrics['r2']:<12.6f}")
    
    print("-" * 80)
    
    # Find best model for each metric
    print("\nBest Models:")
    
    # MAE (lower is better)
    best_mae_model = min(results['models'].items(), key=lambda x: x[1]['metrics']['mae'])
    print(f"  Lowest MAE: {best_mae_model[0]} ({best_mae_model[1]['metrics']['mae']:.6f})")
    
    # RMSE (lower is better)
    best_rmse_model = min(results['models'].items(), key=lambda x: x[1]['metrics']['rmse'])
    print(f"  Lowest RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['metrics']['rmse']:.6f})")
    
    # R² (higher is better)
    best_r2_model = max(results['models'].items(), key=lambda x: x[1]['metrics']['r2'])
    print(f"  Highest R²: {best_r2_model[0]} ({best_r2_model[1]['metrics']['r2']:.6f})")
    
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)
    
    print(f"\nResults contain:")
    print(f"  - Predictions for {len(test_sequences)} timesteps")
    print(f"  - Ground truth values for comparison")
    print(f"  - Timestep indices for temporal alignment")
    print(f"  - Evaluation metrics for each model")
    print(f"\nUse the saved JSON file for further analysis and visualization.")
else:
    print("\n⚠ No models were successfully loaded. Please check:")
    print("  1. Model checkpoint paths in config.yaml")
    print("  2. That models have been trained and saved")
    print("  3. Model architecture matches the saved checkpoints")

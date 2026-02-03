"""
GAT-GRU Training Script for Temporal Graph Sequences

This script trains a GAT-GRU model on temporal graph sequences for next-timestep prediction.
The model uses Graph Attention Networks (GAT) for spatial encoding and GRU for temporal modeling.
It learns to predict the graph-level representation of the next timestep given a sequence of 
historical graph snapshots.

Training process:
1. Load temporal graph sequence from saved file
2. Split data chronologically into train/val/test sets
3. Create input-target sequence pairs
4. Train GAT-GRU model with early stopping
5. Evaluate on test set
6. Save model checkpoint and training metrics
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from tqdm import tqdm
import json
import argparse

from datetime import datetime
import time

from modules.gatgru import TemporalGraphGATGRU

import warnings
warnings.filterwarnings('ignore')

# Functions
def create_sequences(graphs, seq_length):
    """
    Create input-target sequence pairs for temporal prediction.
    
    Args:
        graphs: List of graph snapshots
        seq_length: Length of input sequences
        
    Returns:
        List of (input_sequence, target) tuples
    """
    sequences = []
    for i in range(len(graphs) - seq_length):
        input_seq = graphs[i:i + seq_length]
        target = graphs[i + seq_length]
        sequences.append((input_seq, target))
    return sequences

def precompute_target_representations(model, graphs, device):
    """
    Pre-compute target representations for all graphs using the encoder.
    This prevents data leakage by fixing the target representations before training.
    
    Args:
        model: Model with graph_encoder
        graphs: List of graph snapshots
        device: Device to compute on
        
    Returns:
        List of pre-computed target representations (detached from computation graph)
    """
    model.eval()
    target_representations = []
    
    with torch.no_grad():
        for graph in tqdm(graphs, desc="  Pre-computing targets", leave=False):
            graph = graph.to(device)
            
            # Ensure graph has proper batch tensor for single graph
            if not isinstance(graph, Batch):
                graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
                class SingleGraphBatch:
                    def __init__(self, data):
                        self.x = data.x
                        self.edge_index = data.edge_index
                        self.edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                        self.batch = data.batch
                        self.num_graphs = 1
                graph = SingleGraphBatch(graph)
            
            # Compute representation
            repr = model.graph_encoder(graph)
            target_representations.append(repr.cpu())
    
    model.train()
    return target_representations

def train_epoch(model, sequences, target_reprs, optimizer, criterion, device, gradient_clip, output_dim):
    """
    Train the model for one epoch.
    
    Args:
        model: GAT-GRU model
        sequences: List of (input_sequence, target_idx) tuples
        target_reprs: Pre-computed target representations
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Gradient clipping threshold
        output_dim: Output dimension for projection
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for input_seq, target_idx in tqdm(sequences, desc="  Training", leave=False):
        # Move data to device
        input_seq = [graph.to(device) for graph in input_seq]
        
        # Get pre-computed target representation
        target_repr = target_reprs[target_idx].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _ = model(input_seq)
        
        # Get prediction for last timestep
        prediction = output[:, -1, :]  # [batch_size, output_dim]
        
        # Project target to output dimension if needed
        if target_repr.shape[1] != prediction.shape[1]:
            target_repr = target_repr[:, :prediction.shape[1]]
        
        # Compute loss
        loss = criterion(prediction, target_repr)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, sequences, target_reprs, criterion, device, output_dim):


    """
    Validate the model for one epoch.
    
    Args:
        model: GAT-GRU model
        sequences: List of (input_sequence, target_idx) tuples
        target_reprs: Pre-computed target representations
        criterion: Loss function
        device: Device to validate on        output_dim: Output dimension for projection        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_seq, target_idx in tqdm(sequences, desc="  Validation", leave=False):
            # Move data to device
            input_seq = [graph.to(device) for graph in input_seq]
            
            # Get pre-computed target representation
            target_repr = target_reprs[target_idx].to(device)
            
            # Forward pass
            output, _ = model(input_seq)
            
            # Get prediction for last timestep
            prediction = output[:, -1, :]


            
            # Project target to output dimension if needed
            if target_repr.shape[1] != prediction.shape[1]:
                target_repr = target_repr[:, :prediction.shape[1]]
            
            # Compute loss
            loss = criterion(prediction, target_repr)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_detailed_metrics(model, sequences, target_reprs, device, num_node_features, output_dim):
    """
    Compute detailed metrics including MAE, standardized RMSE, and per-feature RMSE.
    
    Args:
        model: GAT-GRU model
        sequences: List of (input_sequence, target_idx) tuples
        target_reprs: Pre-computed target representations
        device: Device to evaluate on
        num_node_features: Number of node features
        output_dim: Output dimension for projection
        
    Returns:
        Dictionary containing detailed metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for input_seq, target_idx in tqdm(sequences, desc="  Computing metrics", leave=False):
            # Move data to device
            input_seq = [graph.to(device) for graph in input_seq]
            
            # Get pre-computed target representation
            target_repr = target_reprs[target_idx].to(device)
            
            # Forward pass
            output, _ = model(input_seq)
            
            # Get prediction for last timestep
            prediction = output[:, -1, :]  # [batch_size, output_dim]
            
            # Project target to output dimension if needed
            if target_repr.shape[1] != prediction.shape[1]:
                target_repr = target_repr[:, :prediction.shape[1]]
            
            # Store predictions and targets
            all_predictions.append(prediction.cpu())
            all_targets.append(target_repr.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)  # [num_sequences, output_dim]
    all_targets = torch.cat(all_targets, dim=0)  # [num_sequences, output_dim]
    
    # Compute MAE
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
    
    # Compute RMSE
    rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
    
    # Compute standardized RMSE (RMSE / std of targets)
    target_std = torch.std(all_targets).item()
    standardized_rmse = rmse / target_std if target_std > 0 else 0.0
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'standardized_rmse': standardized_rmse,
        'target_mean': torch.mean(all_targets).item(),
        'target_std': target_std,
        'prediction_mean': torch.mean(all_predictions).item(),
        'prediction_std': torch.std(all_predictions).item(),
    }
    
    return metrics

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GAT-GRU on temporal graph sequences')
parser.add_argument('--config', type=str, default='src/config.yaml',
                    help='Path to configuration file (default: config.yaml)')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

print("="*80)
print("GAT-GRU Training for Temporal Graph Sequences")
print("="*80)

# Load data
print("\n[1/7] Loading temporal graph sequence...")
graph_sequence_path = config['paths']['graph_sequence_path']

if config['test'] == 1:
    print('TEST CONFIG')
    temporal_graphs_data = torch.load(graph_sequence_path, weights_only=False)[:100]
else:
    temporal_graphs_data = torch.load(graph_sequence_path, weights_only=False)

# Split data
print("\n[2/7] Splitting data chronologically...")
train_ratio = config['training']['train_split']
val_ratio = config['training']['val_split']
test_ratio = config['training']['test_split']

num_snapshots = len(temporal_graphs_data)
train_size = int(num_snapshots * train_ratio)
val_size = int(num_snapshots * val_ratio)
test_size = num_snapshots - train_size - val_size

train_graphs = temporal_graphs_data[:train_size]
val_graphs = temporal_graphs_data[train_size:train_size + val_size]
test_graphs = temporal_graphs_data[train_size + val_size:]

print(f"  Total snapshots: {num_snapshots}")
print(f"  Train: {len(train_graphs)} snapshots ({train_ratio*100:.0f}%)")
print(f"  Validation: {len(val_graphs)} snapshots ({val_ratio*100:.0f}%)")
print(f"  Test: {len(test_graphs)} snapshots ({test_ratio*100:.0f}%)")

# Store metadata for model initialization
num_node_features = train_graphs[0].x.shape[1]
num_edge_features = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr.numel() > 0 else 0
num_nodes = train_graphs[0].num_nodes

print(f"\n  Graph statistics:")
print(f"    Node features: {num_node_features}")
print(f"    Edge features: {num_edge_features}")
print(f"    Total nodes: {num_nodes}")

# Create input sequences
print("\n[3/7] Creating temporal sequences...")
sequence_length = config['training']['input_sequence_length']

# Note: We'll create sequences with indices after pre-computing target representations
print(f"  Sequence length: {sequence_length}")



# Initialize hyperparameters
print("\n[4/7] Initializing GAT-GRU model...")
gatgru_config = config['gatgru']
num_epochs = config['training']['num_epochs']
patience = config['training']['patience']

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# Initialize GAT-GRU model
model = TemporalGraphGATGRU(
    node_feature_dim=num_node_features,
    edge_feature_dim=num_edge_features,
    hidden_dim=gatgru_config['hidden_dim'],
    num_gat_layers=gatgru_config['num_gat_layers'],
    num_gru_layers=gatgru_config['num_gru_layers'],
    num_heads=gatgru_config['num_heads'],
    output_dim=gatgru_config['output_dim'],
    aggregation=gatgru_config['aggregation'],
    dropout=gatgru_config['dropout'],
    bidirectional=gatgru_config['bidirectional'],
    use_edge_attr_in_gat=gatgru_config['use_edge_attr_in_gat']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
# Pre-compute target representations to prevent data leakage
print("\n  Pre-computing target representations...")
print("  (This prevents the encoder from seeing target graphs during training)")
train_target_reprs = precompute_target_representations(model, train_graphs, device)
val_target_reprs = precompute_target_representations(model, val_graphs, device)
test_target_reprs = precompute_target_representations(model, test_graphs, device)

# Now create sequences with indices pointing to pre-computed representations
def create_sequences_with_indices(num_graphs, seq_length):
    """Create sequences with indices instead of actual graphs for targets."""
    sequences = []
    for i in range(num_graphs - seq_length):
        sequences.append((i, i + seq_length))  # (start_idx, target_idx)
    return sequences

# Create sequences - input_seq will use graphs, target will use pre-computed repr
train_sequences_idx = create_sequences_with_indices(len(train_graphs), sequence_length)
val_sequences_idx = create_sequences_with_indices(len(val_graphs), sequence_length)
test_sequences_idx = create_sequences_with_indices(len(test_graphs), sequence_length)

# Convert to actual sequences with graphs for input
train_sequences = [(train_graphs[start:start+sequence_length], target_idx) 
                   for start, target_idx in train_sequences_idx]
val_sequences = [(val_graphs[start:start+sequence_length], target_idx) 
                 for start, target_idx in val_sequences_idx]
test_sequences = [(test_graphs[start:start+sequence_length], target_idx) 
                  for start, target_idx in test_sequences_idx]

print(f"  Training sequences: {len(train_sequences)}")
print(f"  Validation sequences: {len(val_sequences)}")
print(f"  Test sequences: {len(test_sequences)}")
# Initialize optimizer
if gatgru_config['optimizer'] == 'adam':
    optimizer = optim.Adam(
        model.parameters(),
        lr=gatgru_config['learning_rate'],
        weight_decay=gatgru_config['weight_decay']
    )
elif gatgru_config['optimizer'] == 'sgd':
    optimizer = optim.SGD(
        model.parameters(),
        lr=gatgru_config['learning_rate'],
        weight_decay=gatgru_config['weight_decay'],
        momentum=0.9
    )
elif gatgru_config['optimizer'] == 'rmsprop':
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=gatgru_config['learning_rate'],
        weight_decay=gatgru_config['weight_decay']
    )
else:
    raise ValueError(f"Unknown optimizer: {gatgru_config['optimizer']}")

# Initialize loss function
if gatgru_config['loss_function'] == 'mse':
    criterion = nn.MSELoss()
elif gatgru_config['loss_function'] == 'mae':
    criterion = nn.L1Loss()
elif gatgru_config['loss_function'] == 'huber':
    criterion = nn.SmoothL1Loss()
else:
    raise ValueError(f"Unknown loss function: {gatgru_config['loss_function']}")

# Initialize learning rate scheduler
scheduler = None
if gatgru_config['use_scheduler']:
    if gatgru_config['scheduler_type'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=gatgru_config['scheduler_factor'],
            patience=gatgru_config['scheduler_patience']
        )
    elif gatgru_config['scheduler_type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=gatgru_config['scheduler_step_size'],
            gamma=gatgru_config['scheduler_factor']
        )
    elif gatgru_config['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )

print(f"\n  Training configuration:")
print(f"    Optimizer: {gatgru_config['optimizer']}")
print(f"    Learning rate: {gatgru_config['learning_rate']}")
print(f"    Loss function: {gatgru_config['loss_function']}")
print(f"    Scheduler: {gatgru_config['scheduler_type'] if gatgru_config['use_scheduler'] else 'None'}")
print(f"    Gradient clipping: {gatgru_config['gradient_clip']}")
print(f"    Max epochs: {num_epochs}")
print(f"    Early stopping patience: {patience}")

print(f"\n  GAT-GRU architecture:")
print(f"    Hidden dim: {gatgru_config['hidden_dim']}")
print(f"    GAT layers: {gatgru_config['num_gat_layers']}")
print(f"    GRU layers: {gatgru_config['num_gru_layers']}")
print(f"    Attention heads: {gatgru_config['num_heads']}")
print(f"    Output dim: {gatgru_config['output_dim']}")
print(f"    Aggregation: {gatgru_config['aggregation']}")
print(f"    Bidirectional: {gatgru_config['bidirectional']}")
print(f"    Use edge attr in GAT: {gatgru_config['use_edge_attr_in_gat']}")

# Training loop
print("\n[5/7] Training model...")
print("="*80)

best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses = []
val_losses = []
learning_rates = []
epoch_times = []

training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Train
    train_loss = train_epoch(
        model, train_sequences, train_target_reprs, optimizer, criterion, device, 
        gatgru_config['gradient_clip'], gatgru_config['output_dim']
    )
    train_losses.append(train_loss)
    
    # Validate
    val_loss = validate_epoch(model, val_sequences, val_target_reprs, criterion, device, 
                              gatgru_config['output_dim'])
    val_losses.append(val_loss)
    
    # Track learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Track epoch time
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    
    # Print metrics
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss:   {val_loss:.6f}")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Epoch Time: {epoch_time:.2f}s")
    
    # Learning rate scheduling
    if scheduler is not None:
        if gatgru_config['scheduler_type'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save best model
        best_model_path = config['paths']['gatgru_best_model_path']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }, best_model_path)
        print(f"  ✓ Best model saved to {best_model_path} (val_loss: {val_loss:.6f})")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement for {epochs_without_improvement} epoch(s)")
    
    # Check early stopping
    if epochs_without_improvement >= patience:
        print(f"\n  Early stopping triggered after {epoch + 1} epochs")
        break

training_time = time.time() - training_start_time

# Testing
print("\n" + "="*80)
print("[6/7] Evaluating on test set...")

# Load best model for testing
checkpoint = torch.load(config['paths']['gatgru_best_model_path'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"  Loaded best model from epoch {checkpoint['epoch'] + 1}")

# Compute basic test loss
test_loss = validate_epoch(model, test_sequences, test_target_reprs, criterion, device, 
                           gatgru_config['output_dim'])
print(f"  Test Loss: {test_loss:.6f}")

# Compute detailed metrics
print("\n  Computing detailed metrics...")
test_metrics = compute_detailed_metrics(model, test_sequences, test_target_reprs, device, 
                                       num_node_features, gatgru_config['output_dim'])

print(f"\n  Detailed Test Metrics:")
print(f"    MAE: {test_metrics['mae']:.6f}")
print(f"    RMSE: {test_metrics['rmse']:.6f}")
print(f"    Standardized RMSE: {test_metrics['standardized_rmse']:.6f}")
print(f"    Target Mean: {test_metrics['target_mean']:.6f}")
print(f"    Target Std: {test_metrics['target_std']:.6f}")
print(f"    Prediction Mean: {test_metrics['prediction_mean']:.6f}")
print(f"    Prediction Std: {test_metrics['prediction_std']:.6f}")

# Save metrics
print("\n[7/7] Saving training metrics...")

# Prepare metrics dictionary
metrics = {
    'training_info': {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_training_time_seconds': training_time,
        'total_epochs_trained': len(train_losses),
        'early_stopped': epochs_without_improvement >= patience,
        'device': str(device)
    },
    'model_info': {
        'model_type': 'GAT-GRU',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'node_features': num_node_features,
        'edge_features': num_edge_features,
        'num_nodes': num_nodes,
        'architecture': {
            'hidden_dim': gatgru_config['hidden_dim'],
            'num_gat_layers': gatgru_config['num_gat_layers'],
            'num_gru_layers': gatgru_config['num_gru_layers'],
            'num_heads': gatgru_config['num_heads'],
            'output_dim': gatgru_config['output_dim'],
            'aggregation': gatgru_config['aggregation'],
            'bidirectional': gatgru_config['bidirectional'],
            'use_edge_attr_in_gat': gatgru_config['use_edge_attr_in_gat']
        }
    },
    'data_split': {
        'total_snapshots': num_snapshots,
        'train_snapshots': len(train_graphs),
        'val_snapshots': len(val_graphs),
        'test_snapshots': len(test_graphs),
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'test_sequences': len(test_sequences),
        'sequence_length': sequence_length
    },
    'losses': {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'best_epoch': checkpoint['epoch'] + 1
    },
    'training_dynamics': {
        'learning_rates': learning_rates,
        'epoch_times_seconds': epoch_times,
        'avg_epoch_time_seconds': sum(epoch_times) / len(epoch_times) if epoch_times else 0
    },
    'test_metrics': {
        'mae': test_metrics['mae'],
        'rmse': test_metrics['rmse'],
        'standardized_rmse': test_metrics['standardized_rmse'],
        'target_mean': test_metrics['target_mean'],
        'target_std': test_metrics['target_std'],
        'prediction_mean': test_metrics['prediction_mean'],
        'prediction_std': test_metrics['prediction_std']
    },
    'hyperparameters': {
        'optimizer': gatgru_config['optimizer'],
        'learning_rate': gatgru_config['learning_rate'],
        'weight_decay': gatgru_config['weight_decay'],
        'loss_function': gatgru_config['loss_function'],
        'gradient_clip': gatgru_config['gradient_clip'],
        'dropout': gatgru_config['dropout'],
        'scheduler_type': gatgru_config['scheduler_type'] if gatgru_config['use_scheduler'] else None,
        'scheduler_factor': gatgru_config['scheduler_factor'] if gatgru_config['use_scheduler'] else None,
        'scheduler_patience': gatgru_config['scheduler_patience'] if gatgru_config['use_scheduler'] else None
    }
}

# Save to JSON
metrics_path = config['paths']['gatgru_metrics_path']
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"  Metrics saved to {metrics_path}")

print("\n" + "="*80)
print("Training completed successfully!")
print("="*80)
print(f"\nSummary:")
print(f"  Best validation loss: {best_val_loss:.6f} (epoch {checkpoint['epoch'] + 1})")
print(f"  Test loss: {test_loss:.6f}")
print(f"  Test MAE: {test_metrics['mae']:.6f}")
print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
print(f"  Total training time: {training_time/60:.2f} minutes")
print(f"  Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds")

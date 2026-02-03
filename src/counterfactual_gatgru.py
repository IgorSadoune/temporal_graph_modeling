"""
Counterfactual Analysis Script for Temporal Graph Models (GATGRU)

This script performs counterfactual analysis by simulating a shock to a specific node
(Montreal) in the maritime network and analyzing the propagation effects on other nodes.

Methodology:
1. Load the trained GATGRU model
2. Select test sequences from the test set
3. Apply intervention: INCREASE dwell time at Montreal by 50% in the last graph of input sequence
4. Generate predictions for t+1 using both:
   - Original sequence (factual)
   - Modified sequence (counterfactual)
5. Compare node features between factual and counterfactual predictions
6. Analyze the effect of the shock on all nodes in the network

Output:
- Counterfactual analysis results showing the impact of the intervention
- Node-level comparison of factual vs counterfactual states
- Metrics quantifying the propagation of the shock through the network
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import copy

from modules.gatgru import TemporalGraphGATGRU

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
        timestep = i + seq_length
        sequences.append((input_seq, target, timestep))
    return sequences


def apply_intervention(graph, port_to_idx, idx_to_port, intervention_port='montreal', dwell_time_increase=0.50):
    """
    Apply intervention to a graph by INCREASING dwell time at a specific port.
    
    Args:
        graph: PyTorch Geometric Data object
        port_to_idx: Dictionary mapping port names to node indices (full network)
        idx_to_port: Dictionary mapping node indices to port names (full network)
        intervention_port: Name of the port to intervene on (default: 'montreal')
        dwell_time_increase: Fraction to INCREASE dwell time by (default: 0.50 = 50%)
        
    Returns:
        Modified graph with intervention applied, or None if port not in this graph
    """
    # Create a deep copy to avoid modifying the original
    modified_graph = copy.deepcopy(graph)
    
    # Find the node index for the intervention port in the full network
    intervention_port_lower = intervention_port.lower()
    full_network_node_idx = None
    found_port_name = None
    
    for port_name, idx in port_to_idx.items():
        if intervention_port_lower in port_name.lower():
            full_network_node_idx = idx
            found_port_name = port_name
            break
    
    if full_network_node_idx is None:
        raise ValueError(f"Port '{intervention_port}' not found in port_to_idx mapping")
    
    # Check if this graph has node_mapping attribute (from k-core filtering)
    if hasattr(graph, 'node_mapping'):

        if full_network_node_idx not in graph.node_mapping:
            # Port not in this filtered graph
            return None, None, None, None, found_port_name
        
        graph_node_idx = graph.node_mapping[full_network_node_idx]
    elif hasattr(graph, 'original_indices'):
        # Alternative: check original_indices list
        if full_network_node_idx not in graph.original_indices:
            return None, None, None, None, found_port_name
        
        # Find position in sorted list
        graph_node_idx = graph.original_indices.index(full_network_node_idx)
    else:
        # Graph has not been filtered, use full network index directly
        graph_node_idx = full_network_node_idx
        
        # Verify the index is valid for this graph
        if graph_node_idx >= modified_graph.x.shape[0]:
            return None, None, None, None, found_port_name
    
    # Node features: [inbound_count, outbound_count, avg_dwell_time, avg_dwt, avg_sog]
    # Dwell time is at index 2
    dwell_time_idx = 2
    
    # Apply intervention: INCREASE dwell time by specified percentage
    original_dwell_time = modified_graph.x[graph_node_idx, dwell_time_idx].item()
    new_dwell_time = original_dwell_time * (1 + dwell_time_increase)  # Changed from (1 - reduction) to (1 + increase)
    modified_graph.x[graph_node_idx, dwell_time_idx] = new_dwell_time
    
    return modified_graph, graph_node_idx, original_dwell_time, new_dwell_time, found_port_name


def predict_with_sequence(model, sequence, device):
    """
    Generate prediction for a given sequence.
    
    Args:
        model: Trained model
        sequence: List of graph snapshots
        device: Device to run inference on
        
    Returns:
        Predicted graph representation
    """
    model.eval()
    
    with torch.no_grad():
        # Move sequence to device
        sequence = [graph.to(device) for graph in sequence]
        
        # Forward pass
        output, _ = model(sequence)
        
        # Get prediction for last timestep
        prediction = output[:, -1, :]  # [batch_size, output_dim]
        
    return prediction


def compute_node_level_effects(factual_pred, counterfactual_pred, graph, idx_to_port, intervention_node_idx):
    """
    Compute node-level effects of the intervention.
    
    Args:
        factual_pred: Factual prediction [1, output_dim]
        counterfactual_pred: Counterfactual prediction [1, output_dim]
        graph: The graph object (to get node mapping if available)
        idx_to_port: Dictionary mapping full network indices to port names
        intervention_node_idx: Index of the node where intervention was applied (in filtered graph)
        
    Returns:
        Dictionary containing node-level analysis
    """
    # Compute absolute and relative differences
    absolute_diff = (counterfactual_pred - factual_pred).cpu().numpy().squeeze()
    
    # Avoid division by zero
    epsilon = 1e-8
    factual_values = factual_pred.cpu().numpy().squeeze()
    relative_diff = (absolute_diff / (np.abs(factual_values) + epsilon)) * 100
    
    # Get node mapping if available
    if hasattr(graph, 'node_mapping'):
        # node_mapping is old_idx -> new_idx, we need reverse
        reverse_mapping = {v: k for k, v in graph.node_mapping.items()}    
    elif hasattr(graph, 'original_indices'):
        # Create mapping from filtered index to original index
        reverse_mapping = {i: orig_idx for i, orig_idx in enumerate(graph.original_indices)}    
    else:
        # No filtering, identity mapping
        reverse_mapping = {i: i for i in range(len(absolute_diff))}
    

    # Create node-level results
    node_effects = []
    
    for filtered_idx in range(len(absolute_diff)):
        # Get full network index
        full_idx = reverse_mapping.get(filtered_idx, filtered_idx)
        port_name = idx_to_port.get(full_idx, f"Node_{full_idx}")
        
        node_effects.append({
            'node_idx': int(filtered_idx),
            'full_network_idx': int(full_idx),
            'port_name': port_name,
            'is_intervention_node': filtered_idx == intervention_node_idx,
            'factual_value': float(factual_values[filtered_idx]),
            'counterfactual_value': float(counterfactual_pred.cpu().numpy().squeeze()[filtered_idx]),
            'absolute_difference': float(absolute_diff[filtered_idx]),
            'relative_difference_percent': float(relative_diff[filtered_idx])
        })
    
    # Sort by absolute difference (descending)
    node_effects_sorted = sorted(node_effects, key=lambda x: abs(x['absolute_difference']), reverse=True)
    
    return node_effects_sorted


def compute_aggregate_metrics(node_effects, intervention_node_idx):
    """
    Compute aggregate metrics for the counterfactual analysis.
    
    Args:
        node_effects: List of node-level effects
        intervention_node_idx: Index of the intervention node
        
    Returns:
        Dictionary of aggregate metrics
    """
    # Separate intervention node from other nodes
    intervention_node = [n for n in node_effects if n['node_idx'] == intervention_node_idx][0]
    other_nodes = [n for n in node_effects if n['node_idx'] != intervention_node_idx]
    
    # Compute metrics for other nodes (spillover effects)
    abs_diffs = [abs(n['absolute_difference']) for n in other_nodes]
    rel_diffs = [abs(n['relative_difference_percent']) for n in other_nodes]
    
    metrics = {
        'intervention_node': {
            'port_name': intervention_node['port_name'],
            'absolute_difference': intervention_node['absolute_difference'],
            'relative_difference_percent': intervention_node['relative_difference_percent']
        },
        'spillover_effects': {
            'num_affected_nodes': len(other_nodes),
            'mean_absolute_difference': float(np.mean(abs_diffs)),
            'median_absolute_difference': float(np.median(abs_diffs)),
            'max_absolute_difference': float(np.max(abs_diffs)),
            'std_absolute_difference': float(np.std(abs_diffs)),
            'mean_relative_difference_percent': float(np.mean(rel_diffs)),
            'median_relative_difference_percent': float(np.median(rel_diffs)),
            'max_relative_difference_percent': float(np.max(rel_diffs))
        },
        'top_affected_nodes': [
            {
                'port_name': n['port_name'],
                'absolute_difference': n['absolute_difference'],
                'relative_difference_percent': n['relative_difference_percent']
            }
            for n in other_nodes[:3]  # Top 3 most affected nodes
        ]
    }
    
    return metrics


# Parse command line arguments
parser = argparse.ArgumentParser(description='Counterfactual analysis on trained GATGRU model')
parser.add_argument('--config', type=str, default='src/config.yaml',
                    help='Path to configuration file (default: config.yaml)')
parser.add_argument('--intervention_port', type=str, default='montreal',
                    help='Port to apply intervention on (default: montreal)')
parser.add_argument('--dwell_time_increase', type=float, default=0.50,
                    help='Fraction to INCREASE dwell time by (default: 0.50 = 50%%)')
parser.add_argument('--num_sequences', type=int, default=None,
                    help='Number of test sequences to analyze (default: all)')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

print("="*80)
print("Counterfactual Analysis - GATGRU (INCREASED DWELL TIME)")
print("="*80)
print(f"\nIntervention: INCREASE dwell time at {args.intervention_port.upper()} by {args.dwell_time_increase*100:.0f}%")
print("="*80)

# Load data
print("\n[1/7] Loading temporal graph sequence...")
graph_sequence_path = config['paths']['graph_sequence_path']

if config['test'] == 1:
    print('TEST CONFIG')
    temporal_graphs_data = torch.load(graph_sequence_path, weights_only=False)[:100]
else:
    temporal_graphs_data = torch.load(graph_sequence_path, weights_only=False)

# Load port mappings (from the first graph)
first_graph = temporal_graphs_data[0]
num_nodes = first_graph.x.shape[0]

print("\n[2/7] Loading port mappings...")

# Load the processed AIS data to get port mappings
processed_data_path = config['paths']['processed_ais_data_path']
ais_data = pd.read_parquet(processed_data_path)

# Recreate port mappings
unique_ports = pd.concat([
    ais_data['origin_port'],
    ais_data['destination_port']
]).unique()

port_to_idx = {port_name: idx for idx, port_name in enumerate(sorted(unique_ports))}
idx_to_port = {idx: port_name for port_name, idx in port_to_idx.items()}

print(f"  Total ports in network: {len(port_to_idx)}")
print(f"  Intervention port: {args.intervention_port}")

# Verify intervention port exists
intervention_port_lower = args.intervention_port.lower()
found_port = None
for port_name in port_to_idx.keys():
    if intervention_port_lower in port_name.lower():
        found_port = port_name
        break

if found_port is None:
    print(f"\n  ✗ Error: Port '{args.intervention_port}' not found in the network")
    print(f"  Available ports containing '{args.intervention_port}':")
    for port_name in sorted(port_to_idx.keys()):
        if args.intervention_port.lower() in port_name.lower():
            print(f"    - {port_name}")
    exit(1)
else:
    print(f"  ✓ Found intervention port: {found_port}")

# Split data (use same split as training)
print("\n[3/7] Splitting data chronologically...")
train_ratio = config['training']['train_split']
val_ratio = config['training']['val_split']

num_snapshots = len(temporal_graphs_data)
train_size = int(num_snapshots * train_ratio)
val_size = int(num_snapshots * val_ratio)

# Use test set for counterfactual analysis
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
print("\n[4/7] Creating temporal sequences...")
sequence_length = config['training']['input_sequence_length']
test_sequences = create_sequences(test_graphs, sequence_length)

# Limit number of sequences if specified
if args.num_sequences is not None:
    test_sequences = test_sequences[:args.num_sequences]

print(f"  Sequence length: {sequence_length}")
print(f"  Test sequences to analyze: {len(test_sequences)}")

# Diagnostic: Check which ports are actually present in test sequences
print("\n  Checking ports present in test sequences...")
ports_in_sequences = set()
for input_seq, target, timestep in test_sequences:
    for graph in input_seq:
        # Check which original node indices are in this graph
        if hasattr(graph, 'original_indices'):
            # These are the node indices from the full network that are in this filtered graph
            for orig_idx in graph.original_indices:
                if orig_idx in idx_to_port:
                    ports_in_sequences.add(idx_to_port[orig_idx])
        elif hasattr(graph, 'node_mapping'):
            # node_mapping is old_idx -> new_idx
            for orig_idx in graph.node_mapping.keys():
                if orig_idx in idx_to_port:
                    ports_in_sequences.add(idx_to_port[orig_idx])

if len(ports_in_sequences) > 0:
    print(f"  Ports found in test sequences: {len(ports_in_sequences)}")
    if found_port not in ports_in_sequences:
        print(f"\n  ⚠ WARNING: Intervention port '{found_port}' not found in any test sequences!")
        print(f"  This port was likely filtered out by k-core decomposition.")
        print(f"\n  Available ports in test sequences:")
        
        sorted_ports = sorted(list(ports_in_sequences))
        
        # Display all available ports with numbers
        for i, port in enumerate(sorted_ports):
            print(f"    {i+1}. {port}")
        
        # Ask user to choose
        print(f"\n  Please select a port for intervention:")
        print(f"    - Enter a number (1-{len(sorted_ports)}) to select from the list above")
        print(f"    - Enter a port name directly")
        print(f"    - Press Ctrl+C to exit")
        
        while True:
            try:
                user_input = input("\n  Your choice: ").strip()
                
                # Check if input is a number
                if user_input.isdigit():
                    choice_idx = int(user_input) - 1
                    if 0 <= choice_idx < len(sorted_ports):
                        found_port = sorted_ports[choice_idx]
                        print(f"\n  ✓ Selected port: {found_port}")
                        break
                    else:
                        print(f"  ✗ Invalid number. Please enter a number between 1 and {len(sorted_ports)}")
                else:
                    # Check if the entered port name exists
                    user_input_lower = user_input.lower()
                    matching_ports = [p for p in sorted_ports if user_input_lower in p.lower()]
                    
                    if len(matching_ports) == 1:
                        found_port = matching_ports[0]
                        print(f"\n  ✓ Selected port: {found_port}")
                        break
                    elif len(matching_ports) > 1:
                        print(f"  ✗ Multiple ports match '{user_input}':")
                        for i, p in enumerate(matching_ports):
                            print(f"      {i+1}. {p}")
                        print(f"  Please be more specific or use the number from the list above")
                    else:
                        print(f"  ✗ Port '{user_input}' not found in available ports")
                        print(f"  Please choose from the list above")
            except KeyboardInterrupt:
                print("\n\n  Exiting...")
                exit(0)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                print(f"  Please try again")
    else:
        print(f"  ✓ Intervention port '{found_port}' is present in test sequences")
else:
    print(f"  ⚠ Warning: Could not determine which ports are in sequences")
    print(f"  Will check for port presence during analysis...")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")

# Update the intervention port to use the selected one
intervention_port = found_port
print(f"\n  Final intervention port: {intervention_port}")
print(f"  Dwell time INCREASE: {args.dwell_time_increase*100:.0f}%")

# Load GATGRU model
print("\n[5/7] Loading trained GATGRU model...")
print("-" * 80)

# First load checkpoint to get the actual model configuration
checkpoint_path = config['paths']['gatgru_best_model_path']
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract model configuration from checkpoint if available
if 'config' in checkpoint:
    # Use the configuration that was used during training
    gatgru_config = checkpoint['config']['gatgru']
    print("  Using model configuration from checkpoint")
else:
    # Fall back to config file
    gatgru_config = config['gatgru']
    print("  Using model configuration from config file")

model = TemporalGraphGATGRU(
    node_feature_dim=num_node_features,
    edge_feature_dim=num_edge_features,
    hidden_dim=gatgru_config['hidden_dim'],
    num_gat_layers=gatgru_config['num_gat_layers'],
    num_heads=gatgru_config['num_heads'],
    num_gru_layers=gatgru_config['num_gru_layers'],
    output_dim=gatgru_config['output_dim'],
    aggregation=gatgru_config['aggregation'],
    dropout=gatgru_config['dropout']
).to(device)

# Load model weights
model.load_state_dict(checkpoint['model_state_dict'])

print(f"  ✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
print(f"  ✓ Training val_loss: {checkpoint['val_loss']:.6f}")
print(f"  ✓ Model hidden_dim: {gatgru_config['hidden_dim']}")
print(f"  ✓ Model output_dim: {gatgru_config['output_dim']}")

# Perform counterfactual analysis
print("\n[6/7] Performing counterfactual analysis...")
print("="*80)

all_results = []
skipped_sequences = 0

for seq_idx, (input_seq, target, timestep) in enumerate(tqdm(test_sequences, desc="  Analyzing sequences")):
    # Apply intervention to the last graph
    last_graph = input_seq[-1]
    intervention_result = apply_intervention(
        last_graph, 
        port_to_idx,
        idx_to_port,
        intervention_port,  # Use the selected intervention port
        args.dwell_time_increase  # Changed from dwell_time_reduction to dwell_time_increase
    )
    
    # Check if intervention was successful
    if intervention_result[0] is None:
        # Port not in this graph, skip this sequence
        skipped_sequences += 1
        continue
    
    modified_last_graph, intervention_node_idx, original_dwell, new_dwell, found_port_name = intervention_result
    
    # Store idx_to_port in the graph for later use
    last_graph.idx_to_port = idx_to_port
    if hasattr(last_graph, 'node_mapping'):
        modified_last_graph.node_mapping = last_graph.node_mapping
    if hasattr(last_graph, 'original_indices'):
        modified_last_graph.original_indices = last_graph.original_indices
    
    # Create factual sequence (original)
    factual_seq = [graph.to(device) for graph in input_seq]
    
    # Create counterfactual sequence (with intervention on last graph)
    counterfactual_seq = [graph.to(device) for graph in input_seq[:-1]]
    counterfactual_seq.append(modified_last_graph.to(device))
    
    # Generate predictions
    factual_pred = predict_with_sequence(model, factual_seq, device)
    counterfactual_pred = predict_with_sequence(model, counterfactual_seq, device)
    
    # Compute node-level effects
    node_effects = compute_node_level_effects(
        factual_pred, 
        counterfactual_pred, 
        last_graph,
        idx_to_port,
        intervention_node_idx
    )
    
    # Compute aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(node_effects, intervention_node_idx)
    
    # Store results
    all_results.append({
        'sequence_idx': seq_idx,
        'timestep': int(timestep),
        'intervention': {
            'port_name': found_port_name,
            'node_idx': int(intervention_node_idx),
            'original_dwell_time': float(original_dwell),
            'new_dwell_time': float(new_dwell),
            'increase_percent': float(args.dwell_time_increase * 100)  # Changed from reduction_percent
        },
        'aggregate_metrics': aggregate_metrics,
        'node_effects': node_effects
    })

if skipped_sequences > 0:
    print(f"\n  Note: Skipped {skipped_sequences} sequences where {intervention_port} was not present in the graph")

# Compute overall statistics across all sequences
print("\n[7/7] Computing overall statistics...")
print("="*80)

if len(all_results) == 0:
    print(f"\n  ✗ Error: No sequences could be analyzed.")
    print(f"  The intervention port '{intervention_port}' was not found in any test sequences.")
    print(f"  This likely means the port was filtered out by k-core decomposition.")
    print(f"\n  Suggestions:")
    print(f"    1. Try a different intervention port")
    print(f"    2. Check if the port exists in the filtered network")
    print(f"    3. Adjust k-core parameters in config.yaml")
    exit(1)

# Aggregate metrics across all sequences
overall_spillover_metrics = {
    'mean_absolute_difference': [],
    'median_absolute_difference': [],
    'max_absolute_difference': [],
    'mean_relative_difference_percent': [],
    'median_relative_difference_percent': [],
    'max_relative_difference_percent': []
}

for result in all_results:
    spillover = result['aggregate_metrics']['spillover_effects']
    for key in overall_spillover_metrics.keys():
        overall_spillover_metrics[key].append(spillover[key])

# Compute statistics
overall_statistics = {
    'num_sequences_analyzed': len(all_results),
    'intervention': {
        'port_name': all_results[0]['intervention']['port_name'] if len(all_results) > 0 else intervention_port,
        'dwell_time_increase_percent': float(args.dwell_time_increase * 100)  # Changed from reduction
    },
    'spillover_effects_summary': {
        key: {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        for key, values in overall_spillover_metrics.items()
    }
}

# Compile final results
final_results = {
    'overall_statistics': overall_statistics,
    'sequence_results': all_results
}

# Save results
print("\nSaving counterfactual analysis results...")
counterfactual_results_path = config['paths'].get('counterfactual_gatgru_results_path', '../outputs/counterfactual_gatgru_results.json')

# Modify output filename to indicate increased dwell time
output_path = Path(counterfactual_results_path)
new_filename = output_path.stem + '_increased_dwell' + output_path.suffix
counterfactual_results_path = output_path.parent / new_filename

# Create output directory if it doesn't exist
output_dir = Path(counterfactual_results_path).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Save to JSON
with open(counterfactual_results_path, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"  ✓ Results saved to {counterfactual_results_path}")

# Print summary
print("\n" + "="*80)
print("Counterfactual Analysis Summary (INCREASED DWELL TIME)")
print("="*80)

print(f"\nIntervention Details:")
print(f"  Port: {overall_statistics['intervention']['port_name']}")
print(f"  Dwell time INCREASE: {args.dwell_time_increase*100:.0f}%")
print(f"  Sequences analyzed: {len(all_results)}")

print(f"\nSpillover Effects (averaged across all sequences):")
print("-" * 80)

spillover_summary = overall_statistics['spillover_effects_summary']

print(f"\nAbsolute Differences:")
print(f"  Mean: {spillover_summary['mean_absolute_difference']['mean']:.6f} "
      f"(±{spillover_summary['mean_absolute_difference']['std']:.6f})")
print(f"  Median: {spillover_summary['median_absolute_difference']['mean']:.6f}")
print(f"  Max: {spillover_summary['max_absolute_difference']['mean']:.6f}")

print(f"\nRelative Differences (%):")
print(f"  Mean: {spillover_summary['mean_relative_difference_percent']['mean']:.2f}% "
      f"(±{spillover_summary['mean_relative_difference_percent']['std']:.2f}%)")
print(f"  Median: {spillover_summary['median_relative_difference_percent']['mean']:.2f}%")
print(f"  Max: {spillover_summary['max_relative_difference_percent']['mean']:.2f}%")

# Show top affected nodes from first sequence as example
print(f"\nTop 10 Most Affected Nodes (Example from first sequence):")
print("-" * 80)
print(f"{'Rank':<6} {'Port Name':<30} {'Abs. Diff.':<15} {'Rel. Diff. (%)':<15}")
print("-" * 80)

top_nodes = all_results[0]['node_effects'][:10]  # Get top 10 instead of just top 3
for rank, node in enumerate(top_nodes, 1):
    print(f"{rank:<6} {node['port_name']:<30} {node['absolute_difference']:<15.6f} "
          f"{node['relative_difference_percent']:<15.2f}")

print("\n" + "="*80)
print("Counterfactual analysis completed successfully!")
print("="*80)

print(f"\nInterpretation:")
print(f"  - A {args.dwell_time_increase*100:.0f}% INCREASE in dwell time at {overall_statistics['intervention']['port_name']}")
print(f"  - Simulates congestion/delays propagating through the network")
print(f"  - Average spillover effect: {spillover_summary['mean_absolute_difference']['mean']:.6f}")
print(f"  - Results saved for detailed analysis and visualization")
print(f"\nNote: Positive differences indicate increased congestion at other ports")
print(f"      Negative differences indicate potential rerouting effects")
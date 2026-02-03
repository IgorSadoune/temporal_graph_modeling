"""
"""

import yaml
import torch
import pandas as pd
from modules.temporal_graphs import TemporalGraphBuilder

import warnings
warnings.filterwarnings('ignore')

# Load configuration
with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load ais data
processed_ais_data_path = config['paths']['processed_ais_data_path']
processed_ais_data = pd.read_parquet(processed_ais_data_path)

# Build temporal graph sequence + k-core filtering
k_core = config['graph']['k_core']
h_threshold = config['graph']['h_threshold']
temporal_graph_builder = TemporalGraphBuilder(data=processed_ais_data, k=k_core, h=h_threshold)
temporal_graphs_data = temporal_graph_builder.build_temporal_graphs()

# Save the list of graphs
output_path = config['paths']['graph_sequence_path']
torch.save(temporal_graphs_data, output_path)
print('Saved graph_sequence')

# Get list of active nodes
active_nodes = temporal_graph_builder.get_active_nodes()
port_to_idx, idx_to_port = temporal_graph_builder.get_port_mapping()
active_ports = [idx_to_port[node] for node in active_nodes]
print(active_ports)

# Get number of instances
number_of_instances = temporal_graph_builder.get_num_snapshots()
print(f"Number of instances: {number_of_instances}")

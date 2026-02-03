"""
Graph WaveNet module for temporal graph sequence modeling.

This module implements a Graph WaveNet-based model for processing temporal graph sequences.
Graph WaveNet combines graph convolutions with dilated causal convolutions to capture both
spatial and temporal dependencies in dynamic graphs.

The model processes sequences of graph snapshots where each snapshot contains:
- Node features (e.g., inbound/outbound flows, dwell time, deadweight)
- Edge features (e.g., speed, trip duration)
- Graph structure (edge indices)

Architecture:
1. Graph Convolution Layers: Learn spatial dependencies on the graph structure
2. Dilated Causal Convolutions: Capture temporal patterns with increasing receptive fields
3. Skip Connections: Aggregate information from multiple temporal scales
4. Output Layer: Generate predictions for next timestep

Key Features:
- Adaptive adjacency matrix learning
- Dilated causal convolutions for efficient temporal modeling
- Multi-scale temporal feature extraction
- Residual and skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from typing import Tuple, Optional, List


class GraphConvLayer(nn.Module):
    """
    Graph convolution layer for spatial feature extraction.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        dropout (float): Dropout rate
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(GraphConvLayer, self).__init__()
        
        self.gcn = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        x = self.gcn(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class DilatedCausalConv(nn.Module):
    """
    Dilated causal convolution for temporal modeling.
    
    Causal convolutions ensure that predictions at time t only depend on
    observations up to time t. Dilation increases the receptive field
    exponentially without increasing parameters.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        kernel_size (int): Size of convolution kernel
        dilation (int): Dilation factor
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super(DilatedCausalConv, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal padding to ensure causality
        self.padding = (kernel_size - 1) * dilation
        
        # Temporal convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dilated causal convolution.
        
        Args:
            x: Input tensor [batch_size, in_channels, seq_len]
            
        Returns:
            Output tensor [batch_size, out_channels, seq_len]
        """
        # Apply convolution
        x = self.conv(x)
        
        # Remove future information (causal)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        
        x = self.dropout(x)
        return x


class WaveNetBlock(nn.Module):
    """
    WaveNet block combining dilated causal convolution with gated activation.
    
    Uses gated activation units (tanh and sigmoid gates) for better gradient flow
    and feature learning.
    
    Args:
        residual_channels (int): Number of residual channels
        skip_channels (int): Number of skip connection channels
        kernel_size (int): Convolution kernel size
        dilation (int): Dilation factor
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super(WaveNetBlock, self).__init__()
        
        # Dilated causal convolutions for filter and gate
        self.filter_conv = DilatedCausalConv(
            residual_channels,
            residual_channels,
            kernel_size,
            dilation,
            dropout
        )
        
        self.gate_conv = DilatedCausalConv(
            residual_channels,
            residual_channels,
            kernel_size,
            dilation,
            dropout
        )
        
        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        
        self.batch_norm = nn.BatchNorm1d(residual_channels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through WaveNet block.
        
        Args:
            x: Input tensor [batch_size, residual_channels, seq_len]
            
        Returns:
            Tuple of (residual output, skip output)
        """
        residual = x
        
        # Gated activation unit
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x = filter_out * gate_out
        
        # Skip connection
        skip = self.skip_conv(x)
        
        # Residual connection
        x = self.residual_conv(x)
        x = x + residual
        x = self.batch_norm(x)
        
        return x, skip


class GraphWaveNetEncoder(nn.Module):
    """
    Graph WaveNet encoder combining graph convolutions with WaveNet blocks.
    
    This encoder first applies graph convolutions to learn spatial features,
    then uses WaveNet blocks to capture temporal dependencies with multi-scale
    receptive fields.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of hidden representations
        num_gcn_layers (int): Number of graph convolution layers
        num_wavenet_blocks (int): Number of WaveNet blocks
        kernel_size (int): Convolution kernel size
        aggregation (str): Graph-level aggregation method
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_gcn_layers: int = 2,
        num_wavenet_blocks: int = 4,
        kernel_size: int = 2,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        super(GraphWaveNetEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_wavenet_blocks = num_wavenet_blocks
        self.aggregation = aggregation
        
        # Graph convolution layers for spatial encoding
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(node_feature_dim, hidden_dim, dropout))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim, dropout))
        
        # Edge feature encoder
        if edge_feature_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.edge_encoder = None
        
        # Graph-level pooling projection
        if aggregation == 'concat':
            self.pool_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Encode a batch of graphs using graph convolutions.
        
        Args:
            batch (Batch): PyTorch Geometric Batch object
            
        Returns:
            torch.Tensor: Graph embeddings [batch_size, hidden_dim]
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
        batch_idx = batch.batch
        
        # Apply graph convolution layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
        
        # Graph-level pooling
        if self.aggregation == 'mean':
            graph_repr = global_mean_pool(x, batch_idx)
        elif self.aggregation == 'sum':
            graph_repr = global_add_pool(x, batch_idx)
        elif self.aggregation == 'max':
            graph_repr = global_max_pool(x, batch_idx)
        elif self.aggregation == 'concat':
            mean_pool = global_mean_pool(x, batch_idx)
            max_pool = global_max_pool(x, batch_idx)
            sum_pool = global_add_pool(x, batch_idx)
            graph_repr = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
            graph_repr = self.pool_projection(graph_repr)
        else:
            graph_repr = global_mean_pool(x, batch_idx)
        
        # Incorporate edge information if available
        if self.edge_encoder is not None and edge_attr is not None:
            edge_embeddings = self.edge_encoder(edge_attr)
            edge_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=x.device)
            edge_batch = batch_idx[edge_index[0]]
            
            for i in range(batch.num_graphs):
                mask = edge_batch == i
                if mask.sum() > 0:
                    edge_repr[i] = edge_embeddings[mask].mean(dim=0)
            
            graph_repr = graph_repr + edge_repr
        
        graph_repr = self.dropout(graph_repr)
        
        return graph_repr


class TemporalGraphWaveNet(nn.Module):
    """
    Graph WaveNet model for temporal graph sequence modeling.
    
    This model combines graph convolutions for spatial encoding with WaveNet
    architecture for temporal modeling. It uses dilated causal convolutions
    to efficiently capture long-range temporal dependencies.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of hidden representations
        num_gcn_layers (int): Number of graph convolution layers
        num_wavenet_blocks (int): Number of WaveNet blocks
        num_layers_per_block (int): Number of layers in each WaveNet block
        kernel_size (int): Convolution kernel size
        output_dim (int): Dimension of output predictions
        aggregation (str): Graph aggregation method
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_gcn_layers: int = 2,
        num_wavenet_blocks: int = 4,
        num_layers_per_block: int = 2,
        kernel_size: int = 2,
        output_dim: int = 128,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        super(TemporalGraphWaveNet, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_wavenet_blocks = num_wavenet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.output_dim = output_dim
        
        # Graph encoder
        self.graph_encoder = GraphWaveNetEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gcn_layers,
            aggregation=aggregation,
            dropout=dropout
        )
        
        # Input projection
        self.input_projection = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # WaveNet blocks with increasing dilation
        self.wavenet_blocks = nn.ModuleList()
        for block_idx in range(num_wavenet_blocks):
            for layer_idx in range(num_layers_per_block):
                dilation = 2 ** layer_idx
                self.wavenet_blocks.append(
                    WaveNetBlock(
                        residual_channels=hidden_dim,
                        skip_channels=hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout
                    )
                )
        
        # Output layers
        self.output_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.output_conv2 = nn.Conv1d(hidden_dim, output_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        graph_sequence: list,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through the temporal graph WaveNet.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data/Batch objects
            hidden_state: Not used (for compatibility with LSTM/GRU interface)
            
        Returns:
            Tuple containing:
                - output: Predictions of shape [batch_size, seq_len, output_dim]
                - None: No hidden state (for compatibility)
        """
        batch_size = 1
        seq_len = len(graph_sequence)
        
        # Encode each graph in the sequence
        graph_embeddings = []
        for graph in graph_sequence:
            # If graph is not batched, create a batch-like wrapper
            if not isinstance(graph, Batch):
                # Manually add batch tensor for single graph
                graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)
                # Create a simple Batch-like wrapper
                class SingleGraphBatch:
                    def __init__(self, data):
                        self.x = data.x
                        self.edge_index = data.edge_index
                        self.edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                        self.batch = data.batch
                        self.num_graphs = 1
                        
                graph = SingleGraphBatch(graph)
            
            graph_emb = self.graph_encoder(graph)  # [num_graphs_in_batch, hidden_dim]
            graph_embeddings.append(graph_emb)
        
        # Stack graph embeddings into sequence
        # Shape: [batch_size, seq_len, hidden_dim]
        graph_sequence_tensor = torch.stack(graph_embeddings, dim=1)
        
        # Transpose for Conv1d: [batch_size, hidden_dim, seq_len]
        x = graph_sequence_tensor.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply WaveNet blocks and accumulate skip connections
        skip_connections = []
        for wavenet_block in self.wavenet_blocks:
            x, skip = wavenet_block(x)
            skip_connections.append(skip)
        
        # Sum all skip connections
        x = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Output layers
        x = F.relu(x)
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_conv2(x)
        
        # Transpose back: [batch_size, seq_len, output_dim]
        output = x.transpose(1, 2)
        
        return output, None
    
    def predict_next(
        self,
        graph_sequence: list,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Predict the next timestep given a sequence of graphs.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data/Batch objects
            hidden_state: Not used (for compatibility)
            
        Returns:
            Tuple containing:
                - prediction: Next timestep prediction [batch_size, output_dim]
                - None: No hidden state (for compatibility)
        """
        output, _ = self.forward(graph_sequence, hidden_state)
        
        # Return only the last timestep prediction
        next_prediction = output[:, -1, :]  # [batch_size, output_dim]
        
        return next_prediction, None

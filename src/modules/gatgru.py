"""
GAT-GRU module for temporal graph sequence modeling.

This module implements a GAT-GRU-based model for processing temporal graph sequences.
The model processes sequences of graph snapshots where each snapshot contains:
- Node features (e.g., inbound/outbound flows, dwell time, deadweight)
- Edge features (e.g., speed, trip duration)
- Graph structure (edge indices)

The GAT layers learn attention-based node representations that capture spatial dependencies,
while the GRU captures temporal dependencies across graph snapshots.

Architecture:
1. GAT Encoder: Uses Graph Attention Networks to learn node representations with attention
2. Graph Pooling: Aggregates node representations into graph-level embeddings
3. GRU Layers: Process the sequence of graph representations
4. Decoder: Maps GRU hidden states to output predictions
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from typing import Tuple, Optional


class GATGraphEncoder(nn.Module):
    """
    Encodes a graph snapshot using Graph Attention Networks (GAT).
    
    This encoder uses GAT layers to learn node representations with attention mechanisms,
    allowing the model to focus on important nodes and edges. The node representations
    are then aggregated into a graph-level embedding.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of hidden representations
        num_gat_layers (int): Number of GAT layers
        num_heads (int): Number of attention heads in GAT
        aggregation (str): Graph-level aggregation method ('mean', 'sum', 'max', or 'concat')
        dropout (float): Dropout rate for regularization
        edge_dim (int): Dimension for edge features in GAT (if None, edges not used in attention)
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        aggregation: str = 'mean',
        dropout: float = 0.1,
        edge_dim: Optional[int] = None
    ):
        super(GATGraphEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.edge_dim = edge_dim
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=node_feature_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True
            )
        )
        
        # Intermediate GAT layers
        for _ in range(num_gat_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True
                )
            )
        
        # Last GAT layer (average heads instead of concatenating)
        if num_gat_layers > 1:
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=False  # Average attention heads
                )
            )
        
        # Edge feature encoder (if edge_dim is None, we encode edges separately)
        if edge_dim is None and edge_feature_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.edge_encoder = None
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gat_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Graph-level pooling
        if aggregation == 'concat':
            # Concatenate mean, max, and sum pooling
            self.pool_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Encode a batch of graphs into graph-level representations using GAT.
        
        Args:
            batch (Batch): PyTorch Geometric Batch object containing multiple graphs
            
        Returns:
            torch.Tensor: Graph embeddings of shape [batch_size, hidden_dim]
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
        batch_idx = batch.batch
        
        # Process edge features if edge_dim is specified and edge_attr exists
        if self.edge_dim is not None and edge_attr is not None:
            # Edge features will be used directly in GAT attention
            pass
        elif self.edge_encoder is not None and edge_attr is not None:
            # Encode edge features separately (not used in attention)
            edge_embeddings = self.edge_encoder(edge_attr)
        else:
            edge_attr = None
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            if self.edge_dim is not None and edge_attr is not None:
                x = gat_layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = gat_layer(x, edge_index)
            
            # Apply batch normalization
            x = self.batch_norms[i](x)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.gat_layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        
        # Graph-level pooling
        if self.aggregation == 'mean':
            graph_repr = global_mean_pool(x, batch_idx)
        elif self.aggregation == 'sum':
            graph_repr = global_add_pool(x, batch_idx)
        elif self.aggregation == 'max':
            graph_repr = global_max_pool(x, batch_idx)
        elif self.aggregation == 'concat':
            # Concatenate multiple pooling strategies
            mean_pool = global_mean_pool(x, batch_idx)
            max_pool = global_max_pool(x, batch_idx)
            sum_pool = global_add_pool(x, batch_idx)
            graph_repr = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
            graph_repr = self.pool_projection(graph_repr)
        else:
            # Default to mean pooling
            graph_repr = global_mean_pool(x, batch_idx)
        
        # Incorporate edge information if encoded separately
        if self.edge_encoder is not None and edge_attr is not None:
            # Aggregate edge features per graph
            edge_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=x.device)
            edge_batch = batch_idx[edge_index[0]]
            
            for i in range(batch.num_graphs):
                mask = edge_batch == i
                if mask.sum() > 0:
                    edge_repr[i] = edge_embeddings[mask].mean(dim=0)
            
            # Combine node and edge representations
            graph_repr = graph_repr + edge_repr
        
        graph_repr = self.dropout(graph_repr)
        
        return graph_repr


class TemporalGraphGATGRU(nn.Module):
    """
    GAT-GRU-based model for temporal graph sequence modeling.
    
    This model processes sequences of graph snapshots using GAT for spatial encoding
    and GRU for temporal modeling. The GAT layers learn attention-based node representations,
    while the GRU captures temporal dependencies across snapshots.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of hidden representations
        num_gat_layers (int): Number of GAT layers
        num_gru_layers (int): Number of GRU layers
        num_heads (int): Number of attention heads in GAT
        output_dim (int): Dimension of output predictions
        aggregation (str): Graph aggregation method ('mean', 'sum', 'max', or 'concat')
        dropout (float): Dropout rate for regularization
        bidirectional (bool): Whether to use bidirectional GRU
        use_edge_attr_in_gat (bool): Whether to use edge attributes in GAT attention
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_gat_layers: int = 2,
        num_gru_layers: int = 2,
        num_heads: int = 4,
        output_dim: int = 128,
        aggregation: str = 'mean',
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_edge_attr_in_gat: bool = False
    ):
        super(TemporalGraphGATGRU, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_gru_layers = num_gru_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Determine edge_dim for GAT
        edge_dim = edge_feature_dim if use_edge_attr_in_gat and edge_feature_dim > 0 else None
        
        # GAT-based graph encoder
        self.graph_encoder = GATGraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            aggregation=aggregation,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Output decoder
        gru_output_dim = hidden_dim * self.num_directions
        self.decoder = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self,
        graph_sequence: list,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the temporal graph GAT-GRU.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data/Batch objects
                                  representing the temporal sequence
            hidden_state (Optional[torch.Tensor]): Initial hidden state (h_0) for GRU
            If None, initialized to zeros
            
        Returns:
            Tuple containing:
                - output: Predictions of shape [batch_size, seq_len, output_dim]
                - hidden_state: Final GRU hidden state (h_n)
        """
        batch_size = 1  # Number of sequences being processed
        seq_len = len(graph_sequence)
        
        # Encode each graph in the sequence using GAT
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
        
        # Pass through GRU
        gru_out, hidden_state = self.gru(graph_sequence_tensor, hidden_state)
        # gru_out shape: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Decode GRU outputs to predictions
        output = self.decoder(gru_out)
        # output shape: [batch_size, seq_len, output_dim]
        
        return output, hidden_state
    
    def predict_next(
        self,
        graph_sequence: list,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the next timestep given a sequence of graphs.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data/Batch objects
            hidden_state (Optional[torch.Tensor]): Initial GRU hidden state
            
        Returns:
            Tuple containing:
                - prediction: Next timestep prediction of shape [batch_size, output_dim]
                - hidden_state: Final GRU hidden state
        """
        output, hidden_state = self.forward(graph_sequence, hidden_state)
        
        # Return only the last timestep prediction
        next_prediction = output[:, -1, :]  # [batch_size, output_dim]
        
        return next_prediction, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize GRU hidden state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Initial hidden state of shape 
                         [num_layers * num_directions, batch_size, hidden_dim]
        """
        num_layers_total = self.num_gru_layers * self.num_directions
        return torch.zeros(num_layers_total, batch_size, self.hidden_dim, device=device)

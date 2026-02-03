"""
LSTM module for temporal graph sequence modeling.

This module implements an LSTM-based model for processing temporal graph sequences.
The model processes sequences of graph snapshots where each snapshot contains:
- Node features (e.g., inbound/outbound flows, dwell time, deadweight)
- Edge features (e.g., speed, trip duration)
- Graph structure (edge indices)

The LSTM captures temporal dependencies across graph snapshots and can be used for:
- Next-timestep prediction
- Sequence-to-sequence modeling
- Temporal pattern recognition in dynamic graphs

Architecture:
1. Graph Encoder: Aggregates node and edge features into a graph-level representation
2. LSTM Layers: Process the sequence of graph representations
3. Decoder: Maps LSTM hidden states to output predictions
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple, Optional


class GraphEncoder(nn.Module):
    """
    Encodes a graph snapshot into a fixed-size vector representation.
    
    This encoder aggregates node and edge features using various pooling strategies
    to create a graph-level embedding that captures the state of the network at a
    given timestep.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of the output graph embedding
        aggregation (str): Aggregation method ('mean', 'sum', 'max', or 'concat')
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        super(GraphEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine node and edge representations
        if aggregation == 'concat':
            self.combiner = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.combiner = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Encode a batch of graphs into graph-level representations.
        
        Args:
            batch (Batch): PyTorch Geometric Batch object containing multiple graphs
            
        Returns:
            torch.Tensor: Graph embeddings of shape [batch_size, hidden_dim]
        """
        # Encode node features
        node_embeddings = self.node_encoder(batch.x)  # [total_nodes, hidden_dim]
        
        # Aggregate node features per graph
        if self.aggregation == 'mean':
            node_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=batch.x.device)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                if mask.sum() > 0:
                    node_repr[i] = node_embeddings[mask].mean(dim=0)
        elif self.aggregation == 'sum':
            node_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=batch.x.device)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                if mask.sum() > 0:
                    node_repr[i] = node_embeddings[mask].sum(dim=0)
        elif self.aggregation == 'max':
            node_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=batch.x.device)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                if mask.sum() > 0:
                    node_repr[i] = node_embeddings[mask].max(dim=0)[0]
        else:  # concat will be handled below
            node_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=batch.x.device)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                if mask.sum() > 0:
                    node_repr[i] = node_embeddings[mask].mean(dim=0)
        
        # Encode edge features if they exist
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            edge_embeddings = self.edge_encoder(batch.edge_attr)  # [total_edges, hidden_dim]
            
            # Aggregate edge features per graph
            edge_repr = torch.zeros(batch.num_graphs, self.hidden_dim, device=batch.x.device)
            
            # Get edge batch assignment
            edge_batch = batch.batch[batch.edge_index[0]]
            
            for i in range(batch.num_graphs):
                mask = edge_batch == i
                if mask.sum() > 0:
                    if self.aggregation == 'mean' or self.aggregation == 'concat':
                        edge_repr[i] = edge_embeddings[mask].mean(dim=0)
                    elif self.aggregation == 'sum':
                        edge_repr[i] = edge_embeddings[mask].sum(dim=0)
                    elif self.aggregation == 'max':
                        edge_repr[i] = edge_embeddings[mask].max(dim=0)[0]
            
            # Combine node and edge representations
            if self.aggregation == 'concat':
                combined = torch.cat([node_repr, edge_repr], dim=1)
                graph_repr = self.combiner(combined)
            else:
                graph_repr = self.combiner(node_repr + edge_repr)
        else:
            # Only node features available
            graph_repr = self.combiner(node_repr)
        
        graph_repr = self.dropout(graph_repr)
        
        return graph_repr


class TemporalGraphLSTM(nn.Module):
    """
    LSTM-based model for temporal graph sequence modeling.
    
    This model processes sequences of graph snapshots using an LSTM to capture
    temporal dependencies. It can be used for various temporal graph tasks such as
    next-timestep prediction, anomaly detection, or forecasting.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of hidden representations
        num_layers (int): Number of LSTM layers
        output_dim (int): Dimension of output predictions
        aggregation (str): Graph aggregation method ('mean', 'sum', 'max', 'concat')
        dropout (float): Dropout rate for regularization
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        aggregation: str = 'mean',
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super(TemporalGraphLSTM, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            dropout=dropout
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Output decoder
        lstm_output_dim = hidden_dim * self.num_directions
        self.decoder = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self,
        graph_sequence: list,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the temporal graph LSTM.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data/Batch objects
                                  representing the temporal sequence
            hidden_state (Optional[Tuple]): Initial hidden state (h_0, c_0) for LSTM
                                           If None, initialized to zeros
            
        Returns:
            Tuple containing:
                - output: Predictions of shape [batch_size, seq_len, output_dim]
                - hidden_state: Final LSTM hidden state (h_n, c_n)
        """
        batch_size = 1  # Number of sequences being processed
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
        
        # Pass through LSTM
        lstm_out, hidden_state = self.lstm(graph_sequence_tensor, hidden_state)
        # lstm_out shape: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Decode LSTM outputs to predictions
        output = self.decoder(lstm_out)
        # output shape: [batch_size, seq_len, output_dim]
        
        return output, hidden_state
    
    def predict_next(
        self,
        graph_sequence: list,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the next timestep given a sequence of graphs.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data/Batch objects
            hidden_state (Optional[Tuple]): Initial LSTM hidden state
            
        Returns:
            Tuple containing:
                - prediction: Next timestep prediction of shape [batch_size, output_dim]
                - hidden_state: Final LSTM hidden state
        """
        output, hidden_state = self.forward(graph_sequence, hidden_state)
        
        # Return only the last timestep prediction
        next_prediction = output[:, -1, :]  # [batch_size, output_dim]
        
        return next_prediction, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) with shape [num_layers * num_directions, batch_size, hidden_dim]
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        return (h_0, c_0)


class TemporalGraphLSTMPredictor(nn.Module):
    """
    Wrapper class for temporal graph prediction tasks.
    
    This class provides a high-level interface for training and inference with
    the TemporalGraphLSTM model. It handles sequence windowing and prediction logic.
    
    Args:
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        hidden_dim (int): Dimension of hidden representations
        num_layers (int): Number of LSTM layers
        output_dim (int): Dimension of output predictions
        sequence_length (int): Length of input sequences for prediction
        aggregation (str): Graph aggregation method
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        sequence_length: int,
        aggregation: str = 'mean',
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super(TemporalGraphLSTMPredictor, self).__init__()
        
        self.sequence_length = sequence_length
        
        self.model = TemporalGraphLSTM(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            aggregation=aggregation,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def forward(self, graph_sequence: list) -> torch.Tensor:
        """
        Forward pass for prediction.
        
        Args:
            graph_sequence (list): List of graph snapshots
            
        Returns:
            torch.Tensor: Predictions for the sequence
        """
        output, _ = self.model(graph_sequence)
        return output
    
    def predict_next_timestep(self, graph_sequence: list) -> torch.Tensor:
        """
        Predict the next timestep given a sequence.
        
        Args:
            graph_sequence (list): List of graph snapshots (length = sequence_length)
            
        Returns:
            torch.Tensor: Prediction for the next timestep
        """
        prediction, _ = self.model.predict_next(graph_sequence)
        return prediction

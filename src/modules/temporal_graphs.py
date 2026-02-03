"""
This module processes AIS (Automatic Identification System) Origin-Destination data from a Parquet file
and generates a PyTorch Geometric temporal graph dataset suitable for temporal graph neural network modeling.

The module creates daily graph snapshots from the available days in the input data, where:
- Nodes represent ports
- Edges represent trips between ports

Node Features:
- Inbound flows: Number/volume of incoming trips
- Outbound flows: Number/volume of outgoing trips
- Dwell time: Average time vessels spend at the port
- Deadweight: Sum deadweight tonnage of vessels
- Speed over ground (SOG): average speed of vessels entering/leaving the port

Edge Features:
- Trip duration (avg): Average trip duration
- Trip duration (std): Standard deviation of trip durations
- Trip count: Number of trips on an edge

Temporal K-Core Decomposition:
- The k-core is the maximal subgraph where each node has at least k connections
- Useful for identifying the most densely connected ports in the network
- k-core is applied to each daily subgraph to determine the list of active nodes, then a score is calculated over the daily lists of active nodes. The score in question is the frequency at which a certain node appears in the list of active nodes over time. A node being present more than the threshold frequency h over time is retained in the master graph.

The output is a filtered (using temporal k-core defined above) temporal graph dataset compatible with PyTorch Geometric, enabling prediction tasks on maritime traffic patterns. This dataset will serve for training and evaluating LSTM, STGAT, and graph WaveNet.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import networkx as nx


class TemporalGraphBuilder:
    """
    Builds temporal graph snapshots from AIS Origin-Destination data.
    
    This class processes maritime trip data and creates daily graph snapshots where:
    - Nodes represent ports
    - Edges represent trips between ports
    - Features are computed per day for temporal modeling
    - Optional temporal k-core decomposition can be applied to filter graphs
    
    Attributes:
        data (pd.DataFrame): The input AIS O-D data
        port_to_idx (Dict[str, int]): Mapping from port names to node indices
        idx_to_port (Dict[int, str]): Mapping from node indices to port names
        daily_graphs (List[Data]): List of PyTorch Geometric Data objects, one per day
        k (Optional[int]): K value for k-core decomposition
        h (Optional[float]): Frequency threshold for temporal k-core filtering
        active_nodes (Optional[set]): Set of nodes that pass temporal k-core filtering
    """
    
    def __init__(self, data: pd.DataFrame, k: Optional[int] = None, h: Optional[float] = None):
        """
        Initialize the TemporalGraphBuilder with AIS O-D data.
        
        Args:
            data (pd.DataFrame): DataFrame containing AIS Origin-Destination trip data
            k (Optional[int]): K value for k-core decomposition. If None, no k-core filtering is applied.
            h (Optional[float]): Frequency threshold for temporal k-core filtering. 
                                 Nodes appearing in k-core more than h fraction of time are retained.
        """
        self.data = data.copy()
        self.port_to_idx = {}
        self.idx_to_port = {}
        self.daily_graphs = []
        self.k = k
        self.h = h
        self.active_nodes = None  # Will store nodes that pass temporal k-core filtering
        
        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess the input data: create port mappings from origin and destination ports."""
        # Create port mappings (unique port names)
        unique_ports = pd.concat([
            self.data['origin_port'],
            self.data['destination_port']
        ]).unique()
        
        self.port_to_idx = {port_name: idx for idx, port_name in enumerate(sorted(unique_ports))}
        self.idx_to_port = {idx: port_name for port_name, idx in self.port_to_idx.items()}
        
    def _compute_node_features(self, day_data: pd.DataFrame) -> torch.Tensor:
        """
        Compute node features for a specific day.
        
        Features per node:
        - Inbound flow count: Number of trips arriving at this port
        - Outbound flow count: Number of trips departing from this port
        - Average dwell time at origin: Average dwell time for outbound trips
        - Average deadweight tonnage: Average deadweight of vessels using this port
        - Average speed over ground: Average SOG for trips involving this port
        
        Args:
            day_data (pd.DataFrame): Data for a specific day
            
        Returns:
            torch.Tensor: Node feature matrix of shape [num_nodes, num_features]
        """
        num_nodes = len(self.port_to_idx)
        node_features = np.zeros((num_nodes, 5))
        
        # For each port, compute features
        for port_name, node_idx in self.port_to_idx.items():
            # Inbound flows (trips arriving at this port)
            inbound = day_data[day_data['destination_port'] == port_name]
            inbound_count = len(inbound)
            
            # Outbound flows (trips departing from this port)
            outbound = day_data[day_data['origin_port'] == port_name]
            outbound_count = len(outbound)
            
            # Average dwell time at origin (for outbound trips)
            avg_dwell_time = outbound['dwell_time'].mean() if len(outbound) > 0 else 0.0
            avg_dwell_time = 0.0 if pd.isna(avg_dwell_time) else avg_dwell_time
            
            # Average deadweight tonnage (from both inbound and outbound)
            all_trips = pd.concat([inbound, outbound])
            avg_dwt = all_trips['deadweight'].mean() if len(all_trips) > 0 else 0.0
            avg_dwt = 0.0 if pd.isna(avg_dwt) else avg_dwt
            
            # Average speed over ground (from both inbound and outbound)
            avg_sog = all_trips['sog'].mean() if len(all_trips) > 0 else 0.0
            avg_sog = 0.0 if pd.isna(avg_sog) else avg_sog
            
            node_features[node_idx] = [
                inbound_count,
                outbound_count,
                avg_dwell_time,
                avg_dwt,
                avg_sog
            ]
        
        return torch.tensor(node_features, dtype=torch.float)
    
    def _compute_edge_features(self, day_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge indices and features for a specific day.
        
        Edge features:
        - Mean trip duration: Average duration of trips on this edge
        - Std trip duration: Standard deviation of trip durations on this edge
        - Trip count: Number of trips on this edge
        
        Args:
            day_data (pd.DataFrame): Data for a specific day
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - edge_index: Tensor of shape [2, num_edges]
                - edge_attr: Tensor of shape [num_edges, num_features]
        """
        # Group trips by origin-destination pairs
        edge_dict = {}
        
        for _, trip in day_data.iterrows():
            origin_idx = self.port_to_idx[trip['origin_port']]
            dest_idx = self.port_to_idx[trip['destination_port']]
            edge_key = (origin_idx, dest_idx)
            
            # Collect trip durations for this edge
            trip_duration = trip['duration'] if not pd.isna(trip['duration']) else 0.0
            
            if edge_key not in edge_dict:
                edge_dict[edge_key] = []
            edge_dict[edge_key].append(trip_duration)
        
        # Compute aggregated edge features
        edge_list = []
        edge_features = []
        
        for edge_key, durations in edge_dict.items():
            origin_idx, dest_idx = edge_key
            edge_list.append([origin_idx, dest_idx])
            
            # Compute statistics
            durations_array = np.array(durations)
            mean_duration = np.mean(durations_array)
            std_duration = np.std(durations_array) if len(durations_array) > 1 else 0.0
            trip_count = len(durations_array)
            
            edge_features.append([mean_duration, std_duration, trip_count])
        
        if len(edge_list) == 0:
            # No edges for this day
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def _apply_kcore_decomposition(self, graph: Data) -> Tuple[Data, set]:
        """
        Apply k-core decomposition to a graph snapshot.
        
        The k-core of a graph is the maximal subgraph where each node has at least k connections.
        This method returns the graph and the set of nodes that are in the k-core.
        
        Args:
            graph (Data): PyTorch Geometric Data object
            
        Returns:
            Tuple[Data, set]: Original graph and set of node indices in the k-core
        """
        if self.k is None or self.k <= 0:
            # Return all nodes as active
            return graph, set(range(graph.num_nodes))
        
        # Convert PyG graph to NetworkX for k-core computation
        edge_index = graph.edge_index.numpy()
        G = nx.Graph()
        
        # Add all nodes
        G.add_nodes_from(range(graph.num_nodes))
        
        # Add edges (convert directed to undirected for k-core)
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # Compute k-core
        try:
            G.remove_edges_from(nx.selfloop_edges(G))
            kcore_graph = nx.k_core(G, k=self.k)
            kcore_nodes = set(kcore_graph.nodes())
        except nx.NetworkXError:
            # If k-core doesn't exist, return empty set
            kcore_nodes = set()
        
        return graph, kcore_nodes
    
    def _compute_temporal_kcore_nodes(self, daily_graphs_with_kcores: List[Tuple[Data, set]]) -> set:
        """
        Compute which nodes should be retained based on temporal k-core filtering.
        
        A node is retained if it appears in the k-core of daily snapshots more than 
        threshold frequency h over time.
        
        Args:
            daily_graphs_with_kcores: List of tuples (graph, kcore_nodes_set) for each day
            
        Returns:
            set: Set of node indices that pass the temporal k-core threshold
        """
        if self.h is None or self.k is None:
            # No temporal filtering, return all nodes
            return set(range(len(self.port_to_idx)))
        
        # Count how many times each node appears in k-cores
        node_appearance_count = {}
        total_days = len(daily_graphs_with_kcores)
        
        for _, kcore_nodes in daily_graphs_with_kcores:
            for node_idx in kcore_nodes:
                node_appearance_count[node_idx] = node_appearance_count.get(node_idx, 0) + 1
        
        # Calculate frequency for each node and filter by threshold h
        active_nodes = set()
        for node_idx, count in node_appearance_count.items():
            frequency = count / total_days
            if frequency > self.h:
                active_nodes.add(node_idx)
        
        return active_nodes
    
    def _filter_graph_by_nodes(self, graph: Data, active_nodes: set) -> Data:
        """
        Filter a graph to keep only the specified active nodes.
        
        Args:
            graph (Data): Original graph
            active_nodes (set): Set of node indices to keep
            
        Returns:
            Data: Filtered graph containing only active nodes and their edges
        """
        if len(active_nodes) == 0:
            # Return empty graph
            empty_graph = Data(
                x=torch.zeros((0, graph.x.shape[1]), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, graph.edge_attr.shape[1]), dtype=torch.float),
                num_nodes=0
            )
            if hasattr(graph, 'day_count'):
                empty_graph.day_count = graph.day_count
            return empty_graph
        
        # Create mapping from old node indices to new node indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(active_nodes))}
        
        # Filter node features
        new_node_features = graph.x[sorted(active_nodes)]
        
        # Filter edges: keep only edges where both nodes are in active_nodes
        edge_index = graph.edge_index.numpy()
        new_edge_list = []
        new_edge_features = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in active_nodes and dst in active_nodes:
                new_edge_list.append([old_to_new[src], old_to_new[dst]])
                new_edge_features.append(graph.edge_attr[i].numpy())
        
        # Create new graph
        if len(new_edge_list) > 0:
            new_edge_index = torch.tensor(new_edge_list, dtype=torch.long).t().contiguous()
            new_edge_attr = torch.tensor(new_edge_features, dtype=torch.float)
        else:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long)
            new_edge_attr = torch.zeros((0, graph.edge_attr.shape[1]), dtype=torch.float)
        
        filtered_graph = Data(
            x=new_node_features,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            num_nodes=len(active_nodes)
        )
        
        # Preserve metadata
        if hasattr(graph, 'day_count'):
            filtered_graph.day_count = graph.day_count
        
        # Store the mapping for reference
        filtered_graph.node_mapping = old_to_new
        filtered_graph.original_indices = sorted(active_nodes)
        
        return filtered_graph
    
    def build_temporal_graphs(self) -> List[Data]:
        """
        Build temporal graph snapshots for each day in the dataset.
        
        Creates one PyTorch Geometric Data object per day, containing:
        - x: Node features
        - edge_index: Edge connectivity
        - edge_attr: Edge features
        - day_count: The temporal identifier for the snapshot
        
        If k and h are specified, applies temporal k-core decomposition:
        1. Computes k-core for each daily snapshot
        2. Tracks which nodes appear in k-cores across time
        3. Calculates frequency of appearance for each node
        4. Retains only nodes appearing more than threshold h
        5. Filters all daily graphs to contain only these nodes
        
        Returns:
            List[Data]: List of PyTorch Geometric Data objects, one per day
        """
        self.daily_graphs = []
        
        # Get unique day_counts and sort them (chronological order)
        unique_days = sorted(self.data['day_count'].unique())
        
        # First pass: Build all daily graphs
        daily_graphs_unfiltered = []
        for day_count in unique_days:
            # Filter data for this day
            day_data = self.data[self.data['day_count'] == day_count]
            
            # Compute node features
            node_features = self._compute_node_features(day_data)
            
            # Compute edge features
            edge_index, edge_attr = self._compute_edge_features(day_data)
            
            # Create PyTorch Geometric Data object
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(self.port_to_idx)
            )
            
            # Store the day_count as metadata
            graph.day_count = day_count
            
            daily_graphs_unfiltered.append(graph)
        
        # Apply temporal k-core decomposition if k and h are specified
        if self.k is not None and self.h is not None:
            # Apply k-core to each daily graph and collect k-core nodes
            daily_graphs_with_kcores = []
            for graph in daily_graphs_unfiltered:
                graph, kcore_nodes = self._apply_kcore_decomposition(graph)
                daily_graphs_with_kcores.append((graph, kcore_nodes))
            
            # Compute which nodes pass the temporal frequency threshold
            self.active_nodes = self._compute_temporal_kcore_nodes(daily_graphs_with_kcores)
            
            # Third pass: Filter all graphs to contain only active nodes
            for graph, _ in daily_graphs_with_kcores:
                filtered_graph = self._filter_graph_by_nodes(graph, self.active_nodes)
                self.daily_graphs.append(filtered_graph)
        else:
            # No temporal k-core filtering, use all graphs as-is
            self.daily_graphs = daily_graphs_unfiltered
            self.active_nodes = set(range(len(self.port_to_idx)))
        
        return self.daily_graphs
    
    def get_port_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the port name to node index mappings.
        
        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: 
                - port_to_idx: Mapping from port names to node indices
                - idx_to_port: Mapping from node indices to port names
        """
        return self.port_to_idx, self.idx_to_port
    
    def get_num_nodes(self) -> int:
        """
        Get the total number of nodes (ports) in the graph.
        
        Returns:
            int: Number of unique ports
        """
        return len(self.port_to_idx)
    
    def get_num_snapshots(self) -> int:
        """
        Get the number of temporal snapshots (days) in the dataset.
        
        Returns:
            int: Number of daily graph snapshots
        """
        return len(self.data['day_count'].unique())
    
    def get_date_range(self) -> Tuple[int, int]:
        """
        Get the day_count range of the dataset.
        
        Returns:
            Tuple[int, int]: Minimum and maximum day_count values
        """
        return int(self.data['day_count'].min()), int(self.data['day_count'].max())
    
    def get_active_nodes(self) -> Optional[set]:
        """
        Get the set of active nodes after temporal k-core filtering.
        
        Returns:
            Optional[set]: Set of node indices that passed temporal k-core filtering,
                          or None if filtering hasn't been applied yet
        """
        return self.active_nodes

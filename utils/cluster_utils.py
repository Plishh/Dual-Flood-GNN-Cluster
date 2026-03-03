import torch
import numpy as np
import os
import random
from torch_geometric.data import Data
#from torch_geometric.transforms import Metis
import pymetis
from torch_geometric.utils import to_undirected

# --- Import data retrieval functions from your provided files ---
# (Assuming these files are accessible in the Python path, e.g., in a 'data' package)
try:
    # Adjust the import path if your files are located elsewhere
    from data.shp_data_retrieval import get_edge_index, get_cell_elevation, get_edge_length, get_edge_slope
    from data.hecras_data_retrieval import get_cell_area, get_roughness, get_face_length, get_edge_direction_x, get_edge_direction_y
except ImportError as e:
    print(f"Warning: Could not import data retrieval functions. Ensure the 'data' package is in your PYTHONPATH. Error: {e}")
    # Define dummy functions if imports fail, to allow script loading without errors
    def get_edge_index(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_cell_elevation(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_edge_length(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_edge_slope(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_cell_area(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_roughness(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_face_length(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_edge_direction_x(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")
    def get_edge_direction_y(*args, **kwargs): raise NotImplementedError(f"Import failed: {e}")

# Define the expected static feature names based on FloodEventDataset
# Ensure this order matches how features are combined below
STATIC_NODE_FEATURES_ORDER = ['area', 'roughness', 'elevation']
# Adjust this list based on the actual static edge features your model will use
STATIC_EDGE_FEATURES_ORDER = ['face_length', 'length', 'slope'] # Add 'direction_x', 'direction_y' if needed

def load_base_graph_structure(raw_dir: str, nodes_shp_file: str, edges_shp_file: str, hec_ras_file: str) -> Data:
    """
    Loads the base graph structure (static features and connectivity) directly
    from raw SHP and HDF files using the provided data retrieval functions.
    This structure is loaded into memory for partitioning.

    Args:
        raw_dir: The directory containing the raw data files (SHP, HDF).
        nodes_shp_file: Name of the nodes shapefile (e.g., 'nodes.shp').
        edges_shp_file: Name of the edges shapefile (e.g., 'edges.shp').
        hec_ras_file: Name of a representative HEC-RAS HDF file for geometry (e.g., 'event1.p01.hdf').

    Returns:
        A PyG Data object containing the static graph structure needed for partitioning.
    """
    nodes_shp_path = os.path.join(raw_dir, nodes_shp_file)
    edges_shp_path = os.path.join(raw_dir, edges_shp_file)
    hec_ras_path = os.path.join(raw_dir, hec_ras_file)

    # --- Input File Validation ---
    required_files = [nodes_shp_path, edges_shp_path, hec_ras_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Required file for partitioning not found: {f_path}")

    print(f"Loading base graph structure using:\n  Nodes: {nodes_shp_path}\n  Edges: {edges_shp_path}\n  Geo:   {hec_ras_path}")

    # --- Load Connectivity ---
    edge_index_np = get_edge_index(edges_shp_path)

    # --- Load Static Node Features ---
    node_feature_map = {
        "area": get_cell_area(hec_ras_path),
        "roughness": get_roughness(hec_ras_path),
        "elevation": get_cell_elevation(nodes_shp_path),
    }
    # Stack node features in the predefined order
    static_nodes_list = []
    for feat in STATIC_NODE_FEATURES_ORDER:
        try:
            feature_data = node_feature_map[feat]
            # Ensure feature_data is 1D or reshape if necessary
            if feature_data.ndim > 1:
                 feature_data = feature_data.squeeze() # Adjust if needed
            static_nodes_list.append(feature_data)
        except KeyError:
            raise ValueError(f"Static node feature '{feat}' not found in loaded data.")
    static_nodes_np = np.stack(static_nodes_list, axis=-1) # Shape: (num_nodes, num_node_features)

    # --- Load Static Edge Features ---
    edge_feature_map = {
        "face_length": get_face_length(hec_ras_path),
        "length": get_edge_length(edges_shp_path),
        "slope": get_edge_slope(edges_shp_path),
        # Uncomment if directions are static and needed by the model
        # "direction_x": get_edge_direction_x(hec_ras_path),
        # "direction_y": get_edge_direction_y(hec_ras_path),
    }
     # Stack edge features in the predefined order
    static_edges_list = []
    for feat in STATIC_EDGE_FEATURES_ORDER:
        try:
            feature_data = edge_feature_map[feat]
             # Ensure feature_data matches edge dimension or reshape
            if feature_data.ndim > 1:
                 feature_data = feature_data.squeeze()
            # Ensure length matches number of edges
            if len(feature_data) != edge_index_np.shape[1]:
                 # Try aligning based on potential HDF vs SHP edge ordering differences
                 # This might require more sophisticated mapping if orders differ
                 print(f"Warning: Length mismatch for edge feature '{feat}'. Expected {edge_index_np.shape[1]}, got {len(feature_data)}. Alignment might be needed.")
                 # Attempt simple truncation/padding if desperate, but proper mapping is better
                 # feature_data = feature_data[:edge_index_np.shape[1]] # Example: Truncate (Likely incorrect)

            static_edges_list.append(feature_data)

        except KeyError:
            raise ValueError(f"Static edge feature '{feat}' not found in loaded data.")

    # Ensure all edge features have the same length after potential adjustments
    num_edges = edge_index_np.shape[1]
    static_edges_list_aligned = []
    for i, arr in enumerate(static_edges_list):
        if len(arr) != num_edges:
             raise ValueError(f"Aligned length mismatch for edge feature '{STATIC_EDGE_FEATURES_ORDER[i]}'. Expected {num_edges}, got {len(arr)}. Check HDF/SHP edge consistency.")
        static_edges_list_aligned.append(arr)


    static_edges_np = np.stack(static_edges_list_aligned, axis=-1) # Shape: (num_edges, num_edge_features)


    # --- Convert to Tensors and Create Data Object ---
    edge_index = torch.from_numpy(edge_index_np).long()
    x_s = torch.from_numpy(static_nodes_np).float()   # Static node features
    edge_attr = torch.from_numpy(static_edges_np).float() # Static edge features

    num_nodes = x_s.shape[0]

    # Create the Data object for partitioning
    # Note: Using 'x' for node features as expected by Metis transform
    # This Data object exists only in memory.
    data = Data(x=x_s, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes

    # Extract and store node positions for visualization
    from data.shp_data_retrieval import get_cell_position
    pos = get_cell_position(nodes_shp_path)
    data.pos = torch.from_numpy(pos).float()

    print(f"Loaded base graph structure with {data.num_nodes} nodes and {data.num_edges} edges.")
    return data


# def partition_graph(data: Data, num_clusters: int) -> Data:
#     """
#     Partitions the graph into clusters using METIS.
#     This is the core preprocessing step of Cluster-GCN performed IN MEMORY.

#     Args:
#         data: The PyG Data object (in memory) containing the graph structure.
#               Must have 'x', 'edge_index', 'num_nodes'.
#         num_clusters: The desired number of partitions.

#     Returns:
#         The same Data object (still in memory) with an added 'part' attribute
#         mapping nodes to clusters. This 'part' attribute IS NOT SAVED TO DISK by this function.
#     """
#     if data.x is None or data.edge_index is None or data.num_nodes is None:
#         raise ValueError("Input Data object must contain 'x', 'edge_index', and 'num_nodes'.")

#     print(f"Partitioning graph into {num_clusters} clusters using METIS (in memory)...")

#     # METIS partitioning works best if the graph is undirected and contiguous
#     # Add checks or transformations if necessary, e.g., data = data.coalesce()
#     # from torch_geometric.utils import to_undirected # Example
#     # data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, num_nodes=data.num_nodes) # Ensure undirected if needed

#     # The Metis transform adds the 'part' attribute directly TO THE IN-MEMORY data object.
#     # It does NOT write any new files.
#     #cluster_transform = Metis(num_partitions=num_clusters, recursive=False)
#     cluster_transform = metis.part_graph(data, num_clusters)
#     # Metis might modify the input data object in place, or return a new one (PyG behavior might vary).
#     # Assign the result back to ensure we have the clustered version.
#     clustered_data = cluster_transform(data)

#     if not hasattr(clustered_data, 'part'):
#          raise RuntimeError("METIS transform did not add the 'part' attribute to the data object.")
#     actual_clusters = clustered_data.part.max().item() + 1
#     if actual_clusters != num_clusters:
#         print(f"Warning: METIS produced {actual_clusters} clusters, but {num_clusters} were requested.")

#     print("Graph partitioning complete (in memory). 'part' attribute added to Data object.")
#     # The returned object contains the cluster assignments in memory.
#     return clustered_data

# def partition_graph(data: Data, num_clusters: int) -> Data:
#     """
#     Partitions the graph into spatial clusters using PyMetis.
#     """
#     # METIS requires an undirected graph for balanced partitioning
#     edge_index = to_undirected(data.edge_index)
    
#     # Convert edge_index to an adjacency list format required by pymetis
#     # adj_list[i] contains all neighbors of node i
#     adj_list = [[] for _ in range(data.num_nodes)]
#     edge_index_np = edge_index.cpu().numpy()
    
#     for i in range(edge_index_np.shape[1]):
#         u = edge_index_np[0, i]
#         v = edge_index_np[1, i]
#         adj_list[u].append(v)

#     print(f"Partitioning graph into {num_clusters} clusters...")
#     # pymetis.part_graph returns (cuts, partition_indices)
#     cuts, parts = pymetis.part_graph(num_clusters, adjacency=adj_list)
    
#     # Store the partition info back in the Data object for ClusterLoader
#     data.part = torch.tensor(parts, dtype=torch.long)
    
#     return data


import matplotlib.pyplot as plt

def visualize_partitions(data: Data, num_clusters: int, save_path: str = None):
    """
    Partitions the data and visualizes it in one go.
    """
    # 1. Perform the partitioning internally
    data = partition_graph(data, num_clusters)

    # 2. Extract coordinates and partition IDs
    # Assumes data.pos contains [x, y] coordinates
    pos = data.pos.cpu().numpy()
    parts = data.part.cpu().numpy()

    # 3. Plotting
    plt.figure(figsize=(12, 10))
    # 'tab20' or 'prism' are good colormaps for distinct clusters
    scatter = plt.scatter(pos[:, 0], pos[:, 1], c=parts, cmap='tab20', s=15, alpha=0.8)
    
    plt.colorbar(scatter, label='Partition ID')
    plt.title(f'FloodGNN Mesh Partitioning: {num_clusters} Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()


def partition_graph(edge_index, num_nodes, num_clusters):
    """
    Generates a spatial partition map using PyMetis. 
    This is run only ONCE before training starts.
    """
    # METIS requires an undirected graph
    edge_index_undirected = to_undirected(edge_index, num_nodes=num_nodes)
    
    # Create adjacency list for PyMetis
    adj_list = [[] for _ in range(num_nodes)]
    for u, v in edge_index_undirected.t().tolist():
        adj_list[u].append(v)

    print(f"Partitioning mesh into {num_clusters} spatial clusters...", flush=True)
    cuts, parts = pymetis.part_graph(num_clusters, adjacency=adj_list)
    
    return torch.tensor(parts, dtype=torch.long)


def get_clusters_list(
    num_clusters: int,
    clusters_per_batch: int,
    rng: random.Random | None = None
) -> list[list[int]]:
    """
    Randomly partition all clusters into groups, each used to form a combined subgraph.

    Args:
        num_clusters: Total number of clusters in the partition map.
        clusters_per_batch: Number of clusters to combine per batch.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        List of cluster id groups. Every cluster appears in exactly one group.
    """
    if num_clusters <= 0 or clusters_per_batch <= 0:
        return []

    rand = rng if rng is not None else random
    cluster_ids = list(range(num_clusters))
    rand.shuffle(cluster_ids)

    groups: list[list[int]] = []
    for i in range(0, num_clusters, clusters_per_batch):
        groups.append(cluster_ids[i:i + clusters_per_batch])

    return groups

def get_sliding_window_clusters(
    num_clusters: int,
    clusters_per_batch: int
) -> list[list[int]]:
    """
    Generate overlapping cluster groups using a sliding window approach.

    Args:
        num_clusters: Total number of clusters in the partition map.
        clusters_per_batch: Number of clusters to combine per batch.

    Returns:
        List of cluster id groups. Each group overlaps with the previous by (clusters_per_batch - 1).
    """
    if num_clusters <= 0 or clusters_per_batch <= 0:
        return []

    groups: list[list[int]] = []
    for start in range(0, num_clusters - clusters_per_batch + 1):
        group = list(range(start, start + clusters_per_batch))
        groups.append(group)

    return groups


def get_centered_neighbor_groups(
    edge_index: torch.Tensor,
    part: torch.Tensor,
    num_clusters: int | None = None
) -> list[list[int]]:
    """
    Build one group per cluster where the group contains the center cluster
    and every cluster connected to it by at least one edge.

    Args:
        edge_index: Graph connectivity (2, E) with node indices.
        part: Cluster assignment per node (num_nodes,).
        num_clusters: Optional total number of clusters. Inferred from part if None.

    Returns:
        List of groups, one per center cluster id. Each group covers all
        clusters adjacent to the center, so all outgoing cluster edges are included.
    """
    if edge_index.numel() == 0 or part.numel() == 0:
        return []

    total_clusters = int(part.max().item() + 1) if num_clusters is None else num_clusters
    if total_clusters <= 0:
        return []

    edge_index_undirected = to_undirected(edge_index, num_nodes=part.numel())

    adjacency: list[set[int]] = [set() for _ in range(total_clusters)]
    
    for u, v in edge_index_undirected.t().tolist():
        cu = int(part[u])
        cv = int(part[v])
        if cu != cv:
            adjacency[cu].add(cv)
            adjacency[cv].add(cu)

    groups: list[list[int]] = []
    for center in range(total_clusters):
        neighbors = sorted(adjacency[center])
        groups.append([center, *neighbors])

    return groups
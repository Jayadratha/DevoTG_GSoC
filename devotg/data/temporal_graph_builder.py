"""
Temporal Graph Builder for C. elegans Cell Division Data

This module contains the core functionality for converting cell division data
into Continuous Time Dynamic Graphs (CTDG) format for temporal graph neural networks.

"""

from __future__ import annotations
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import TemporalData
from collections import defaultdict
from typing import Dict


def pad_feature(base: list[float], dim: int = 172) -> Tensor:
    """
    Return a 1-D tensor of length `dim`, left-aligned with `base` values.
    
    Args:
        base: List of base feature values
        dim: Target dimension for the feature vector
        
    Returns:
        Padded feature tensor of shape (dim,)
    """
    out = torch.zeros(dim, dtype=torch.float32)
    out[: len(base)] = torch.tensor(base, dtype=torch.float32)
    return out


def build_cell_ctdg(
    csv_path: str,
    feature_dim: int = 172,
) -> TemporalData:
    """
    Build Continuous Time Dynamic Graph from cell division data.
    
    Parameters
    ----------
    csv_path : str
        Path to the lineage table (CSV).  It contains the columns
        Parent Cell, parent_x, parent_y, parent_z,
        Daughter 1, Daughter 2, Birth Time.
    feature_dim : int, default 172
        Size of node and edge feature vectors expected by the model.

    Returns
    -------
    TemporalData
        PyTorch Geometric object with
           • src / dst ............. Tensor[num_edges]
           • t  .................... Tensor[num_edges]
           • msg  .................. Tensor[num_edges, feature_dim]
           • x   ................... Tensor[num_nodes, feature_dim]
           • birth_time ............ Tensor[num_nodes]
           • generation ............ Tensor[num_nodes]  (optional)
    """

    ## Read and normalise the table
    df = pd.read_csv(csv_path, sep=None, engine="python").fillna("")
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )

    # Required columns after normalisation
    req = [
        "parent_cell",
        "parent_x",
        "parent_y",
        "parent_z",
        "daughter_1",
        "daughter_2",
        "birth_time",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    ## Assign every cell name a compact integer id
    id_map: dict[str, int] = {}
    def get_id(name: str) -> int:
        if name not in id_map:
            id_map[name] = len(id_map)
        return id_map[name]

    # Scan once to enumerate every unique cell label
    for row in df.itertuples(index=False):
        get_id(row.parent_cell)
        get_id(row.daughter_1)
        get_id(row.daughter_2)

    num_nodes = len(id_map)

    ## Initialise per-node meta data
    coords = torch.zeros((num_nodes, 3))
    birth   = torch.full((num_nodes,), float("inf"))
    gen     = torch.full((num_nodes,), -1, dtype=torch.long)  # generation depth

    # Generation depth = shortest-path distance from root (P0)
    # We derive it incrementally while streaming the rows chronologically.
    # Assumption: the table is already sorted by Birth Time.
    parent_generation: dict[str, int] = defaultdict(lambda: -1)
    parent_generation["P0"] = 0

    ## Build edge lists + edge features (= messages)
    src_list, dst_list, t_list, msg_list = [], [], [], []

    for row in df.itertuples(index=False):
        p_id = get_id(row.parent_cell)
        d1_id = get_id(row.daughter_1)
        d2_id = get_id(row.daughter_2)

        # Update parent node attributes (only once is fine)
        coords[p_id] = torch.tensor([row.parent_x, row.parent_y, row.parent_z])
        birth[p_id]  = min(birth[p_id], float(row.birth_time))

        g_p = parent_generation[row.parent_cell]
        if g_p >= 0:
            gen[p_id] = g_p
        else:
            # unseen ancestor – assume last known generation + 1 later
            pass

        # Assign generation numbers to daughters
        parent_generation[row.daughter_1] = g_p + 1
        parent_generation[row.daughter_2] = g_p + 1
        gen[d1_id] = g_p + 1
        gen[d2_id] = g_p + 1
        birth[d1_id] = min(birth[d1_id], float(row.birth_time))
        birth[d2_id] = min(birth[d2_id], float(row.birth_time))

        # ---------------- Edge 1: parent → daughter 1 ----------------
        # ---------------- Edge 2: parent → daughter 2 ----------------
        for d_id in (d1_id, d2_id):
            src_list.append(p_id)
            dst_list.append(d_id)
            t_list.append(float(row.birth_time))

            # Edge feature:
            # [parent_xyz (3), daughter_xyz (3, default 0), birth_time (1)] → pad to 172
            msg = pad_feature(
                [
                    row.parent_x,
                    row.parent_y,
                    row.parent_z,
                    # daughter coords might be absent at division time
                    0.0,
                    0.0,
                    0.0,
                    float(row.birth_time),
                ],
                feature_dim,
            )
            msg_list.append(msg)

    ##  Build node feature matrix
    node_features = []
    for nid in range(num_nodes):
        base = [
            *coords[nid].tolist(),     # 0-2: spatial coordinates
            birth[nid],               # 3 : birth time
            float(gen[nid]),          # 4 : generation level
        ]
        node_features.append(pad_feature(base, feature_dim))
    node_features = torch.stack(node_features, dim=0)

    ## Assemble TemporalData object
    data = TemporalData(
        src=torch.tensor(src_list, dtype=torch.long),
        dst=torch.tensor(dst_list, dtype=torch.long),
        t=torch.tensor(t_list, dtype=torch.long),
        msg=torch.stack(msg_list, dim=0),
        x=node_features,
    )

    # Optional convenience attributes
    data.birth_time = birth
    data.generation = gen

    print(f"Created CTDG with {data.num_nodes} nodes and {data.num_events} events.")
    return data


class TemporalGraphBuilder:
    """
    Class wrapper for temporal graph building functionality.
    
    Provides additional utilities and configuration options for building
    temporal graphs from cell division data.
    """
    
    def __init__(self, feature_dim: int = 172):
        """
        Initialize the temporal graph builder.
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        self.feature_dim = feature_dim
        
    def build_from_csv(self, csv_path: str) -> TemporalData:
        """
        Build temporal graph from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            TemporalData object
        """
        return build_cell_ctdg(csv_path, self.feature_dim)
        
    def build_from_dataframe(self, df: pd.DataFrame) -> TemporalData:
        """
        Build temporal graph from pandas DataFrame.
        
        Args:
            df: DataFrame with cell division data
            
        Returns:
            TemporalData object
        """
        # Save to temporary file and use existing function
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
            
        try:
            return build_cell_ctdg(temp_path, self.feature_dim)
        finally:
            os.unlink(temp_path)
    
    def get_node_mapping(self, csv_path: str) -> Dict[str, int]:
        """
        Get mapping from cell names to node IDs.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary mapping cell names to integer IDs
        """
        df = pd.read_csv(csv_path, sep=None, engine="python").fillna("")
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.lower()
        )
        
        id_map: dict[str, int] = {}
        def get_id(name: str) -> int:
            if name not in id_map:
                id_map[name] = len(id_map)
            return id_map[name]

        # Scan once to enumerate every unique cell label
        for row in df.itertuples(index=False):
            get_id(row.parent_cell)
            get_id(row.daughter_1)
            get_id(row.daughter_2)
            
        return id_map
    
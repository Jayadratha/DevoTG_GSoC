"""
Temporal Graph Network (TGN) Model

This module contains the TGN implementation for cell division prediction.
"""

import torch
from torch.nn import Linear
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from torch_geometric.loader import TemporalDataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Tuple


class GraphAttentionEmbedding(torch.nn.Module):
    """
    Graph Attention Embedding layer for TGN.
    
    Initializes the GraphAttentionEmbedding layer with time encoding and
    transformer-based message passing.
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        """
        Initialize the GraphAttentionEmbedding layer.

        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
            msg_dim (int): Dimension of the message vector.
            time_enc: Time encoding module with an attribute `out_channels` 
                     indicating its output dimension.
        """
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        """
        Forward pass of the attention embedding.
        
        Args:
            x: Node features
            last_update: Last update times for nodes
            edge_index: Edge indices
            t: Edge times
            msg: Edge messages
            
        Returns:
            Updated node embeddings
        """
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    """
    Link prediction module for TGN.
    
    Predicts the likelihood of edges between pairs of nodes.
    """

    def __init__(self, in_channels):
        """
        Initialize the link predictor.
        
        Args:
            in_channels: The number of input channels.
        """
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        """
        Forward pass for link prediction.
        
        Args:
            z_src: Source node embeddings
            z_dst: Destination node embeddings
            
        Returns:
            Link prediction scores
        """
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class TGNModel:
    """
    Complete TGN model wrapper for training and evaluation.
    
    Combines memory module, graph attention embedding, and link predictor
    for temporal graph learning on cell division data.
    """
    
    def __init__(self, num_nodes: int, msg_dim: int, memory_dim: int = 100, 
                 time_dim: int = 100, embedding_dim: int = 100, device: str = 'auto'):
        """
        Initialize the TGN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            msg_dim: Message dimension
            memory_dim: Memory module dimension
            time_dim: Time encoding dimension
            embedding_dim: Final embedding dimension
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self._build_model()
        self._setup_training()
        
    def _build_model(self):
        """Build the model components."""
        # Memory module
        self.memory = TGNMemory(
            self.num_nodes,
            self.msg_dim,
            self.memory_dim,
            self.time_dim,
            message_module=IdentityMessage(self.msg_dim, self.memory_dim, self.time_dim),
            aggregator_module=LastAggregator(),
        ).to(self.device)

        # Graph neural network
        self.gnn = GraphAttentionEmbedding(
            in_channels=self.memory_dim,
            out_channels=self.embedding_dim,
            msg_dim=self.msg_dim,
            time_enc=self.memory.time_enc,
        ).to(self.device)

        # Link predictor
        self.link_pred = LinkPredictor(in_channels=self.embedding_dim).to(self.device)
        
    def _setup_training(self):
        """Setup optimizer and loss function."""
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters())
            | set(self.link_pred.parameters()), lr=0.001)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        # Helper vector to map global node indices to local ones
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long, device=self.device)
        
    def prepare_data(self, data, val_ratio: float = 0.15, test_ratio: float = 0.15,
                    batch_size: int = 200, neighbor_size: int = 10):
        """
        Prepare data loaders for training.
        
        Args:
            data: TemporalData object
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            batch_size: Batch size for training
            neighbor_size: Number of neighbors to sample
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, neighbor_loader)
        """
        data = data.to(self.device)

        train_data, val_data, test_data = data.train_val_test_split(
            val_ratio=val_ratio, test_ratio=test_ratio)

        train_loader = TemporalDataLoader(
            train_data,
            batch_size=batch_size,
            neg_sampling_ratio=1.0,
        )
        val_loader = TemporalDataLoader(
            val_data,
            batch_size=batch_size,
            neg_sampling_ratio=1.0,
        )
        test_loader = TemporalDataLoader(
            test_data,
            batch_size=batch_size,
            neg_sampling_ratio=1.0,
        )
        neighbor_loader = LastNeighborLoader(data.num_nodes, size=neighbor_size, device=self.device)
        
        self.data = data
        self.neighbor_loader = neighbor_loader
        
        return train_loader, val_loader, test_loader, neighbor_loader

    def train_epoch(self, train_loader) -> float:
        """
        Train the model for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss over the epoch
        """
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()  # Start with a fresh memory
        self.neighbor_loader.reset_state()  # Start with an empty graph

        total_loss = 0
        num_events = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            # Get updated memory of all nodes involved in the computation
            z, last_update = self.memory(n_id)
            z = self.gnn(z, last_update, edge_index, self.data.t[e_id].to(self.device),
                    self.data.msg[e_id].to(self.device))
            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            loss = self.criterion(pos_out, torch.ones_like(pos_out))
            loss += self.criterion(neg_out, torch.zeros_like(neg_out))

            # Update memory and neighbor loader with ground-truth state
            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

            loss.backward()
            self.optimizer.step()
            self.memory.detach()
            
            total_loss += float(loss) * batch.num_events
            num_events += batch.num_events

        return total_loss / num_events if num_events > 0 else 0

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        """
        Evaluate the model on the provided data loader.

        Args:
            loader: Data loader for evaluation

        Returns:
            Tuple of (average_precision, roc_auc)
        """
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        torch.manual_seed(12345)  # Ensure deterministic sampling across epochs

        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z, last_update = self.memory(n_id)
            z = self.gnn(z, last_update, edge_index, self.data.t[e_id].to(self.device),
                    self.data.msg[e_id].to(self.device))
            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)),
                 torch.zeros(neg_out.size(0))], dim=0)

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)
            
        return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

    def train_model(self, data, epochs: int = 20, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, batch_size: int = 200,
                   verbose: bool = True) -> dict:
        """
        Complete training pipeline.
        
        Args:
            data: TemporalData object
            epochs: Number of training epochs
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            batch_size: Batch size
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        # Prepare data
        train_loader, val_loader, test_loader, _ = self.prepare_data(
            data, val_ratio, test_ratio, batch_size)
        
        history = {
            'train_loss': [],
            'val_ap': [],
            'val_auc': [],
            'test_ap': [],
            'test_auc': []
        }
        
        if verbose:
            print(f"Starting training on {self.device}")
            print(f"Model has {sum(p.numel() for p in self.memory.parameters())} memory parameters")
            print(f"Model has {sum(p.numel() for p in self.gnn.parameters())} GNN parameters")
            print(f"Model has {sum(p.numel() for p in self.link_pred.parameters())} predictor parameters")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            history['train_loss'].append(loss)
            
            if verbose:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
                
            val_ap, val_auc = self.evaluate(val_loader)
            test_ap, test_auc = self.evaluate(test_loader)
            
            history['val_ap'].append(val_ap)
            history['val_auc'].append(val_auc)
            history['test_ap'].append(test_ap)
            history['test_auc'].append(test_auc)
            
            if verbose:
                print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
                print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
                print('-' * 50)
        
        return history
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'memory_state_dict': self.memory.state_dict(),
            'gnn_state_dict': self.gnn.state_dict(),
            'link_pred_state_dict': self.link_pred.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'num_nodes': self.num_nodes,
                'msg_dim': self.msg_dim,
                'memory_dim': self.memory_dim,
                'time_dim': self.time_dim,
                'embedding_dim': self.embedding_dim
            }
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.memory.load_state_dict(checkpoint['memory_state_dict'])
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.link_pred.load_state_dict(checkpoint['link_pred_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
        
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = (
            sum(p.numel() for p in self.memory.parameters()) +
            sum(p.numel() for p in self.gnn.parameters()) +
            sum(p.numel() for p in self.link_pred.parameters())
        )
        
        return {
            'total_parameters': total_params,
            'memory_parameters': sum(p.numel() for p in self.memory.parameters()),
            'gnn_parameters': sum(p.numel() for p in self.gnn.parameters()),
            'predictor_parameters': sum(p.numel() for p in self.link_pred.parameters()),
            'device': str(self.device),
            'memory_dim': self.memory_dim,
            'time_dim': self.time_dim,
            'embedding_dim': self.embedding_dim
        }
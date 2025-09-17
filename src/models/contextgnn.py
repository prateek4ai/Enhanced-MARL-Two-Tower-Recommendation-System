"""
ContextGNN Implementation for Enhanced MARL Two-Tower Recommendation System

Replaces standard user tower with graph-based attention over recent user interactions.
Produces 128-dim user context embeddings for integration with MARL controller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import math
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import logging

logger = logging.getLogger(__name__)


class TemporalEncoder(nn.Module):
    """
    Encodes temporal information using sinusoidal embeddings and learnable components.
    """
    
    def __init__(self, temporal_dim: int = 16):
        super(TemporalEncoder, self).__init__()
        self.temporal_dim = temporal_dim
        
        # Sinusoidal position encoding for time
        self.register_buffer('inv_freq', torch.exp(torch.arange(0, temporal_dim, 2) * 
                                                 -(math.log(10000.0) / temporal_dim)))
        
        # Learnable temporal transformation
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim + 7, temporal_dim),  # +7 for day-of-week one-hot
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, timestamps: torch.Tensor, day_of_week: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: [batch_size, seq_len] - Unix timestamps
            day_of_week: [batch_size, seq_len, 7] - One-hot day-of-week
        Returns:
            temporal_embeddings: [batch_size, seq_len, temporal_dim]
        """
        # Convert timestamps to sinusoidal embeddings
        pos = timestamps.unsqueeze(-1)  # [batch_size, seq_len, 1]
        sin_inp = pos * self.inv_freq  # [batch_size, seq_len, temporal_dim//2]
        
        # Sinusoidal encoding
        emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)  # [batch_size, seq_len, temporal_dim]
        
        # Combine with day-of-week
        temporal_features = torch.cat([emb, day_of_week], dim=-1)  # [batch_size, seq_len, temporal_dim + 7]
        
        return self.temporal_proj(temporal_features)


class UserGraphConstructor(nn.Module):
    """
    Constructs user interaction graphs based on temporal and content similarity.
    """
    
    def __init__(
        self, 
        k_neighbors: int = 20,
        edge_threshold: float = 0.1,
        temporal_decay: float = 0.95,
        device: str = 'cuda'
    ):
        super(UserGraphConstructor, self).__init__()
        self.k_neighbors = k_neighbors
        self.edge_threshold = edge_threshold
        self.temporal_decay = temporal_decay
        self.device = device
        
        # Learnable edge weight computation
        self.edge_weight_net = nn.Sequential(
            nn.Linear(3, 16),  # content_sim + temporal_decay + interaction_strength
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        interaction_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs interaction graph for each user.
        
        Args:
            user_embeddings: [batch_size, embed_dim]
            item_embeddings: [batch_size, seq_len, embed_dim] - User's interacted items
            timestamps: [batch_size, seq_len] - Interaction timestamps
            interaction_weights: [batch_size, seq_len] - Optional interaction strengths
            
        Returns:
            edge_index: [2, num_edges] - Graph edge indices
            edge_weights: [num_edges] - Graph edge weights
        """
        batch_size, seq_len, embed_dim = item_embeddings.shape
        
        if interaction_weights is None:
            interaction_weights = torch.ones(batch_size, seq_len, device=self.device)
        
        # Compute pairwise similarities between items
        item_sims = torch.bmm(item_embeddings, item_embeddings.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        item_sims = F.normalize(item_sims, dim=-1)
        
        # Compute temporal decay
        current_time = timestamps.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
        time_diffs = current_time - timestamps  # [batch_size, seq_len]
        temporal_weights = torch.pow(self.temporal_decay, time_diffs.unsqueeze(-1))  # [batch_size, seq_len, 1]
        temporal_decay_matrix = torch.bmm(temporal_weights, temporal_weights.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        # Combine features for edge weight computation
        edge_features = torch.stack([
            item_sims,
            temporal_decay_matrix,
            interaction_weights.unsqueeze(-1).expand(-1, -1, seq_len)
        ], dim=-1)  # [batch_size, seq_len, seq_len, 3]
        
        # Compute edge weights
        edge_weights_raw = self.edge_weight_net(edge_features).squeeze(-1)  # [batch_size, seq_len, seq_len]
        
        # Apply threshold and k-neighbors filtering
        edge_lists = []
        weight_lists = []
        
        for b in range(batch_size):
            # Get upper triangular part (avoid self-loops and duplicates)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
            
            valid_edges = edge_weights_raw[b][mask] > self.edge_threshold
            if valid_edges.sum() == 0:
                continue
                
            # Get edge indices and weights
            edge_idx = mask.nonzero(as_tuple=False).t()  # [2, num_valid]
            edge_w = edge_weights_raw[b][mask][valid_edges]
            valid_edge_idx = edge_idx[:, valid_edges]
            
            # Keep top-k edges
            if valid_edge_idx.size(1) > self.k_neighbors:
                _, topk_indices = torch.topk(edge_w, self.k_neighbors)
                valid_edge_idx = valid_edge_idx[:, topk_indices]
                edge_w = edge_w[topk_indices]
            
            # Add batch offset
            valid_edge_idx[0] += b * seq_len
            valid_edge_idx[1] += b * seq_len
            
            edge_lists.append(valid_edge_idx)
            weight_lists.append(edge_w)
        
        if edge_lists:
            edge_index = torch.cat(edge_lists, dim=1)
            edge_weights = torch.cat(weight_lists, dim=0)
        else:
            # Fallback: create minimal connectivity
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_weights = torch.empty((0,), device=self.device)
        
        return edge_index, edge_weights


class ContextGNNLayer(nn.Module):
    """
    Single layer of ContextGNN with multi-head graph attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        super(ContextGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        # Multi-head graph attention
        self.gat = GATConv(
            input_dim, 
            output_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # Residual connection projection
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim] - Node features
            edge_index: [2, num_edges] - Graph edges
            edge_weights: [num_edges] - Optional edge weights
            
        Returns:
            out: [num_nodes, output_dim] - Updated node features
        """
        # Graph attention
        out = self.gat(x, edge_index, edge_attr=edge_weights)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                x = self.residual_proj(x)
            out = out + x
        
        # Layer normalization and activation
        out = self.layer_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class ContextGNN(nn.Module):
    """
    ContextGNN User Encoder - replaces standard user tower with graph-based attention
    over recent user interactions. Produces rich contextual user representations.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        k_neighbors: int = 20,
        edge_threshold: float = 0.1,
        temporal_decay: float = 0.95,
        max_sequence_length: int = 50,
        user_embed_dim: int = 64,
        temporal_dim: int = 16,
        demographic_dim: int = 32,
        device: str = 'cuda'
    ):
        """
        Initialize ContextGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Final output dimension  
            num_heads: Number of attention heads
            num_layers: Number of GNN layers
            dropout: Dropout rate
            k_neighbors: Maximum neighbors per node
            edge_threshold: Minimum edge weight threshold
            temporal_decay: Temporal decay factor
            max_sequence_length: Maximum user history length
            user_embed_dim: User ID embedding dimension
            temporal_dim: Temporal encoding dimension
            demographic_dim: Demographic feature dimension
            device: Device for computation
        """
        super(ContextGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.device = device
        
        # User feature encoders
        self.user_embedding = nn.Embedding(10000, user_embed_dim)  # Adjust vocab size as needed
        
        # Demographic encoders
        self.gender_embedding = nn.Embedding(3, 8)  # M, F, Unknown
        self.age_embedding = nn.Embedding(8, 8)     # Age groups
        self.occupation_embedding = nn.Embedding(22, 16)  # Occupation categories
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(temporal_dim)
        
        # Graph constructor
        self.graph_constructor = UserGraphConstructor(
            k_neighbors=k_neighbors,
            edge_threshold=edge_threshold,
            temporal_decay=temporal_decay,
            device=device
        )
        
        # Input projection
        total_input_dim = user_embed_dim + temporal_dim + demographic_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim)
        )
        
        # ContextGNN layers
        self.gnn_layers = nn.ModuleList()
        layer_dims = [input_dim] + hidden_dims
        
        for i in range(num_layers):
            layer = ContextGNNLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                num_heads=num_heads,
                dropout=dropout,
                use_residual=True
            )
            self.gnn_layers.append(layer)
        
        # Global pooling and output projection
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(output_dim)
        )
        
        # Attention weights for sequence pooling
        self.seq_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.1)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        user_history: Dict[str, torch.Tensor],
        user_demographics: Dict[str, torch.Tensor],
        item_embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through ContextGNN.
        
        Args:
            user_ids: [batch_size] - User ID indices
            user_history: Dict containing:
                - 'item_ids': [batch_size, seq_len] - Interacted item IDs
                - 'timestamps': [batch_size, seq_len] - Interaction timestamps  
                - 'day_of_week': [batch_size, seq_len, 7] - Day-of-week one-hot
                - 'ratings': [batch_size, seq_len] - Optional ratings/weights
            user_demographics: Dict containing:
                - 'gender': [batch_size] - Gender indices
                - 'age_group': [batch_size] - Age group indices
                - 'occupation': [batch_size] - Occupation indices
            item_embeddings: [batch_size, seq_len, embed_dim] - Item embeddings from item tower
            return_attention: Whether to return attention weights
            
        Returns:
            user_context_embedding: [batch_size, output_dim] - Final user representations
            attention_info: Optional dict with attention weights and intermediate states
        """
        batch_size, seq_len = user_history['item_ids'].shape
        
        # User embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, user_embed_dim]
        
        # Demographic embeddings
        gender_emb = self.gender_embedding(user_demographics['gender'])  # [batch_size, 8]
        age_emb = self.age_embedding(user_demographics['age_group'])     # [batch_size, 8]
        occupation_emb = self.occupation_embedding(user_demographics['occupation'])  # [batch_size, 16]
        demo_emb = torch.cat([gender_emb, age_emb, occupation_emb], dim=-1)  # [batch_size, 32]
        
        # Temporal embeddings  
        temporal_emb = self.temporal_encoder(
            user_history['timestamps'], 
            user_history['day_of_week']
        )  # [batch_size, seq_len, temporal_dim]
        
        # Combine all features for each interaction
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, user_embed_dim]
        demo_emb_expanded = demo_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, demographic_dim]
        
        node_features = torch.cat([
            user_emb_expanded,
            temporal_emb,
            demo_emb_expanded
        ], dim=-1)  # [batch_size, seq_len, total_input_dim]
        
        # Project to input dimension
        node_features = self.input_proj(node_features)  # [batch_size, seq_len, input_dim]
        
        # Flatten for graph processing
        node_features_flat = node_features.view(-1, self.input_dim)  # [batch_size * seq_len, input_dim]
        
        # Construct interaction graphs
        interaction_weights = user_history.get('ratings', None)
        edge_index, edge_weights = self.graph_constructor(
            user_emb, item_embeddings, user_history['timestamps'], interaction_weights
        )
        
        # Apply ContextGNN layers
        x = node_features_flat
        layer_outputs = []
        
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_weights)
            layer_outputs.append(x)
        
        # Reshape back to [batch_size, seq_len, hidden_dim]
        final_features = x.view(batch_size, seq_len, -1)
        
        # Sequence-level attention pooling
        pooled_features, attention_weights = self.seq_attention(
            final_features, final_features, final_features
        )  # [batch_size, seq_len, hidden_dim]
        
        # Global pooling - use attention-weighted mean
        if return_attention:
            # Use attention weights for pooling
            pooled_output = torch.sum(pooled_features * attention_weights.unsqueeze(-1), dim=1)
        else:
            # Simple mean pooling
            pooled_output = torch.mean(pooled_features, dim=1)  # [batch_size, hidden_dim]
        
        # Final projection
        user_context_embedding = self.output_proj(pooled_output)
        
        attention_info = None
        if return_attention:
            attention_info = {
                'sequence_attention': attention_weights,
                'layer_outputs': layer_outputs,
                'edge_index': edge_index,
                'edge_weights': edge_weights,
                'temporal_embeddings': temporal_emb
            }
        
        return user_context_embedding, attention_info
    
    def get_user_embedding_only(
        self, 
        user_ids: torch.Tensor,
        user_demographics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get basic user embedding without history (for cold-start users).
        
        Args:
            user_ids: [batch_size] - User ID indices
            user_demographics: Dict with demographic features
            
        Returns:
            user_embedding: [batch_size, output_dim] - Basic user embeddings
        """
        # User embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Demographic embeddings
        gender_emb = self.gender_embedding(user_demographics['gender'])
        age_emb = self.age_embedding(user_demographics['age_group'])
        occupation_emb = self.occupation_embedding(user_demographics['occupation'])
        demo_emb = torch.cat([gender_emb, age_emb, occupation_emb], dim=-1)
        
        # Combine and project
        combined = torch.cat([user_emb, demo_emb], dim=-1)
        
        # Simple projection to output dimension
        output = self.output_proj[0](self.input_proj[0](combined))  # Use first layer of projections
        
        return output
    
    def compute_graph_stats(self, edge_index: torch.Tensor, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        Compute graph statistics for analysis.
        
        Args:
            edge_index: [2, num_edges] - Graph edge indices
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            stats: Dictionary with graph statistics
        """
        if edge_index.size(1) == 0:
            return {
                'num_edges': 0,
                'avg_degree': 0.0,
                'density': 0.0,
                'connectivity': 0.0
            }
        
        num_nodes = batch_size * seq_len
        num_edges = edge_index.size(1)
        
        # Compute degrees
        degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)
        avg_degree = degrees.float().mean().item()
        
        # Compute density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0
        
        # Compute connectivity (fraction of nodes with degree > 0)
        connectivity = (degrees > 0).float().mean().item()
        
        return {
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density': density,
            'connectivity': connectivity
        }


class ContextGNNManager:
    """
    Manager class for ContextGNN with configuration and utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize ContextGNN
        self.contextgnn = ContextGNN(
            input_dim=config['contextgnn']['input_dim'],
            hidden_dims=config['contextgnn']['hidden_dims'],
            output_dim=config['contextgnn']['output_dim'],
            num_heads=config['contextgnn']['num_heads'],
            dropout=config['contextgnn']['dropout'],
            k_neighbors=config['contextgnn']['k_neighbors'],
            edge_threshold=config['contextgnn']['edge_threshold'],
            temporal_decay=config['contextgnn']['temporal_decay'],
            max_sequence_length=config['dataset']['max_sequence_length'],
            user_embed_dim=config['contextgnn']['user_embed_dim'],
            temporal_dim=config['contextgnn']['temporal_dim'],
            demographic_dim=config['contextgnn']['demographic_dim'],
            device=self.device
        ).to(self.device)
        
        logger.info(f"ContextGNN initialized with output_dim={config['contextgnn']['output_dim']}")
    
    def get_model(self) -> ContextGNN:
        """Get the ContextGNN model."""
        return self.contextgnn
    
    def preprocess_user_data(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict, Dict, torch.Tensor]:
        """
        Preprocess batch data for ContextGNN input.
        
        Args:
            batch: Raw batch data
            
        Returns:
            user_ids, user_history, user_demographics, item_embeddings
        """
        user_ids = batch['user_ids'].to(self.device)
        
        user_history = {
            'item_ids': batch['item_history'].to(self.device),
            'timestamps': batch['timestamps'].to(self.device), 
            'day_of_week': batch['day_of_week'].to(self.device),
            'ratings': batch.get('ratings', None)
        }
        
        if user_history['ratings'] is not None:
            user_history['ratings'] = user_history['ratings'].to(self.device)
        
        user_demographics = {
            'gender': batch['gender'].to(self.device),
            'age_group': batch['age_group'].to(self.device),
            'occupation': batch['occupation'].to(self.device)
        }
        
        item_embeddings = batch['item_embeddings'].to(self.device)
        
        return user_ids, user_history, user_demographics, item_embeddings

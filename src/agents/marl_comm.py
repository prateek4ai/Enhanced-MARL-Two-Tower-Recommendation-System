"""
MARL Communication Layer Implementation for Enhanced Multi-Agent Recommendation System
Enables sophisticated inter-agent communication using Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MARLCommunicationLayer(nn.Module):
    """
    Advanced MARL Communication Layer using Graph Attention Networks.
    
    Features:
    - Multi-head graph attention for selective message passing
    - Agent identity embeddings for role-aware communication
    - Adaptive adjacency matrix construction
    - Hierarchical message aggregation
    - Communication bottlenecking for efficiency
    """
    
    def __init__(
        self,
        num_agents: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        communication_type: str = "gat",
        message_dim: int = 32,
        use_residual: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize MARL Communication Layer.
        
        Args:
            num_agents: Number of agents in the system
            embed_dim: Embedding dimension for agent states
            num_heads: Number of attention heads
            num_layers: Number of GNN layers
            dropout: Dropout rate
            communication_type: Type of GNN ("gat", "gcn", "transformer")
            message_dim: Dimension of inter-agent messages
            use_residual: Whether to use residual connections
            device: Device for computation
        """
        super(MARLCommunicationLayer, self).__init__()
        
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.communication_type = communication_type
        self.message_dim = message_dim
        self.use_residual = use_residual
        self.device = device
        
        # Agent identity embeddings
        self.agent_embeddings = nn.Embedding(num_agents, embed_dim)
        
        # Genre role embeddings (for specialization awareness)
        self.role_embeddings = nn.Embedding(num_agents, embed_dim // 4)
        
        # Message transformation networks
        self.message_encoder = nn.Sequential(
            nn.Linear(embed_dim, message_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim * 2, message_dim)
        )
        
        self.message_decoder = nn.Sequential(
            nn.Linear(message_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if communication_type == "gat":
                layer = GATConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True
                )
            elif communication_type == "gcn":
                layer = GCNConv(embed_dim, embed_dim)
            elif communication_type == "transformer":
                layer = TransformerConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False
                )
            else:
                raise ValueError(f"Unsupported communication type: {communication_type}")
            
            self.gnn_layers.append(layer)
        
        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Attention pooling for message aggregation
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection layer
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Communication strength controller
        self.communication_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Communication statistics
        self.communication_stats = {
            'total_messages': 0,
            'avg_attention_weights': defaultdict(list),
            'communication_strength': []
        }
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
    
    def construct_adjacency_matrix(
        self,
        agent_states: torch.Tensor,
        communication_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dynamically construct adjacency matrix based on agent similarities.
        
        Args:
            agent_states: Agent state representations [num_agents, embed_dim]
            communication_mask: Optional mask for communication restrictions
            
        Returns:
            Adjacency matrix [num_agents, num_agents]
        """
        # Compute pairwise similarities
        normalized_states = F.normalize(agent_states, dim=-1)
        similarity_matrix = torch.mm(normalized_states, normalized_states.t())
        
        # Apply temperature scaling for sharpening
        temperature = 0.1
        similarity_matrix = similarity_matrix / temperature
        
        # Create adjacency matrix (top-k connections per agent)
        k = min(self.num_agents - 1, 5)  # Each agent connects to top-5 similar agents
        _, top_indices = torch.topk(similarity_matrix, k=k, dim=-1)
        
        adjacency = torch.zeros(self.num_agents, self.num_agents, device=self.device)
        for i in range(self.num_agents):
            adjacency[i, top_indices[i]] = 1.0
        
        # Make adjacency symmetric
        adjacency = (adjacency + adjacency.t()) / 2
        
        # Apply communication mask if provided
        if communication_mask is not None:
            adjacency = adjacency * communication_mask
        
        return adjacency
    
    def forward(
        self,
        agent_states: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        communication_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through MARL communication layer.
        
        Args:
            agent_states: Agent state tensor [num_agents, embed_dim]
            edge_index: Optional predefined edge connectivity [2, num_edges]
            communication_mask: Optional communication restrictions
            return_attention: Whether to return attention weights
            
        Returns:
            updated_states: Updated agent states [num_agents, embed_dim]
            attention_info: Optional attention information dict
        """
        batch_size = agent_states.size(0) if agent_states.dim() == 3 else 1
        if agent_states.dim() == 2:
            agent_states = agent_states.unsqueeze(0)  # Add batch dimension
        
        updated_states_list = []
        attention_weights_list = []
        
        for b in range(batch_size):
            states = agent_states[b]  # [num_agents, embed_dim]
            
            # Add agent identity embeddings
            agent_ids = torch.arange(self.num_agents, device=self.device)
            identity_embeddings = self.agent_embeddings(agent_ids)
            role_embeddings = self.role_embeddings(agent_ids)
            
            # Combine state with identity and role information
            enhanced_states = states + identity_embeddings + role_embeddings
            
            # Construct dynamic adjacency if edge_index not provided
            if edge_index is None:
                adjacency = self.construct_adjacency_matrix(enhanced_states, communication_mask)
                edge_index = adjacency.nonzero().t().contiguous()
            
            # Apply GNN layers with residual connections
            x = enhanced_states
            layer_attention_weights = []
            
            for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
                residual = x if self.use_residual else None
                
                # Apply GNN layer
                if self.communication_type == "gat":
                    x, attention_weights = gnn_layer(x, edge_index, return_attention_weights=True)
                    if return_attention:
                        layer_attention_weights.append(attention_weights)
                else:
                    x = gnn_layer(x, edge_index)
                
                # Layer normalization
                x = layer_norm(x)
                
                # Residual connection
                if residual is not None and x.shape == residual.shape:
                    x = x + residual
                
                # Dropout
                x = self.dropout(x)
            
            # Message encoding and decoding
            messages = self.message_encoder(x)
            decoded_messages = self.message_decoder(messages)
            
            # Attention-based message aggregation
            aggregated_messages, agg_attention = self.attention_pooling(
                decoded_messages.unsqueeze(0),
                decoded_messages.unsqueeze(0),
                decoded_messages.unsqueeze(0)
            )
            aggregated_messages = aggregated_messages.squeeze(0)
            
            # Communication strength gating
            gate_input = torch.cat([states, aggregated_messages], dim=-1)
            communication_strength = self.communication_gate(gate_input)
            
            # Apply gating to control communication influence
            gated_messages = aggregated_messages * communication_strength
            
            # Final projection and residual connection
            output = self.output_projection(gated_messages)
            if self.use_residual:
                output = output + states
            
            updated_states_list.append(output)
            attention_weights_list.extend(layer_attention_weights)
            
            # Update statistics
            self.communication_stats['total_messages'] += self.num_agents
            self.communication_stats['communication_strength'].append(
                communication_strength.mean().item()
            )
        
        # Stack results
        updated_states = torch.stack(updated_states_list) if batch_size > 1 else updated_states_list[0]
        
        # Remove batch dimension if it was added
        if agent_states.dim() == 2:
            updated_states = updated_states.squeeze(0)
        
        attention_info = None
        if return_attention:
            attention_info = {
                'layer_attention_weights': attention_weights_list,
                'aggregation_attention': agg_attention,
                'communication_strength': communication_strength,
                'edge_index': edge_index
            }
        
        return updated_states, attention_info
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication layer statistics."""
        return {
            'total_messages_sent': self.communication_stats['total_messages'],
            'avg_communication_strength': np.mean(self.communication_stats['communication_strength'])
                if self.communication_stats['communication_strength'] else 0.0,
            'communication_type': self.communication_type,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads
        }


class HierarchicalMARLController(nn.Module):
    """
    Hierarchical Multi-Agent RL Controller.
    
    Coordinates multiple genre agents and the exposure manager through
    sophisticated communication and global coordination mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(HierarchicalMARLController, self).__init__()
        
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # MARL configuration
        marl_config = config['marl_controller']
        self.num_agents = marl_config['genre_agents']['num_agents']
        
        # Communication layer
        comm_config = marl_config['communication']
        self.communication_layer = MARLCommunicationLayer(
            num_agents=self.num_agents,
            embed_dim=comm_config['embed_dim'],
            num_heads=comm_config['num_heads'],
            num_layers=comm_config['num_layers'],
            dropout=comm_config['dropout'],
            message_dim=comm_config['message_dim'],
            device=self.device
        )
        
        # Global coordinator
        coord_config = marl_config['coordinator']
        self.coordinator_input_dim = self.num_agents * comm_config['embed_dim']
        
        coordinator_layers = []
        current_dim = self.coordinator_input_dim
        
        for i in range(coord_config['num_layers']):
            output_dim = coord_config['hidden_dim'] if i < coord_config['num_layers'] - 1 else coord_config['output_dim']
            coordinator_layers.extend([
                nn.Linear(current_dim, output_dim),
                nn.ReLU() if i < coord_config['num_layers'] - 1 else nn.Identity(),
                nn.Dropout(0.1) if i < coord_config['num_layers'] - 1 else nn.Identity()
            ])
            current_dim = output_dim
        
        self.coordinator = nn.Sequential(*coordinator_layers)
        
        # Multi-objective coordination
        self.objective_weights = nn.Parameter(
            torch.ones(3, device=self.device)  # accuracy, fairness, diversity
        )
        
        # Agent importance weighting (learned)
        self.agent_importance = nn.Parameter(
            torch.ones(self.num_agents, device=self.device)
        )
        
        # Hierarchical state aggregation
        self.hierarchical_aggregator = nn.Sequential(
            nn.Linear(comm_config['embed_dim'], comm_config['embed_dim'] // 2),
            nn.ReLU(),
            nn.Linear(comm_config['embed_dim'] // 2, comm_config['embed_dim'] // 4)
        )
        
        # Global context encoder
        self.global_context_encoder = nn.Sequential(
            nn.Linear(coord_config['output_dim'], comm_config['embed_dim']),
            nn.ReLU(),
            nn.Linear(comm_config['embed_dim'], comm_config['embed_dim'])
        )
        
        # Initialize weights
        self._init_weights()
        
        # Controller statistics
        self.controller_stats = {
            'coordination_steps': 0,
            'avg_agent_importance': defaultdict(list),
            'objective_weights_history': [],
        }
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize objective weights to balanced values
        nn.init.constant_(self.objective_weights, 1.0)
        nn.init.constant_(self.agent_importance, 1.0)
    
    def forward(
        self,
        agent_states: torch.Tensor,
        global_context: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through hierarchical MARL controller.
        
        Args:
            agent_states: Individual agent states [num_agents, embed_dim]
            global_context: Optional global context information
            edge_index: Optional communication graph structure
            return_components: Whether to return detailed components
            
        Returns:
            coordinated_output: Global coordination signal
            updated_agent_states: Updated agent states after communication
            components: Optional detailed component information
        """
        # Inter-agent communication
        updated_agent_states, attention_info = self.communication_layer(
            agent_states,
            edge_index=edge_index,
            return_attention=return_components
        )
        
        # Apply agent importance weighting
        importance_weights = F.softmax(self.agent_importance, dim=0)
        weighted_states = updated_agent_states * importance_weights.unsqueeze(-1)
        
        # Hierarchical aggregation
        hierarchical_features = []
        for i in range(self.num_agents):
            agent_feature = self.hierarchical_aggregator(weighted_states[i])
            hierarchical_features.append(agent_feature)
        
        hierarchical_repr = torch.stack(hierarchical_features).mean(dim=0)
        
        # Global coordination
        flattened_states = weighted_states.view(-1)
        coordinated_output = self.coordinator(flattened_states.unsqueeze(0)).squeeze(0)
        
        # Incorporate global context if provided
        if global_context is not None:
            global_context_encoded = self.global_context_encoder(global_context)
            coordinated_output = coordinated_output + global_context_encoded
        
        # Multi-objective weighting
        objective_weights = F.softmax(self.objective_weights, dim=0)
        
        # Update statistics
        self.controller_stats['coordination_steps'] += 1
        for i, weight in enumerate(importance_weights):
            self.controller_stats['avg_agent_importance'][i].append(weight.item())
        self.controller_stats['objective_weights_history'].append(objective_weights.detach().cpu().numpy())
        
        components = None
        if return_components:
            components = {
                'attention_info': attention_info,
                'agent_importance_weights': importance_weights,
                'objective_weights': objective_weights,
                'hierarchical_representation': hierarchical_repr,
                'communication_stats': self.communication_layer.get_communication_statistics()
            }
        
        return coordinated_output, updated_agent_states, components
    
    def update_agent_importance(self, performance_scores: torch.Tensor):
        """
        Update agent importance based on performance.
        
        Args:
            performance_scores: Performance scores for each agent [num_agents]
        """
        # Exponential moving average update
        alpha = 0.1
        current_importance = F.softmax(self.agent_importance, dim=0)
        new_importance = alpha * performance_scores + (1 - alpha) * current_importance
        
        # Update parameters
        with torch.no_grad():
            self.agent_importance.copy_(torch.log(new_importance + 1e-8))
    
    def update_objective_weights(self, objective_performance: torch.Tensor):
        """
        Update multi-objective weights based on performance.
        
        Args:
            objective_performance: Performance on [accuracy, fairness, diversity]
        """
        # Adaptive weight adjustment based on performance gaps
        target_performance = torch.ones_like(objective_performance)
        performance_gap = target_performance - objective_performance
        
        # Update weights to focus on underperforming objectives
        alpha = 0.05
        current_weights = F.softmax(self.objective_weights, dim=0)
        adjustment = alpha * performance_gap
        new_weights = current_weights + adjustment
        
        with torch.no_grad():
            self.objective_weights.copy_(torch.log(new_weights + 1e-8))
    
    def get_controller_statistics(self) -> Dict[str, Any]:
        """Get comprehensive controller statistics."""
        stats = self.controller_stats.copy()
        
        # Add communication layer stats
        stats.update(self.communication_layer.get_communication_statistics())
        
        # Compute average agent importance
        avg_importance = {}
        for agent_id, importance_list in stats['avg_agent_importance'].items():
            avg_importance[f'agent_{agent_id}'] = np.mean(importance_list[-10:])  # Last 10 steps
        
        stats['recent_avg_agent_importance'] = avg_importance
        
        return stats


# Alias for backward compatibility
MARLController = HierarchicalMARLController
MARL = HierarchicalMARLController

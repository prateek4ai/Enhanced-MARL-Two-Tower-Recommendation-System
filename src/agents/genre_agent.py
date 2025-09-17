"""
Enhanced Genre Agent Implementation for MARL Recommendation System
Integrates BUHS, contrastive learning, and stable rank regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from collections import defaultdict

from ..models.buhs import BUHSModule


class GenreAgent(nn.Module):
    """
    Genre-specific RL agent with advanced optimization techniques.
    
    Features:
    - Actor-Critic architecture with separate policy and value networks
    - BUHS (Biased User History Synthesis) integration for long-tail discovery
    - Contrastive learning support for representation enhancement
    - Stable rank regularization for balanced embedding capacity
    - Genre-specific state representation and action selection
    """
    
    def __init__(
        self,
        genre_name: str,
        feature_dim: int = 128,
        action_dim: int = 32,
        hidden_dims: list = [128, 64, 32],
        buhs_dim: int = 32,
        use_buhs: bool = True,
        use_contrastive: bool = True,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        """
        Initialize Genre Agent.
        
        Args:
            genre_name: Name of the genre this agent specializes in
            feature_dim: Input feature dimension
            action_dim: Output action space dimension (top-K recommendations)
            hidden_dims: Hidden layer dimensions for networks
            buhs_dim: Dimension of BUHS synthesized features
            use_buhs: Whether to use BUHS module
            use_contrastive: Whether to use contrastive learning
            dropout: Dropout rate
            device: Device for computation
        """
        super(GenreAgent, self).__init__()
        
        self.genre_name = genre_name
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.use_buhs = use_buhs
        self.use_contrastive = use_contrastive
        self.device = device
        
        # Base feature encoder for genre-specific user modeling
        self.base_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        
        # BUHS Module for long-tail item discovery
        if self.use_buhs:
            self.buhs = BUHSModule(
                embed_dim=feature_dim,
                hidden_dim=hidden_dims[0],
                output_dim=buhs_dim,
                num_heads=8,
                dropout=dropout
            )
            combined_dim = hidden_dims[2] + buhs_dim
        else:
            self.buhs = None
            combined_dim = hidden_dims[2]
        
        # Policy Network (Actor) - generates recommendation preferences
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim)
        )
        
        # Value Network (Critic) - estimates state value
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        # Contrastive learning projection head
        if self.use_contrastive:
            self.contrastive_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dims[2]),
                nn.ReLU(),
                nn.Linear(hidden_dims[2], hidden_dims[2] // 2)
            )
        
        # Genre-specific preference encoder
        self.genre_preference_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Track agent statistics for analysis
        self.stats = {
            'total_actions': 0,
            'avg_reward': 0.0,
            'genre_coverage': defaultdict(int),
            'exploration_rate': 1.0
        }
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_state_representation(
        self,
        user_features: torch.Tensor,
        genre_history: Optional[torch.Tensor] = None,
        user_context: Optional[Dict[str, torch.Tensor]] = None,
        item_popularities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate agent state from user information and context.
        
        Args:
            user_features: Base user features [batch_size, feature_dim]
            genre_history: User's genre-specific history [batch_size, seq_len, embed_dim]
            user_context: Additional context (demographics, temporal, etc.)
            item_popularities: Popularity scores for BUHS [batch_size, seq_len]
        
        Returns:
            Combined state representation [batch_size, combined_dim]
        """
        # Base feature encoding
        base_encoded = self.base_encoder(user_features)
        
        # BUHS enhancement if enabled
        if self.use_buhs and genre_history is not None:
            buhs_features = self.buhs(
                user_history=genre_history,
                item_popularities=item_popularities,
                alpha=1.0  # Inverse popularity weighting strength
            )
            combined_state = torch.cat([base_encoded, buhs_features], dim=-1)
        else:
            combined_state = base_encoded
        
        return combined_state
    
    def forward(
        self,
        state: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through actor-critic networks.
        
        Args:
            state: Combined state representation [batch_size, combined_dim]
            return_components: Whether to return additional components
        
        Returns:
            policy_logits: Action logits [batch_size, action_dim]
            value: State value estimates [batch_size]
            components: Optional dict with additional outputs
        """
        # Policy network forward pass
        policy_logits = self.policy_net(state)
        
        # Value network forward pass
        value = self.value_net(state).squeeze(-1)
        
        components = None
        if return_components:
            components = {}
            
            # Genre preference score
            genre_pref = self.genre_preference_net(state)
            components['genre_preference'] = genre_pref
            
            # Contrastive features
            if self.use_contrastive:
                contrastive_features = self.contrastive_head(state)
                components['contrastive_features'] = contrastive_features
        
        return policy_logits, value, components
    
    def select_action(
        self,
        state: torch.Tensor,
        exploration: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions based on current policy.
        
        Args:
            state: State representation [batch_size, combined_dim]
            exploration: Whether to use exploration
            temperature: Temperature for softmax (higher = more exploration)
            top_k: Optional top-k sampling
        
        Returns:
            actions: Selected actions [batch_size]
            log_probs: Log probabilities of selected actions [batch_size]
            values: State value estimates [batch_size]
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            policy_logits, values, _ = self.forward(state)
            
            # Apply temperature scaling
            scaled_logits = policy_logits / temperature
            
            # Compute action probabilities
            if top_k is not None:
                # Top-k sampling
                top_logits, top_indices = torch.topk(scaled_logits, top_k, dim=-1)
                top_probs = F.softmax(top_logits, dim=-1)
                
                if exploration:
                    selected_indices = torch.multinomial(top_probs, num_samples=1).squeeze(-1)
                    actions = top_indices.gather(-1, selected_indices.unsqueeze(-1)).squeeze(-1)
                else:
                    actions = top_indices[:, 0]  # Greedy selection
                
                # Get log probabilities
                all_probs = F.softmax(scaled_logits, dim=-1)
                log_probs = torch.log(all_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
            else:
                # Standard sampling
                probs = F.softmax(scaled_logits, dim=-1)
                
                if exploration:
                    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    actions = torch.argmax(probs, dim=-1)
                
                log_probs = torch.log(probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        # Update statistics
        self.stats['total_actions'] += actions.size(0)
        
        return actions, log_probs, values
    
    def compute_preference_score(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute genre preference score for coordination.
        
        Args:
            state: State representation [batch_size, combined_dim]
        
        Returns:
            preference_score: Genre preference [batch_size, 1]
        """
        return self.genre_preference_net(state)
    
    def compute_contrastive_loss(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss for representation learning.
        
        Args:
            anchor_features: Anchor representations [batch_size, feature_dim]
            positive_features: Positive pair representations [batch_size, feature_dim]
            negative_features: Negative pair representations [batch_size, num_neg, feature_dim]
            temperature: Temperature parameter for contrastive loss
        
        Returns:
            contrastive_loss: InfoNCE loss value
        """
        if not self.use_contrastive:
            return torch.tensor(0.0, device=self.device)
        
        # Normalize features
        anchor = F.normalize(anchor_features, dim=-1)
        positive = F.normalize(positive_features, dim=-1)
        negative = F.normalize(negative_features, dim=-1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # [batch_size]
        neg_sim = torch.bmm(anchor.unsqueeze(1), negative.transpose(-2, -1)).squeeze(1) / temperature  # [batch_size, num_neg]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [batch_size, 1 + num_neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return contrastive_loss
    
    def compute_stable_rank_penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute stable rank regularization penalty.
        
        Args:
            embeddings: Embedding matrix [batch_size, embed_dim]
        
        Returns:
            penalty: Stable rank penalty (negative for minimization)
        """
        # Compute SVD
        try:
            U, S, V = torch.svd(embeddings)
            
            # Stable rank = ||S||_F^2 / ||S||_2^2
            frobenius_norm_sq = torch.sum(S ** 2)
            spectral_norm_sq = torch.max(S) ** 2
            
            stable_rank = frobenius_norm_sq / (spectral_norm_sq + 1e-8)
            
            # Return negative for minimization (we want to maximize stable rank)
            penalty = -stable_rank
            
        except RuntimeError:
            # Fallback if SVD fails
            penalty = torch.tensor(0.0, device=self.device)
        
        return penalty
    
    def update_statistics(self, reward: float):
        """Update agent statistics."""
        self.stats['avg_reward'] = (
            0.95 * self.stats['avg_reward'] + 0.05 * reward
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        return {
            'genre': self.genre_name,
            'total_actions': self.stats['total_actions'],
            'avg_reward': self.stats['avg_reward'],
            'exploration_rate': self.stats['exploration_rate']
        }
    
    def set_exploration_rate(self, rate: float):
        """Set exploration rate for action selection."""
        self.stats['exploration_rate'] = max(0.01, min(1.0, rate))


class GenreAgentManager:
    """
    Manager class for coordinating multiple genre agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.device = config.get('device', 'cuda')
        
        # Create agents for each genre
        for genre in config['dataset']['genres']:
            self.agents[genre] = GenreAgent(
                genre_name=genre,
                feature_dim=config['marl_controller']['genre_agents']['state_dim'],
                action_dim=config['marl_controller']['genre_agents']['action_dim'],
                hidden_dims=config['marl_controller']['genre_agents']['hidden_dims'],
                buhs_dim=config['buhs']['output_dim'],
                use_buhs=config['buhs']['enabled'],
                device=self.device
            )
    
    def get_agent(self, genre: str) -> GenreAgent:
        """Get agent for specific genre."""
        return self.agents[genre]
    
    def get_all_agents(self) -> Dict[str, GenreAgent]:
        """Get all genre agents."""
        return self.agents
    
    def update_all_exploration_rates(self, rate: float):
        """Update exploration rates for all agents."""
        for agent in self.agents.values():
            agent.set_exploration_rate(rate)

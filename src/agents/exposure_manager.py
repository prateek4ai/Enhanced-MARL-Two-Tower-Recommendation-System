"""
Exposure Manager (GINI Agent) Implementation for MARL Recommendation System
Monitors item exposure distribution and provides fairness adjustments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class ExposureManager(nn.Module):
    """
    Advanced Exposure Manager (GINI Agent) for fairness coordination.
    
    Features:
    - Real-time GINI coefficient computation
    - Genre-aware fairness adjustments
    - Multi-objective fairness optimization (item-level, genre-level, demographic)
    - Adaptive threshold management
    - Long-term fairness tracking
    - Integration with genre agents for coordinated optimization
    """
    
    def __init__(
        self,
        num_genres: int = 18,
        num_items: int = 3706,
        hidden_dims: List[int] = [64, 32],
        gini_threshold: float = 0.6,
        adjustment_strength: float = 1.0,
        smoothing_factor: float = 0.9,
        history_window: int = 1000,
        device: str = 'cuda'
    ):
        """
        Initialize Exposure Manager.
        
        Args:
            num_genres: Number of genres in the system
            num_items: Total number of items in catalog
            hidden_dims: Hidden layer dimensions for neural networks
            gini_threshold: Target GINI coefficient threshold
            adjustment_strength: Strength of fairness adjustments
            smoothing_factor: Exponential smoothing for exposure tracking
            history_window: Window size for exposure history tracking
            device: Device for computation
        """
        super(ExposureManager, self).__init__()
        
        self.num_genres = num_genres
        self.num_items = num_items
        self.gini_threshold = gini_threshold
        self.adjustment_strength = adjustment_strength
        self.smoothing_factor = smoothing_factor
        self.history_window = history_window
        self.device = device
        
        # Neural network for fairness coefficient computation
        self.fairness_net = nn.Sequential(
            nn.Linear(num_genres + 3, hidden_dims[0]),  # +3 for GINI, coverage, diversity
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dims[1], num_genres),
            nn.Sigmoid()  # Output adjustment coefficients in [0,1]
        )
        
        # Genre-specific adjustment networks
        self.genre_adjusters = nn.ModuleDict()
        for i in range(num_genres):
            self.genre_adjusters[str(i)] = nn.Sequential(
                nn.Linear(4, 16),  # exposure, popularity, recency, diversity
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        
        # Demographic fairness network
        self.demographic_net = nn.Sequential(
            nn.Linear(num_genres * 3, 32),  # genre × (age, gender, occupation)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_genres)
        )
        
        # Initialize tracking variables
        self.reset_tracking()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def reset_tracking(self):
        """Reset all tracking variables."""
        # Genre-level exposure tracking
        self.genre_exposure_counts = torch.zeros(self.num_genres, device=self.device)
        self.genre_exposure_history = deque(maxlen=self.history_window)
        
        # Item-level exposure tracking
        self.item_exposure_counts = torch.zeros(self.num_items, device=self.device)
        self.item_exposure_history = deque(maxlen=self.history_window)
        
        # Long-tail tracking
        self.tail_exposure_count = 0
        self.head_exposure_count = 0
        
        # Demographic fairness tracking
        self.demographic_exposure = {
            'age': defaultdict(lambda: torch.zeros(self.num_genres, device=self.device)),
            'gender': defaultdict(lambda: torch.zeros(self.num_genres, device=self.device)),
            'occupation': defaultdict(lambda: torch.zeros(self.num_genres, device=self.device))
        }
        
        # Performance tracking
        self.gini_history = deque(maxlen=100)
        self.adjustment_history = deque(maxlen=100)
        
        # Statistics
        self.total_recommendations = 0
        self.fairness_interventions = 0
        
        logger.info("Exposure tracking reset completed")
    
    def compute_gini_coefficient(
        self, 
        exposures: torch.Tensor, 
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute GINI coefficient for exposure distribution.
        
        Args:
            exposures: Exposure counts tensor [num_items or num_genres]
            epsilon: Small value to avoid division by zero
            
        Returns:
            GINI coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Add small epsilon to avoid issues with zero exposures
        exposures = exposures + epsilon
        
        # Sort exposures in ascending order
        sorted_exposures, _ = torch.sort(exposures)
        
        n = len(sorted_exposures)
        index = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        
        # GINI coefficient formula
        gini = (2 * torch.sum(index * sorted_exposures)) / (n * torch.sum(sorted_exposures)) - (n + 1) / n
        
        return torch.clamp(gini, 0.0, 1.0)
    
    def update_exposure_tracking(
        self,
        recommendations: torch.Tensor,
        genre_assignments: torch.Tensor,
        item_popularities: Optional[torch.Tensor] = None,
        user_demographics: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Update exposure tracking with new recommendations.
        
        Args:
            recommendations: Recommended item IDs [batch_size, top_k]
            genre_assignments: Genre assignments for recommended items [batch_size, top_k, num_genres]
            item_popularities: Popularity scores for items [batch_size, top_k]
            user_demographics: User demographic information
        """
        batch_size, top_k = recommendations.shape
        
        # Flatten recommendations for easier processing
        flat_recs = recommendations.view(-1)
        flat_genres = genre_assignments.view(-1, self.num_genres)
        
        # Update item-level exposure counts
        for item_id in flat_recs:
            if 0 <= item_id < self.num_items:
                self.item_exposure_counts[item_id] += 1
        
        # Update genre-level exposure counts
        genre_exposures = torch.sum(flat_genres, dim=0)
        self.genre_exposure_counts += genre_exposures
        
        # Track long-tail vs head exposure
        if item_popularities is not None:
            flat_pops = item_popularities.view(-1)
            tail_mask = flat_pops < 0.2  # Bottom 20% popularity threshold
            self.tail_exposure_count += torch.sum(tail_mask).item()
            self.head_exposure_count += torch.sum(~tail_mask).item()
        
        # Update demographic exposure tracking
        if user_demographics is not None:
            for demo_type, demo_values in user_demographics.items():
                if demo_type in self.demographic_exposure:
                    for i, demo_val in enumerate(demo_values):
                        self.demographic_exposure[demo_type][demo_val.item()] += torch.sum(flat_genres[i * top_k:(i + 1) * top_k], dim=0)
        
        # Update histories
        self.genre_exposure_history.append(genre_exposures.cpu().numpy())
        self.item_exposure_history.append(flat_recs.cpu().numpy())
        
        self.total_recommendations += batch_size * top_k
    
    def get_fairness_state(self) -> torch.Tensor:
        """
        Compute current fairness state representation.
        
        Returns:
            Fairness state tensor [num_genres + 3]
        """
        # Normalize genre exposures
        total_genre_exposure = torch.sum(self.genre_exposure_counts) + 1e-8
        normalized_genre_exposure = self.genre_exposure_counts / total_genre_exposure
        
        # Compute fairness metrics
        genre_gini = self.compute_gini_coefficient(self.genre_exposure_counts)
        item_gini = self.compute_gini_coefficient(self.item_exposure_counts)
        
        # Compute coverage (percentage of items with non-zero exposure)
        coverage = torch.sum(self.item_exposure_counts > 0).float() / self.num_items
        
        # Compute diversity (entropy of genre distribution)
        genre_probs = normalized_genre_exposure + 1e-8
        diversity = -torch.sum(genre_probs * torch.log(genre_probs))
        
        # Combine into state representation
        fairness_state = torch.cat([
            normalized_genre_exposure,
            torch.tensor([genre_gini, coverage, diversity], device=self.device)
        ])
        
        return fairness_state
    
    def compute_genre_adjustments(
        self,
        genre_exposures: torch.Tensor,
        item_popularities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute genre-specific fairness adjustments.
        
        Args:
            genre_exposures: Current genre exposure counts [num_genres]
            item_popularities: Item popularity scores for context
            
        Returns:
            Genre adjustment coefficients [num_genres]
        """
        adjustments = torch.zeros(self.num_genres, device=self.device)
        
        total_exposure = torch.sum(genre_exposures) + 1e-8
        normalized_exposures = genre_exposures / total_exposure
        expected_exposure = 1.0 / self.num_genres  # Uniform expectation
        
        for i in range(self.num_genres):
            # Current exposure stats for this genre
            current_exposure = normalized_exposures[i]
            exposure_deficit = expected_exposure - current_exposure
            
            # Recency: how recently was this genre exposed
            recent_exposure = 0.0
            if len(self.genre_exposure_history) > 0:
                recent_history = list(self.genre_exposure_history)[-10:]  # Last 10 batches
                recent_exposure = np.mean([h[i] for h in recent_history])
            
            # Popularity context
            popularity_factor = 1.0
            if item_popularities is not None:
                # Items in this genre tend to be popular/unpopular
                popularity_factor = 1.0 - torch.mean(item_popularities).item()
            
            # Diversity factor
            diversity_factor = 1.0 - current_exposure  # Promote less exposed genres
            
            # Combine factors
            genre_input = torch.tensor([
                current_exposure,
                popularity_factor,
                recent_exposure,
                diversity_factor
            ], device=self.device)
            
            adjustment = self.genre_adjusters[str(i)](genre_input.unsqueeze(0)).squeeze()
            
            # Scale adjustment based on deficit
            if exposure_deficit > 0:  # Under-exposed genre
                adjustment = adjustment * (1 + exposure_deficit * self.adjustment_strength)
            else:  # Over-exposed genre
                adjustment = adjustment * (1 + exposure_deficit * 0.5)  # Milder reduction
            
            adjustments[i] = torch.clamp(adjustment, 0.1, 2.0)  # Reasonable bounds
        
        return adjustments
    
    def forward(
        self,
        genre_exposures: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass to compute fairness adjustments.
        
        Args:
            genre_exposures: Optional external genre exposure vector
            return_components: Whether to return detailed components
            
        Returns:
            fairness_adjustments: Adjustment coefficients [num_genres]
            components: Optional dict with detailed fairness information
        """
        # Use current tracked exposures if not provided
        if genre_exposures is None:
            genre_exposures = self.genre_exposure_counts
        
        # Get current fairness state
        fairness_state = self.get_fairness_state()
        
        # Compute main fairness adjustments
        main_adjustments = self.fairness_net(fairness_state.unsqueeze(0)).squeeze(0)
        
        # Compute genre-specific fine-grained adjustments
        genre_adjustments = self.compute_genre_adjustments(genre_exposures)
        
        # Combine adjustments
        combined_adjustments = main_adjustments * genre_adjustments
        
        # Apply softmax to ensure they sum to reasonable values
        fairness_adjustments = F.softmax(combined_adjustments, dim=0) * self.num_genres
        
        components = None
        if return_components:
            current_gini = self.compute_gini_coefficient(self.genre_exposure_counts)
            item_gini = self.compute_gini_coefficient(self.item_exposure_counts)
            
            components = {
                'genre_gini': current_gini,
                'item_gini': item_gini,
                'fairness_state': fairness_state,
                'main_adjustments': main_adjustments,
                'genre_adjustments': genre_adjustments,
                'total_recommendations': self.total_recommendations,
                'fairness_interventions': self.fairness_interventions,
                'tail_ratio': self.tail_exposure_count / (self.total_recommendations + 1e-8)
            }
        
        # Track adjustment history
        self.adjustment_history.append(fairness_adjustments.cpu().numpy())
        
        # Check if intervention is needed
        current_gini = self.compute_gini_coefficient(self.genre_exposure_counts)
        if current_gini > self.gini_threshold:
            self.fairness_interventions += 1
        
        self.gini_history.append(current_gini.item())
        
        return fairness_adjustments, components
    
    def compute_reward(self) -> float:
        """
        Compute reward for the exposure manager based on fairness improvement.
        
        Returns:
            Reward value (higher is better)
        """
        if len(self.gini_history) < 2:
            return 0.0
        
        # Reward for reducing GINI coefficient
        current_gini = self.gini_history[-1]
        previous_gini = self.gini_history[-2]
        gini_improvement = previous_gini - current_gini
        
        # Reward for maintaining coverage
        coverage = torch.sum(self.item_exposure_counts > 0).float() / self.num_items
        coverage_reward = coverage.item()
        
        # Reward for long-tail promotion
        tail_ratio = self.tail_exposure_count / (self.total_recommendations + 1e-8)
        tail_reward = min(tail_ratio, 0.3) / 0.3  # Cap at 30% tail exposure
        
        # Combined reward
        reward = (
            2.0 * gini_improvement +      # Primary: GINI improvement
            1.0 * coverage_reward +       # Secondary: coverage maintenance
            0.5 * tail_reward            # Tertiary: long-tail promotion
        )
        
        return float(reward)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive exposure manager statistics."""
        current_gini = self.compute_gini_coefficient(self.genre_exposure_counts)
        item_gini = self.compute_gini_coefficient(self.item_exposure_counts)
        coverage = torch.sum(self.item_exposure_counts > 0).float() / self.num_items
        
        return {
            'current_genre_gini': current_gini.item(),
            'current_item_gini': item_gini.item(),
            'item_coverage': coverage.item(),
            'total_recommendations': self.total_recommendations,
            'fairness_interventions': self.fairness_interventions,
            'intervention_rate': self.fairness_interventions / max(1, len(self.gini_history)),
            'avg_gini_last_10': np.mean(list(self.gini_history)[-10:]) if self.gini_history else 0.0,
            'tail_exposure_ratio': self.tail_exposure_count / (self.total_recommendations + 1e-8),
            'genre_exposure_distribution': self.genre_exposure_counts.cpu().numpy().tolist(),
            'gini_threshold': self.gini_threshold,
            'below_threshold': current_gini.item() < self.gini_threshold
        }
    
    def adjust_threshold(self, new_threshold: float):
        """Dynamically adjust the GINI threshold."""
        self.gini_threshold = max(0.1, min(0.9, new_threshold))
        logger.info(f"GINI threshold adjusted to {self.gini_threshold}")
    
    def save_state(self, filepath: str):
        """Save exposure manager state for analysis."""
        state = {
            'genre_exposure_counts': self.genre_exposure_counts.cpu().numpy(),
            'item_exposure_counts': self.item_exposure_counts.cpu().numpy(),
            'gini_history': list(self.gini_history),
            'adjustment_history': list(self.adjustment_history),
            'statistics': self.get_statistics()
        }
        torch.save(state, filepath)
        logger.info(f"Exposure manager state saved to {filepath}")


class ExposureManagerController:
    """
    Controller for managing multiple exposure managers or advanced coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Main exposure manager
        self.exposure_manager = ExposureManager(
            num_genres=config['dataset']['num_genres'],
            num_items=config['dataset']['num_items'],
            hidden_dims=config['marl_controller']['exposure_manager']['hidden_dims'],
            gini_threshold=config['marl_controller']['exposure_manager']['gini_threshold'],
            device=self.device
        )
        
        # Optional: demographic-specific managers
        self.demographic_managers = {}
        if config.get('fairness', {}).get('demographic_fairness', False):
            for demo_type in ['age', 'gender', 'occupation']:
                self.demographic_managers[demo_type] = ExposureManager(
                    num_genres=config['dataset']['num_genres'],
                    num_items=config['dataset']['num_items'],
                    gini_threshold=config['marl_controller']['exposure_manager']['gini_threshold'] * 1.1,  # Slightly more lenient
                    device=self.device
                )
    
    def get_main_manager(self) -> ExposureManager:
        """Get the main exposure manager."""
        return self.exposure_manager
    
    def get_demographic_manager(self, demo_type: str) -> Optional[ExposureManager]:
        """Get demographic-specific exposure manager."""
        return self.demographic_managers.get(demo_type)
    
    def coordinate_fairness(
        self,
        recommendations: torch.Tensor,
        genre_assignments: torch.Tensor,
        user_demographics: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Coordinate fairness adjustments across all managers.
        
        Returns:
            Combined fairness adjustments
        """
        # Get main adjustments
        main_adjustments, _ = self.exposure_manager(return_components=True)
        
        # Get demographic adjustments if available
        demographic_adjustments = []
        if user_demographics and self.demographic_managers:
            for demo_type, demo_manager in self.demographic_managers.items():
                demo_adj, _ = demo_manager()
                demographic_adjustments.append(demo_adj)
        
        # Combine adjustments
        if demographic_adjustments:
            # Weighted combination
            combined = 0.7 * main_adjustments
            for demo_adj in demographic_adjustments:
                combined += 0.3 / len(demographic_adjustments) * demo_adj
            return combined
        else:
            return main_adjustments

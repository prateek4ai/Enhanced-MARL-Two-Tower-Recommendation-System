"""
Fair Sampler Implementation for Enhanced MARL Two-Tower Recommendation System

Implements multiple sampling strategies to address popularity bias and promote fairness:
- Inverse frequency sampling for long-tail promotion
- Genre-aware sampling for multi-agent coordination
- Popularity-aware sampling with temperature control
- Integration with BUHS and exposure manager
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
import random
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SamplingMethod(Enum):
    """Enumeration of available sampling methods"""
    UNIFORM = "uniform"
    POPULARITY_AWARE = "popularity_aware"
    INVERSE_FREQUENCY = "inverse_frequency"
    GENRE_AWARE_INVERSE_FREQUENCY = "genre_aware_inverse_frequency"
    BUHS_BIASED = "buhs_biased"

class BaseSampler(ABC):
    """Abstract base class for sampling strategies"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.total_samples = 0
        
    @abstractmethod
    def sample(self, candidates: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample items from candidates"""
        pass
    
    @abstractmethod
    def update_statistics(self, sampled_items: torch.Tensor, **kwargs):
        """Update internal statistics based on sampled items"""
        pass

class UniformSampler(BaseSampler):
    """Uniform random sampling"""
    
    def sample(self, candidates: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Uniform random sampling without replacement"""
        if len(candidates) <= num_samples:
            return candidates
        
        indices = torch.randperm(len(candidates), device=self.device)[:num_samples]
        return candidates[indices]
    
    def update_statistics(self, sampled_items: torch.Tensor, **kwargs):
        """No statistics to update for uniform sampling"""
        self.total_samples += len(sampled_items)

class PopularityAwareSampler(BaseSampler):
    """Popularity-proportional sampling with optional temperature control"""
    
    def __init__(self, temperature: float = 1.0, device: str = 'cuda'):
        super().__init__(device)
        self.temperature = temperature
        self.item_popularity = defaultdict(float)
    
    def sample(self, candidates: torch.Tensor, num_samples: int, 
              popularities: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Sample based on item popularity scores"""
        if popularities is None:
            # Use stored popularity scores
            pop_scores = torch.tensor([
                self.item_popularity[item.item()] for item in candidates
            ], device=self.device, dtype=torch.float32)
        else:
            pop_scores = popularities.to(self.device)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            pop_scores = pop_scores / self.temperature
        
        # Softmax for sampling probabilities
        probs = torch.softmax(pop_scores, dim=0)
        
        # Sample with replacement using multinomial
        sampled_indices = torch.multinomial(probs, num_samples, replacement=True)
        return candidates[sampled_indices]
    
    def update_statistics(self, sampled_items: torch.Tensor, 
                         popularities: Optional[torch.Tensor] = None, **kwargs):
        """Update popularity statistics"""
        if popularities is not None:
            for item, pop in zip(sampled_items, popularities):
                self.item_popularity[item.item()] = pop.item()
        self.total_samples += len(sampled_items)

class InverseFrequencySampler(BaseSampler):
    """Inverse frequency sampling to promote long-tail items"""
    
    def __init__(self, temperature: float = 1.0, min_frequency: int = 1, 
                 smoothing_factor: float = 0.1, device: str = 'cuda'):
        super().__init__(device)
        self.temperature = temperature
        self.min_frequency = min_frequency
        self.smoothing_factor = smoothing_factor
        self.item_frequencies = defaultdict(int)
        self.frequency_history = deque(maxlen=1000)  # Track recent frequencies
    
    def sample(self, candidates: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample with probability inversely proportional to frequency"""
        # Calculate inverse frequency weights
        weights = []
        for item in candidates:
            freq = max(self.item_frequencies[item.item()], self.min_frequency)
            weight = 1.0 / (freq + self.smoothing_factor)
            weights.append(weight)
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            weights = weights / self.temperature
        
        # Normalize to probabilities
        probs = weights / weights.sum()
        
        # Sample without replacement
        sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
        return candidates[sampled_indices]
    
    def update_statistics(self, sampled_items: torch.Tensor, **kwargs):
        """Update frequency statistics"""
        for item in sampled_items:
            item_id = item.item()
            self.item_frequencies[item_id] += 1
            self.frequency_history.append(item_id)
        self.total_samples += len(sampled_items)
    
    def get_frequency_stats(self) -> Dict[str, float]:
        """Get frequency distribution statistics"""
        if not self.item_frequencies:
            return {}
        
        frequencies = list(self.item_frequencies.values())
        return {
            'mean_frequency': np.mean(frequencies),
            'std_frequency': np.std(frequencies),
            'min_frequency': min(frequencies),
            'max_frequency': max(frequencies),
            'unique_items': len(self.item_frequencies)
        }

class GenreAwareInverseFrequencySampler(BaseSampler):
    """Genre-aware inverse frequency sampling for multi-agent systems"""
    
    def __init__(self, num_genres: int, temperature: float = 0.8, 
                 genre_boost_factors: Optional[Dict[str, float]] = None,
                 device: str = 'cuda'):
        super().__init__(device)
        self.num_genres = num_genres
        self.temperature = temperature
        self.genre_boost_factors = genre_boost_factors or {}
        
        # Per-genre frequency tracking
        self.genre_item_frequencies = {
            i: defaultdict(int) for i in range(num_genres)
        }
        self.genre_total_samples = defaultdict(int)
        
        # Cross-genre frequency tracking
        self.global_item_frequencies = defaultdict(int)
    
    def sample(self, candidates: torch.Tensor, num_samples: int,
              genre_id: Optional[int] = None, 
              item_genres: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Genre-aware sampling with inverse frequency weighting"""
        
        if genre_id is not None:
            # Single genre sampling
            return self._sample_single_genre(candidates, num_samples, genre_id)
        elif item_genres is not None:
            # Multi-genre sampling
            return self._sample_multi_genre(candidates, num_samples, item_genres)
        else:
            # Fallback to global inverse frequency
            return self._sample_global(candidates, num_samples)
    
    def _sample_single_genre(self, candidates: torch.Tensor, num_samples: int, 
                           genre_id: int) -> torch.Tensor:
        """Sample for a specific genre"""
        genre_frequencies = self.genre_item_frequencies[genre_id]
        
        weights = []
        for item in candidates:
            item_id = item.item()
            genre_freq = max(genre_frequencies[item_id], 1)
            global_freq = max(self.global_item_frequencies[item_id], 1)
            
            # Combine genre-specific and global frequencies
            combined_freq = 0.7 * genre_freq + 0.3 * global_freq
            weight = 1.0 / combined_freq
            
            # Apply genre-specific boost
            genre_name = f"genre_{genre_id}"
            if genre_name in self.genre_boost_factors:
                weight *= self.genre_boost_factors[genre_name]
            
            weights.append(weight)
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights = weights / self.temperature
        probs = weights / weights.sum()
        
        sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
        return candidates[sampled_indices]
    
    def _sample_multi_genre(self, candidates: torch.Tensor, num_samples: int,
                          item_genres: torch.Tensor) -> torch.Tensor:
        """Sample considering multiple genres per item"""
        weights = []
        
        for i, item in enumerate(candidates):
            item_id = item.item()
            item_genre_vector = item_genres[i]  # [num_genres] binary vector
            
            # Calculate weighted frequency across genres
            weighted_freq = 0.0
            active_genres = 0
            
            for genre_id in range(self.num_genres):
                if item_genre_vector[genre_id] > 0:  # Item belongs to this genre
                    genre_freq = max(self.genre_item_frequencies[genre_id][item_id], 1)
                    weighted_freq += genre_freq * item_genre_vector[genre_id]
                    active_genres += 1
            
            if active_genres > 0:
                weighted_freq /= active_genres
            else:
                weighted_freq = max(self.global_item_frequencies[item_id], 1)
            
            weight = 1.0 / weighted_freq
            weights.append(weight)
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights = weights / self.temperature
        probs = weights / weights.sum()
        
        sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
        return candidates[sampled_indices]
    
    def _sample_global(self, candidates: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Fallback global sampling"""
        weights = []
        for item in candidates:
            freq = max(self.global_item_frequencies[item.item()], 1)
            weights.append(1.0 / freq)
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        probs = weights / weights.sum()
        
        sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
        return candidates[sampled_indices]
    
    def update_statistics(self, sampled_items: torch.Tensor, 
                         genre_id: Optional[int] = None,
                         item_genres: Optional[torch.Tensor] = None, **kwargs):
        """Update genre-aware frequency statistics"""
        for i, item in enumerate(sampled_items):
            item_id = item.item()
            
            # Update global frequency
            self.global_item_frequencies[item_id] += 1
            
            # Update genre-specific frequencies
            if genre_id is not None:
                self.genre_item_frequencies[genre_id][item_id] += 1
                self.genre_total_samples[genre_id] += 1
            elif item_genres is not None:
                item_genre_vector = item_genres[i]
                for genre_idx in range(self.num_genres):
                    if item_genre_vector[genre_idx] > 0:
                        self.genre_item_frequencies[genre_idx][item_id] += 1
                        self.genre_total_samples[genre_idx] += 1
        
        self.total_samples += len(sampled_items)

class BUHSBiasedSampler(BaseSampler):
    """BUHS-integrated sampler for long-tail promotion"""
    
    def __init__(self, alpha: float = 1.0, temperature: float = 0.1, device: str = 'cuda'):
        super().__init__(device)
        self.alpha = alpha  # Inverse popularity weighting strength
        self.temperature = temperature
        self.item_popularity_scores = {}
    
    def sample(self, candidates: torch.Tensor, num_samples: int,
              item_popularities: torch.Tensor, **kwargs) -> torch.Tensor:
        """BUHS biased sampling toward tail items"""
        # BUHS sampling formula: P(item) ∝ (1/popularity)^alpha
        inv_popularities = 1.0 / (item_popularities + 1e-8)
        weights = torch.pow(inv_popularities, self.alpha)
        
        # Apply temperature scaling
        weights = weights / self.temperature
        probs = torch.softmax(weights, dim=0)
        
        # Sample with replacement (BUHS allows duplicates)
        sampled_indices = torch.multinomial(probs, num_samples, replacement=True)
        return candidates[sampled_indices]
    
    def update_statistics(self, sampled_items: torch.Tensor,
                         item_popularities: torch.Tensor, **kwargs):
        """Update popularity statistics"""
        for item, pop in zip(sampled_items, item_popularities):
            self.item_popularity_scores[item.item()] = pop.item()
        self.total_samples += len(sampled_items)

class FairSampler:
    """
    Enhanced Fair Sampler for MARL Two-Tower Recommendation System
    
    Features:
    - Multiple sampling strategies
    - Genre-aware sampling for multi-agent systems
    - Integration with BUHS and exposure manager
    - Configurable via YAML configuration
    - Real-time statistics and monitoring
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Extract sampling configuration
        sampling_config = config.get('fair_sampling', {})
        self.enabled = sampling_config.get('enabled', True)
        self.method = SamplingMethod(sampling_config.get('method', 'inverse_frequency'))
        self.temperature = sampling_config.get('temperature', 1.0)
        self.update_frequency = sampling_config.get('update_frequency', 100)
        
        # Initialize the appropriate sampler
        self.sampler = self._create_sampler()
        
        # Statistics tracking
        self.sampling_stats = {
            'total_calls': 0,
            'total_samples': 0,
            'method_usage': defaultdict(int),
            'genre_usage': defaultdict(int)
        }
        
        # Update counter
        self.update_counter = 0
        
        logger.info(f"FairSampler initialized with method: {self.method.value}")
    
    def _create_sampler(self) -> BaseSampler:
        """Create the appropriate sampler based on configuration"""
        if self.method == SamplingMethod.UNIFORM:
            return UniformSampler(self.device)
        
        elif self.method == SamplingMethod.POPULARITY_AWARE:
            popularity_alpha = self.config.get('fair_sampling', {}).get('popularity_alpha', 0.75)
            return PopularityAwareSampler(
                temperature=self.temperature, 
                device=self.device
            )
        
        elif self.method == SamplingMethod.INVERSE_FREQUENCY:
            return InverseFrequencySampler(
                temperature=self.temperature,
                min_frequency=self.config.get('fair_sampling', {}).get('min_frequency', 1),
                device=self.device
            )
        
        elif self.method == SamplingMethod.GENRE_AWARE_INVERSE_FREQUENCY:
            genre_config = self.config.get('fair_sampling', {}).get('genre_importance', {})
            return GenreAwareInverseFrequencySampler(
                num_genres=self.config['dataset']['num_genres'],
                temperature=self.temperature,
                genre_boost_factors=genre_config,
                device=self.device
            )
        
        elif self.method == SamplingMethod.BUHS_BIASED:
            buhs_config = self.config.get('buhs', {})
            return BUHSBiasedSampler(
                alpha=buhs_config.get('alpha', 1.0),
                temperature=self.temperature,
                device=self.device
            )
        
        else:
            raise ValueError(f"Unsupported sampling method: {self.method}")
    
    def sample_negatives(self, positive_items: torch.Tensor, 
                        all_items: torch.Tensor,
                        num_negatives: int,
                        genre_id: Optional[int] = None,
                        item_genres: Optional[torch.Tensor] = None,
                        item_popularities: Optional[torch.Tensor] = None,
                        **kwargs) -> torch.Tensor:
        """
        Sample negative items for training
        
        Args:
            positive_items: Items that are positive for this user
            all_items: All available items to sample from
            num_negatives: Number of negative samples needed
            genre_id: Optional genre ID for genre-specific sampling
            item_genres: Optional genre vectors for multi-genre sampling
            item_popularities: Optional popularity scores for BUHS sampling
            
        Returns:
            Sampled negative items
        """
        if not self.enabled:
            # Fallback to uniform sampling
            return self._uniform_sample(all_items, positive_items, num_negatives)
        
        # Get candidate negative items (exclude positives)
        positive_set = set(positive_items.cpu().numpy())
        candidates_mask = torch.tensor([
            item.item() not in positive_set for item in all_items
        ], device=self.device)
        candidates = all_items[candidates_mask]
        
        if len(candidates) == 0:
            logger.warning("No negative candidates available")
            return torch.empty(0, dtype=all_items.dtype, device=self.device)
        
        # Adjust num_negatives if we don't have enough candidates
        num_negatives = min(num_negatives, len(candidates))
        
        # Prepare sampling arguments
        sample_kwargs = {
            'genre_id': genre_id,
            'item_genres': item_genres[candidates_mask] if item_genres is not None else None,
            'item_popularities': item_popularities[candidates_mask] if item_popularities is not None else None,
            **kwargs
        }
        
        # Sample using the configured method
        sampled_negatives = self.sampler.sample(
            candidates, num_negatives, **sample_kwargs
        )
        
        # Update statistics
        self.sampler.update_statistics(sampled_negatives, **sample_kwargs)
        self._update_stats(genre_id, len(sampled_negatives))
        
        return sampled_negatives
    
    def sample_for_buhs(self, user_history: torch.Tensor,
                       item_popularities: torch.Tensor,
                       sample_size: int,
                       alpha: float = 1.0) -> torch.Tensor:
        """
        BUHS-specific sampling for user history synthesis
        
        Args:
            user_history: User's interaction history item IDs
            item_popularities: Popularity scores for history items
            sample_size: Number of items to sample
            alpha: Inverse popularity weighting strength
            
        Returns:
            Sampled items for BUHS synthesis
        """
        if len(user_history) <= sample_size:
            return user_history
        
        # BUHS sampling: P(item) ∝ (1/popularity)^alpha
        inv_popularities = 1.0 / (item_popularities + 1e-8)
        weights = torch.pow(inv_popularities, alpha)
        
        # Apply temperature
        weights = weights / self.temperature
        probs = torch.softmax(weights, dim=0)
        
        # Sample with replacement (BUHS allows duplicates)
        sampled_indices = torch.multinomial(probs, sample_size, replacement=True)
        return user_history[sampled_indices]
    
    def _uniform_sample(self, all_items: torch.Tensor, 
                       positive_items: torch.Tensor,
                       num_samples: int) -> torch.Tensor:
        """Fallback uniform sampling"""
        positive_set = set(positive_items.cpu().numpy())
        candidates = torch.tensor([
            item for item in all_items if item.item() not in positive_set
        ], device=self.device)
        
        if len(candidates) <= num_samples:
            return candidates
        
        indices = torch.randperm(len(candidates))[:num_samples]
        return candidates[indices]
    
    def _update_stats(self, genre_id: Optional[int], num_samples: int):
        """Update sampling statistics"""
        self.sampling_stats['total_calls'] += 1
        self.sampling_stats['total_samples'] += num_samples
        self.sampling_stats['method_usage'][self.method.value] += 1
        
        if genre_id is not None:
            self.sampling_stats['genre_usage'][genre_id] += 1
        
        self.update_counter += 1
    
    def should_update_frequencies(self) -> bool:
        """Check if frequency statistics should be updated"""
        return self.update_counter % self.update_frequency == 0
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sampling statistics"""
        base_stats = dict(self.sampling_stats)
        
        # Add sampler-specific statistics
        if hasattr(self.sampler, 'get_frequency_stats'):
            base_stats['frequency_stats'] = self.sampler.get_frequency_stats()
        
        if hasattr(self.sampler, 'genre_total_samples'):
            base_stats['genre_samples'] = dict(self.sampler.genre_total_samples)
        
        base_stats['current_method'] = self.method.value
        base_stats['total_sampler_calls'] = self.sampler.total_samples
        
        return base_stats
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update sampling configuration dynamically"""
        old_method = self.method
        
        # Update configuration
        sampling_config = new_config.get('fair_sampling', {})
        new_method = SamplingMethod(sampling_config.get('method', 'inverse_frequency'))
        new_temperature = sampling_config.get('temperature', 1.0)
        
        # Recreate sampler if method changed
        if new_method != old_method:
            self.method = new_method
            self.config = new_config
            self.sampler = self._create_sampler()
            logger.info(f"Sampling method changed from {old_method.value} to {new_method.value}")
        
        # Update temperature
        if hasattr(self.sampler, 'temperature'):
            self.sampler.temperature = new_temperature
    
    def save_state(self, filepath: str):
        """Save sampler state for checkpointing"""
        state = {
            'config': self.config,
            'method': self.method.value,
            'sampling_stats': dict(self.sampling_stats),
            'update_counter': self.update_counter
        }
        
        # Add sampler-specific state
        if hasattr(self.sampler, 'item_frequencies'):
            state['item_frequencies'] = dict(self.sampler.item_frequencies)
        
        if hasattr(self.sampler, 'genre_item_frequencies'):
            state['genre_item_frequencies'] = {
                k: dict(v) for k, v in self.sampler.genre_item_frequencies.items()
            }
        
        torch.save(state, filepath)
        logger.info(f"FairSampler state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load sampler state from checkpoint"""
        state = torch.load(filepath, map_location=self.device)
        
        self.sampling_stats = state.get('sampling_stats', {})
        self.update_counter = state.get('update_counter', 0)
        
        # Restore sampler-specific state
        if 'item_frequencies' in state and hasattr(self.sampler, 'item_frequencies'):
            self.sampler.item_frequencies = defaultdict(int, state['item_frequencies'])
        
        if 'genre_item_frequencies' in state and hasattr(self.sampler, 'genre_item_frequencies'):
            self.sampler.genre_item_frequencies = {
                int(k): defaultdict(int, v) for k, v in state['genre_item_frequencies'].items()
            }
        
        logger.info(f"FairSampler state loaded from {filepath}")


# Factory function for easy integration
def create_fair_sampler(config: Dict[str, Any], device: str = 'cuda') -> FairSampler:
    """
    Factory function to create FairSampler with configuration
    
    Args:
        config: Configuration dictionary from YAML
        device: Device to use for computation
        
    Returns:
        Configured FairSampler instance
    """
    return FairSampler(config, device)


# Usage example and integration helpers
class FairSamplingMixin:
    """Mixin class for easy integration with training components"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fair_sampler = None
    
    def setup_fair_sampling(self, config: Dict[str, Any], device: str = 'cuda'):
        """Setup fair sampling for this component"""
        if config.get('fair_sampling', {}).get('enabled', False):
            self.fair_sampler = create_fair_sampler(config, device)
    
    def sample_negatives(self, *args, **kwargs) -> torch.Tensor:
        """Sample negatives using fair sampler if available"""
        if self.fair_sampler:
            return self.fair_sampler.sample_negatives(*args, **kwargs)
        else:
            # Fallback to uniform sampling
            return self._uniform_sample_negatives(*args, **kwargs)
    
    def _uniform_sample_negatives(self, positive_items: torch.Tensor,
                                 all_items: torch.Tensor, 
                                 num_negatives: int) -> torch.Tensor:
        """Fallback uniform negative sampling"""
        positive_set = set(positive_items.cpu().numpy())
        candidates = torch.tensor([
            item for item in all_items if item.item() not in positive_set
        ])
        
        if len(candidates) <= num_negatives:
            return candidates
        
        indices = torch.randperm(len(candidates))[:num_negatives]
        return candidates[indices]

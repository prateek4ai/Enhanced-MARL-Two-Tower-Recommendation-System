"""
Enhanced MARL Two-Tower Recommendation System - Comprehensive Metrics Module

Implements evaluation metrics for multi-agent reinforcement learning recommendation system
supporting ContextGNN, BUHS, fairness optimization, and genre-specific performance tracking.

Key Features:
- Recommendation quality metrics (HR@K, NDCG@K, etc.)
- Fairness metrics (GINI coefficient, demographic parity)
- Long-tail discovery metrics (tail coverage, popularity bias)
- Agent-specific performance tracking
- Diversity and coverage metrics
- Training convergence monitoring
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class RecommendationQualityMetrics:
    """
    Core recommendation quality metrics for evaluating system performance.
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50]):
        self.k_values = k_values
        
    def hit_rate_at_k(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """
        Calculate Hit Rate @ K (HR@K)
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            targets: [batch_size, num_items] - Ground truth (binary)
            k: Top-K value
            
        Returns:
            Hit rate @ K
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        hits = 0
        for i in range(batch_size):
            if torch.any(targets[i, top_k_indices[i]] > 0):
                hits += 1
                
        return hits / batch_size
    
    def ndcg_at_k(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain @ K (NDCG@K)
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            targets: [batch_size, num_items] - Ground truth relevance scores
            k: Top-K value
            
        Returns:
            NDCG @ K
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        ndcg_scores = []
        for i in range(batch_size):
            # Get relevance scores for top-k predictions
            relevance_scores = targets[i, top_k_indices[i]]
            
            # Calculate DCG
            dcg = torch.sum(relevance_scores / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)))
            
            # Calculate IDCG (ideal DCG)
            ideal_relevance = torch.sort(targets[i], descending=True)[0][:k]
            idcg = torch.sum(ideal_relevance / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)))
            
            # NDCG
            if idcg > 0:
                ndcg_scores.append((dcg / idcg).item())
            else:
                ndcg_scores.append(0.0)
                
        return np.mean(ndcg_scores)
    
    def recall_at_k(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Calculate Recall @ K"""
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        recall_scores = []
        for i in range(batch_size):
            relevant_items = torch.sum(targets[i] > 0).item()
            if relevant_items == 0:
                continue
                
            retrieved_relevant = torch.sum(targets[i, top_k_indices[i]] > 0).item()
            recall_scores.append(retrieved_relevant / relevant_items)
            
        return np.mean(recall_scores) if recall_scores else 0.0
    
    def precision_at_k(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Calculate Precision @ K"""
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        precision_scores = []
        for i in range(batch_size):
            retrieved_relevant = torch.sum(targets[i, top_k_indices[i]] > 0).item()
            precision_scores.append(retrieved_relevant / k)
            
        return np.mean(precision_scores)
    
    def mean_average_precision(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Mean Average Precision (MAP)"""
        batch_size = predictions.size(0)
        ap_scores = []
        
        for i in range(batch_size):
            # Sort by prediction scores
            sorted_indices = torch.argsort(predictions[i], descending=True)
            sorted_targets = targets[i, sorted_indices]
            
            # Calculate AP
            relevant_positions = torch.where(sorted_targets > 0)[0]
            if len(relevant_positions) == 0:
                continue
                
            precisions = []
            for pos in relevant_positions:
                precision = torch.sum(sorted_targets[:pos+1] > 0).item() / (pos + 1)
                precisions.append(precision)
                
            ap_scores.append(np.mean(precisions))
            
        return np.mean(ap_scores) if ap_scores else 0.0

class FairnessMetrics:
    """
    Fairness metrics for evaluating system bias and distribution equality.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def gini_coefficient(self, exposures: torch.Tensor, epsilon: float = 1e-8) -> float:
        """
        Calculate GINI coefficient for exposure distribution.
        
        Args:
            exposures: Item exposure counts [num_items]
            epsilon: Small value for numerical stability
            
        Returns:
            GINI coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Add epsilon to avoid division by zero
        exposures = exposures + epsilon
        
        # Sort exposures in ascending order
        sorted_exposures, _ = torch.sort(exposures)
        n = len(sorted_exposures)
        
        # Create index tensor
        index = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        
        # GINI coefficient formula
        gini = (2 * torch.sum(index * sorted_exposures)) / (n * torch.sum(sorted_exposures)) - (n + 1) / n
        
        return torch.clamp(gini, 0.0, 1.0).item()
    
    def demographic_parity(self, predictions: torch.Tensor, 
                          user_demographics: torch.Tensor,
                          k: int = 10) -> Dict[str, float]:
        """
        Calculate demographic parity across user groups.
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            user_demographics: [batch_size, num_demo_features] - User demographic features
            k: Top-K for evaluation
            
        Returns:
            Dictionary of demographic parity metrics
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Group users by demographics (simplified for binary features)
        demo_groups = {}
        for demo_idx in range(user_demographics.size(1)):
            demo_feature = user_demographics[:, demo_idx]
            for group_val in torch.unique(demo_feature):
                group_mask = demo_feature == group_val
                group_name = f"demo_{demo_idx}_group_{group_val.item()}"
                demo_groups[group_name] = group_mask
        
        # Calculate recommendation rates per group
        parity_metrics = {}
        for group_name, group_mask in demo_groups.items():
            if torch.sum(group_mask) == 0:
                continue
                
            group_predictions = top_k_indices[group_mask]
            # Simplified metric: average number of recommendations per user in group
            avg_recs = group_predictions.size(0) * k / torch.sum(group_mask).item()
            parity_metrics[group_name] = avg_recs
            
        return parity_metrics
    
    def individual_fairness(self, predictions: torch.Tensor, 
                           user_similarities: torch.Tensor,
                           k: int = 10,
                           similarity_threshold: float = 0.8) -> float:
        """
        Calculate individual fairness - similar users should get similar recommendations.
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            user_similarities: [batch_size, batch_size] - User similarity matrix
            k: Top-K for evaluation
            similarity_threshold: Threshold for considering users as similar
            
        Returns:
            Individual fairness score
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        fairness_scores = []
        
        for i in range(batch_size):
            # Find similar users
            similar_users = torch.where(user_similarities[i] > similarity_threshold)[0]
            similar_users = similar_users[similar_users != i]  # Exclude self
            
            if len(similar_users) == 0:
                continue
                
            # Calculate recommendation overlap with similar users
            user_recs = set(top_k_indices[i].cpu().numpy())
            
            overlaps = []
            for similar_user in similar_users:
                similar_recs = set(top_k_indices[similar_user].cpu().numpy())
                overlap = len(user_recs.intersection(similar_recs)) / k
                overlaps.append(overlap)
                
            fairness_scores.append(np.mean(overlaps))
            
        return np.mean(fairness_scores) if fairness_scores else 0.0

class DiversityMetrics:
    """
    Diversity and coverage metrics for recommendation quality assessment.
    """
    
    def __init__(self, item_features: Optional[torch.Tensor] = None):
        self.item_features = item_features
        
    def intra_list_diversity(self, predictions: torch.Tensor, 
                           item_features: torch.Tensor,
                           k: int = 10) -> float:
        """
        Calculate intra-list diversity (average pairwise distance within recommendation lists).
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            item_features: [num_items, feature_dim] - Item feature vectors
            k: Top-K for evaluation
            
        Returns:
            Average intra-list diversity
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        diversity_scores = []
        
        for i in range(batch_size):
            # Get features for recommended items
            rec_features = item_features[top_k_indices[i]]  # [k, feature_dim]
            
            # Calculate pairwise cosine distances
            normalized_features = F.normalize(rec_features, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # Convert similarities to distances
            distance_matrix = 1 - similarity_matrix
            
            # Average pairwise distance (excluding diagonal)
            mask = torch.ones_like(distance_matrix) - torch.eye(k)
            avg_distance = torch.sum(distance_matrix * mask) / torch.sum(mask)
            diversity_scores.append(avg_distance.item())
            
        return np.mean(diversity_scores)
    
    def catalog_coverage(self, predictions: torch.Tensor, 
                        num_items: int,
                        k: int = 10) -> float:
        """
        Calculate catalog coverage - percentage of items recommended at least once.
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            num_items: Total number of items in catalog
            k: Top-K for evaluation
            
        Returns:
            Catalog coverage ratio
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        unique_items = torch.unique(top_k_indices.flatten())
        
        return len(unique_items) / num_items
    
    def genre_diversity(self, predictions: torch.Tensor,
                       item_genres: torch.Tensor,
                       k: int = 10) -> float:
        """
        Calculate genre diversity in recommendations.
        
        Args:
            predictions: [batch_size, num_items] - Predicted scores
            item_genres: [num_items, num_genres] - Item genre assignments (binary)
            k: Top-K for evaluation
            
        Returns:
            Average genre diversity (entropy-based)
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        diversity_scores = []
        
        for i in range(batch_size):
            # Get genres for recommended items
            rec_genres = item_genres[top_k_indices[i]]  # [k, num_genres]
            
            # Calculate genre distribution
            genre_counts = torch.sum(rec_genres, dim=0)
            genre_probs = genre_counts.float() / torch.sum(genre_counts).float()
            
            # Remove zero probabilities for entropy calculation
            genre_probs = genre_probs[genre_probs > 0]
            
            if len(genre_probs) > 1:
                # Calculate entropy
                entropy_score = -torch.sum(genre_probs * torch.log(genre_probs))
                diversity_scores.append(entropy_score.item())
            else:
                diversity_scores.append(0.0)
                
        return np.mean(diversity_scores)

class LongTailMetrics:
    """
    Metrics specifically for evaluating long-tail item recommendation performance.
    """
    
    def __init__(self, item_popularity: torch.Tensor, tail_threshold: float = 0.2):
        """
        Args:
            item_popularity: [num_items] - Item popularity scores (normalized)
            tail_threshold: Threshold for classifying tail items (bottom percentile)
        """
        self.item_popularity = item_popularity
        self.tail_threshold = tail_threshold
        
        # Classify items as head/tail
        popularity_percentile = torch.quantile(item_popularity, tail_threshold)
        self.tail_items = item_popularity <= popularity_percentile
        self.head_items = ~self.tail_items
        
    def tail_hit_rate_at_k(self, predictions: torch.Tensor, 
                          targets: torch.Tensor, 
                          k: int) -> float:
        """Calculate hit rate specifically for tail items."""
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        hits = 0
        total_users_with_tail_items = 0
        
        for i in range(batch_size):
            # Check if user has any tail items in ground truth
            user_tail_items = targets[i] * self.tail_items.float()
            if torch.sum(user_tail_items) == 0:
                continue
                
            total_users_with_tail_items += 1
            
            # Check if any recommended items are tail items the user liked
            recommended_tail_relevance = user_tail_items[top_k_indices[i]]
            if torch.any(recommended_tail_relevance > 0):
                hits += 1
                
        return hits / total_users_with_tail_items if total_users_with_tail_items > 0 else 0.0
    
    def head_vs_tail_performance(self, predictions: torch.Tensor, 
                                targets: torch.Tensor,
                                k: int = 10) -> Dict[str, float]:
        """Compare performance between head and tail items."""
        # Separate predictions and targets for head/tail items
        head_mask = self.head_items.unsqueeze(0).expand_as(predictions)
        tail_mask = self.tail_items.unsqueeze(0).expand_as(predictions)
        
        head_predictions = predictions * head_mask.float()
        tail_predictions = predictions * tail_mask.float()
        
        head_targets = targets * head_mask.float()
        tail_targets = targets * tail_mask.float()
        
        # Calculate metrics
        quality_metrics = RecommendationQualityMetrics()
        
        results = {
            'head_hr@k': quality_metrics.hit_rate_at_k(head_predictions, head_targets, k),
            'tail_hr@k': quality_metrics.hit_rate_at_k(tail_predictions, tail_targets, k),
            'head_ndcg@k': quality_metrics.ndcg_at_k(head_predictions, head_targets, k),
            'tail_ndcg@k': quality_metrics.ndcg_at_k(tail_predictions, tail_targets, k)
        }
        
        return results
    
    def popularity_bias(self, predictions: torch.Tensor, k: int = 10) -> float:
        """
        Calculate popularity bias in recommendations.
        
        Returns:
            Average popularity of recommended items (higher = more biased toward popular items)
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        recommended_popularities = self.item_popularity[top_k_indices]
        
        return torch.mean(recommended_popularities).item()
    
    def tail_coverage_ratio(self, predictions: torch.Tensor, k: int = 10) -> float:
        """
        Calculate the ratio of tail items in recommendations.
        
        Returns:
            Percentage of recommendations that are tail items
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        recommended_tail_mask = self.tail_items[top_k_indices]
        
        return torch.mean(recommended_tail_mask.float()).item()

class AgentSpecificMetrics:
    """
    Metrics for evaluating individual agent and multi-agent coordination performance.
    """
    
    def __init__(self, num_agents: int, agent_names: List[str]):
        self.num_agents = num_agents
        self.agent_names = agent_names
        
    def per_agent_performance(self, agent_predictions: Dict[str, torch.Tensor],
                             agent_targets: Dict[str, torch.Tensor],
                             k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each individual agent.
        
        Args:
            agent_predictions: Dict[agent_name -> predictions tensor]
            agent_targets: Dict[agent_name -> targets tensor]
            k: Top-K for evaluation
            
        Returns:
            Nested dict of agent performance metrics
        """
        quality_metrics = RecommendationQualityMetrics()
        results = {}
        
        for agent_name in self.agent_names:
            if agent_name not in agent_predictions or agent_name not in agent_targets:
                continue
                
            pred = agent_predictions[agent_name]
            target = agent_targets[agent_name]
            
            results[agent_name] = {
                f'hr@{k}': quality_metrics.hit_rate_at_k(pred, target, k),
                f'ndcg@{k}': quality_metrics.ndcg_at_k(pred, target, k),
                f'recall@{k}': quality_metrics.recall_at_k(pred, target, k),
                f'precision@{k}': quality_metrics.precision_at_k(pred, target, k)
            }
            
        return results
    
    def agent_coordination_effectiveness(self, 
                                       individual_predictions: Dict[str, torch.Tensor],
                                       coordinated_predictions: torch.Tensor,
                                       targets: torch.Tensor,
                                       k: int = 10) -> float:
        """
        Measure how much coordination improves over individual agent performance.
        
        Returns:
            Coordination improvement ratio (>1 means coordination helps)
        """
        quality_metrics = RecommendationQualityMetrics()
        
        # Calculate coordinated performance
        coordinated_hr = quality_metrics.hit_rate_at_k(coordinated_predictions, targets, k)
        
        # Calculate average individual performance
        individual_hrs = []
        for agent_name, pred in individual_predictions.items():
            hr = quality_metrics.hit_rate_at_k(pred, targets, k)
            individual_hrs.append(hr)
            
        avg_individual_hr = np.mean(individual_hrs) if individual_hrs else 0.0
        
        if avg_individual_hr > 0:
            return coordinated_hr / avg_individual_hr
        else:
            return 1.0
    
    def agent_specialization_score(self, 
                                  agent_predictions: Dict[str, torch.Tensor],
                                  item_genres: torch.Tensor,
                                  genre_mapping: Dict[str, int],
                                  k: int = 10) -> Dict[str, float]:
        """
        Measure how well each agent specializes in its assigned genre/category.
        
        Args:
            agent_predictions: Dict[agent_name -> predictions tensor]
            item_genres: [num_items, num_genres] - Item genre assignments
            genre_mapping: Dict[agent_name -> genre_index]
            k: Top-K for evaluation
            
        Returns:
            Specialization scores for each agent
        """
        specialization_scores = {}
        
        for agent_name, predictions in agent_predictions.items():
            if agent_name not in genre_mapping:
                continue
                
            genre_idx = genre_mapping[agent_name]
            _, top_k_indices = torch.topk(predictions, k, dim=1)
            
            # Calculate percentage of recommendations in agent's specialized genre
            recommended_genres = item_genres[top_k_indices]  # [batch_size, k, num_genres]
            specialized_genre_recs = recommended_genres[:, :, genre_idx]  # [batch_size, k]
            
            specialization_ratio = torch.mean(specialized_genre_recs.float()).item()
            specialization_scores[agent_name] = specialization_ratio
            
        return specialization_scores

class TrainingMetrics:
    """
    Metrics for monitoring training progress and convergence.
    """
    
    def __init__(self):
        self.loss_history = defaultdict(list)
        self.gradient_norms = defaultdict(list)
        self.policy_stats = defaultdict(list)
        
    def update_loss_history(self, loss_dict: Dict[str, float], step: int):
        """Update loss history with current step losses."""
        for loss_name, loss_value in loss_dict.items():
            self.loss_history[loss_name].append((step, loss_value))
            
    def update_gradient_norms(self, grad_norms: Dict[str, float], step: int):
        """Update gradient norm history."""
        for component, norm in grad_norms.items():
            self.gradient_norms[component].append((step, norm))
            
    def update_policy_stats(self, policy_stats: Dict[str, Dict[str, float]], step: int):
        """Update policy statistics (KL divergence, advantage variance, etc.)."""
        for agent_name, stats in policy_stats.items():
            for stat_name, stat_value in stats.items():
                key = f"{agent_name}_{stat_name}"
                self.policy_stats[key].append((step, stat_value))
                
    def convergence_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """
        Calculate convergence metrics based on recent training history.
        
        Args:
            window_size: Number of recent steps to consider
            
        Returns:
            Convergence metrics dictionary
        """
        metrics = {}
        
        # Loss stability (coefficient of variation)
        for loss_name, history in self.loss_history.items():
            if len(history) < window_size:
                continue
                
            recent_losses = [loss for _, loss in history[-window_size:]]
            if len(recent_losses) > 1:
                cv = np.std(recent_losses) / (np.mean(recent_losses) + 1e-8)
                metrics[f"{loss_name}_stability"] = 1.0 / (1.0 + cv)  # Higher is more stable
                
        # Gradient norm trends
        for component, history in self.gradient_norms.items():
            if len(history) < window_size:
                continue
                
            recent_norms = [norm for _, norm in history[-window_size:]]
            if len(recent_norms) > 1:
                # Check if gradients are exploding or vanishing
                max_norm = max(recent_norms)
                min_norm = min(recent_norms)
                avg_norm = np.mean(recent_norms)
                
                metrics[f"{component}_grad_max"] = max_norm
                metrics[f"{component}_grad_avg"] = avg_norm
                metrics[f"{component}_grad_stability"] = min_norm / (max_norm + 1e-8)
                
        return metrics

class MetricsCalculator:
    """
    Main metrics calculator that orchestrates all metric computations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize metric components
        self.quality_metrics = RecommendationQualityMetrics(
            k_values=config.get('evaluation', {}).get('k_values', [1, 5, 10, 20])
        )
        self.fairness_metrics = FairnessMetrics(device=self.device)
        self.diversity_metrics = DiversityMetrics()
        
        # Initialize long-tail metrics if popularity data available
        self.longtail_metrics = None
        if 'item_popularity' in config:
            self.longtail_metrics = LongTailMetrics(
                item_popularity=torch.tensor(config['item_popularity']),
                tail_threshold=config.get('tail_threshold', 0.2)
            )
            
        # Initialize agent-specific metrics
        if 'dataset' in config and 'genres' in config['dataset']:
            self.agent_metrics = AgentSpecificMetrics(
                num_agents=len(config['dataset']['genres']),
                agent_names=config['dataset']['genres']
            )
        else:
            self.agent_metrics = None
            
        # Initialize training metrics
        self.training_metrics = TrainingMetrics()
        
    def compute_comprehensive_metrics(self, 
                                    predictions: torch.Tensor,
                                    targets: torch.Tensor,
                                    agent_predictions: Optional[Dict[str, torch.Tensor]] = None,
                                    exposure_data: Optional[Dict[str, torch.Tensor]] = None,
                                    user_demographics: Optional[torch.Tensor] = None,
                                    item_features: Optional[torch.Tensor] = None,
                                    item_genres: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: [batch_size, num_items] - Main system predictions
            targets: [batch_size, num_items] - Ground truth
            agent_predictions: Optional dict of per-agent predictions
            exposure_data: Optional dict containing exposure information
            user_demographics: Optional user demographic features
            item_features: Optional item feature vectors
            item_genres: Optional item genre assignments
            
        Returns:
            Comprehensive metrics dictionary
        """
        results = {}
        
        # Recommendation quality metrics
        for k in self.quality_metrics.k_values:
            results[f'hr@{k}'] = self.quality_metrics.hit_rate_at_k(predictions, targets, k)
            results[f'ndcg@{k}'] = self.quality_metrics.ndcg_at_k(predictions, targets, k)
            results[f'recall@{k}'] = self.quality_metrics.recall_at_k(predictions, targets, k)
            results[f'precision@{k}'] = self.quality_metrics.precision_at_k(predictions, targets, k)
            
        results['map'] = self.quality_metrics.mean_average_precision(predictions, targets)
        
        # Fairness metrics
        if exposure_data and 'item_exposures' in exposure_data:
            results['gini_coefficient'] = self.fairness_metrics.gini_coefficient(
                exposure_data['item_exposures']
            )
            
        if user_demographics is not None:
            demographic_parity = self.fairness_metrics.demographic_parity(
                predictions, user_demographics
            )
            results.update({f'demo_parity_{k}': v for k, v in demographic_parity.items()})
            
        # Diversity metrics
        if item_features is not None:
            results['intra_list_diversity'] = self.diversity_metrics.intra_list_diversity(
                predictions, item_features
            )
            
        results['catalog_coverage'] = self.diversity_metrics.catalog_coverage(
            predictions, predictions.size(1)
        )
        
        if item_genres is not None:
            results['genre_diversity'] = self.diversity_metrics.genre_diversity(
                predictions, item_genres
            )
            
        # Long-tail metrics
        if self.longtail_metrics is not None:
            results['tail_hr@10'] = self.longtail_metrics.tail_hit_rate_at_k(predictions, targets, 10)
            results['popularity_bias'] = self.longtail_metrics.popularity_bias(predictions)
            results['tail_coverage_ratio'] = self.longtail_metrics.tail_coverage_ratio(predictions)
            
            head_tail_perf = self.longtail_metrics.head_vs_tail_performance(predictions, targets)
            results.update(head_tail_perf)
            
        # Agent-specific metrics
        if self.agent_metrics is not None and agent_predictions is not None:
            agent_targets = {name: targets for name in agent_predictions.keys()}  # Simplified
            per_agent_perf = self.agent_metrics.per_agent_performance(
                agent_predictions, agent_targets
            )
            
            for agent_name, metrics in per_agent_perf.items():
                for metric_name, value in metrics.items():
                    results[f'{agent_name}_{metric_name}'] = value
                    
            # Coordination effectiveness
            results['coordination_effectiveness'] = self.agent_metrics.agent_coordination_effectiveness(
                agent_predictions, predictions, targets
            )
            
        return results
    
    def update_training_metrics(self, 
                               loss_dict: Dict[str, float],
                               grad_norms: Dict[str, float],
                               policy_stats: Dict[str, Dict[str, float]],
                               step: int):
        """Update training metrics."""
        self.training_metrics.update_loss_history(loss_dict, step)
        self.training_metrics.update_gradient_norms(grad_norms, step)
        self.training_metrics.update_policy_stats(policy_stats, step)
        
    def get_convergence_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """Get convergence metrics."""
        return self.training_metrics.convergence_metrics(window_size)
    
    def get_fairness_summary(self, exposure_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get summary of fairness metrics."""
        summary = {}
        
        if 'item_exposures' in exposure_data:
            summary['gini_coefficient'] = self.fairness_metrics.gini_coefficient(
                exposure_data['item_exposures']
            )
            
        if 'genre_exposures' in exposure_data:
            summary['genre_gini'] = self.fairness_metrics.gini_coefficient(
                exposure_data['genre_exposures']
            )
            
        return summary
    
    def export_metrics_for_monitoring(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Export key metrics in format suitable for monitoring systems (Prometheus, etc.).
        
        Returns:
            Flattened metrics dictionary with only numeric values
        """
        monitoring_metrics = {}
        
        # Key performance indicators
        key_metrics = [
            'hr@10', 'ndcg@10', 'gini_coefficient', 'catalog_coverage',
            'tail_hr@10', 'popularity_bias', 'coordination_effectiveness'
        ]
        
        for metric in key_metrics:
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                monitoring_metrics[metric] = float(metrics[metric])
                
        # Agent-specific key metrics
        for key, value in metrics.items():
            if ('_hr@10' in key or '_ndcg@10' in key) and isinstance(value, (int, float)):
                monitoring_metrics[key] = float(value)
                
        return monitoring_metrics

# Factory function for easy integration
def create_metrics_calculator(config: Dict[str, Any]) -> MetricsCalculator:
    """
    Factory function to create MetricsCalculator with configuration.
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        Configured MetricsCalculator instance
    """
    return MetricsCalculator(config)

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'device': 'cuda',
        'evaluation': {
            'k_values': [1, 5, 10, 20]
        },
        'dataset': {
            'genres': ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
        },
        'item_popularity': np.random.rand(1000),  # Example popularity scores
        'tail_threshold': 0.2
    }
    
    # Initialize metrics calculator
    metrics_calc = create_metrics_calculator(config)
    
    # Example usage
    batch_size, num_items = 32, 1000
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, 2, (batch_size, num_items)).float()
    
    # Compute metrics
    results = metrics_calc.compute_comprehensive_metrics(predictions, targets)
    
    print("Example metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
    logger.info("Metrics calculator testing completed successfully")

"""
Loss Utilities for Enhanced MARL Two-Tower Recommendation System

Comprehensive loss function implementations supporting:
- BPR loss for recommendation quality
- Multi-Agent PPO losses for RL optimization
- Contrastive learning for long-tail discovery
- Stable rank regularization for embedding capacity
- Fairness losses (GINI, exposure distribution)
- Multi-objective loss combination and weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseLoss(ABC):
    """Abstract base class for loss functions"""
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        self.weight = weight
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute loss"""
        pass
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

class BPRLoss(BaseLoss):
    """
    Bayesian Personalized Ranking Loss for recommendation quality
    
    Optimizes ranking between positive and negative items
    """
    
    def __init__(self, weight: float = 1.0, margin: float = 1.0, 
                 reduction: str = 'mean'):
        super().__init__(weight, reduction)
        self.margin = margin
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_scores: [batch_size] - Scores for positive items
            neg_scores: [batch_size, num_negatives] - Scores for negative items
            
        Returns:
            BPR loss value
        """
        # Expand positive scores to match negative scores shape
        pos_expanded = pos_scores.unsqueeze(-1)  # [batch_size, 1]
        
        # Compute margin-based ranking loss
        diff = self.margin - (pos_expanded - neg_scores)  # [batch_size, num_negatives]
        loss = F.relu(diff)  # Hinge loss
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return self.weight * loss

class PPOLoss(BaseLoss):
    """
    Proximal Policy Optimization Loss for multi-agent RL
    
    Supports both policy and value function losses with clipping
    """
    
    def __init__(self, weight: float = 1.0, clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        super().__init__(weight)
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def forward(self, policy_logits: torch.Tensor, old_log_probs: torch.Tensor,
                actions: torch.Tensor, advantages: torch.Tensor,
                values: torch.Tensor, returns: torch.Tensor,
                value_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            policy_logits: [batch_size, action_dim] - Current policy logits
            old_log_probs: [batch_size] - Old policy log probabilities
            actions: [batch_size] - Selected actions
            advantages: [batch_size] - GAE advantages
            values: [batch_size] - Current value estimates
            returns: [batch_size] - Discounted returns
            value_targets: [batch_size] - Optional value targets
            
        Returns:
            Dict containing policy_loss, value_loss, entropy_loss, total_loss
        """
        # Policy loss with clipping
        action_probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # Current log probabilities for selected actions
        current_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value function loss
        if value_targets is not None:
            value_loss = F.mse_loss(values, value_targets)
        else:
            value_loss = F.mse_loss(values, returns)
        
        # Entropy loss for exploration
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        total_loss = self.weight * total_loss
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'ratio_mean': ratio.mean(),
            'advantages_mean': advantages.mean()
        }

class ContrastiveLoss(BaseLoss):
    """
    InfoNCE Contrastive Loss for long-tail item discovery
    
    Encourages distinct representations between head and tail items
    """
    
    def __init__(self, weight: float = 1.0, temperature: float = 0.1):
        super().__init__(weight)
        self.temperature = temperature
    
    def forward(self, anchor_embeddings: torch.Tensor, 
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, embed_dim] - Anchor item embeddings
            positive_embeddings: [batch_size, embed_dim] - Positive pair embeddings
            negative_embeddings: [batch_size, num_negatives, embed_dim] - Negative pairs
            
        Returns:
            InfoNCE contrastive loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor_embeddings, dim=-1)
        positive = F.normalize(positive_embeddings, dim=-1)
        negative = F.normalize(negative_embeddings, dim=-1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch_size]
        
        # Negative similarities
        neg_sim = torch.bmm(
            anchor.unsqueeze(1), 
            negative.transpose(-2, -1)
        ).squeeze(1) / self.temperature  # [batch_size, num_negatives]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        return self.weight * loss

class StableRankLoss(BaseLoss):
    """
    Stable Rank Regularization Loss
    
    Encourages balanced embedding capacity across users/items
    """
    
    def __init__(self, weight: float = 1.0, eps: float = 1e-8):
        super().__init__(weight)
        self.eps = eps
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embed_dim] - Embedding matrix
            
        Returns:
            Stable rank penalty (negative for minimization)
        """
        try:
            # SVD decomposition
            U, S, V = torch.svd(embeddings)
            
            # Stable rank = ||S||_F^2 / ||S||_2^2
            frobenius_norm_sq = torch.sum(S ** 2)
            spectral_norm_sq = torch.max(S) ** 2
            
            stable_rank = frobenius_norm_sq / (spectral_norm_sq + self.eps)
            
            # Return negative for minimization (we want to maximize stable rank)
            penalty = -stable_rank
            
        except RuntimeError:
            # Fallback if SVD fails
            logger.warning("SVD failed in stable rank computation, returning zero loss")
            penalty = torch.tensor(0.0, device=embeddings.device)
        
        return self.weight * penalty

class GINIPenalty(BaseLoss):
    """
    GINI Coefficient Penalty for fairness optimization
    
    Penalizes unequal exposure distribution across items/genres
    """
    
    def __init__(self, weight: float = 1.0, target_gini: float = 0.5, 
                 eps: float = 1e-8):
        super().__init__(weight)
        self.target_gini = target_gini
        self.eps = eps
    
    def compute_gini_coefficient(self, exposures: torch.Tensor) -> torch.Tensor:
        """
        Compute GINI coefficient for exposure distribution
        
        Args:
            exposures: [num_items] - Item exposure counts
            
        Returns:
            GINI coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Add small epsilon to avoid division by zero
        exposures = exposures + self.eps
        
        # Sort exposures in ascending order
        sorted_exposures, _ = torch.sort(exposures)
        n = len(sorted_exposures)
        
        # Create index tensor
        index = torch.arange(1, n + 1, device=exposures.device, dtype=torch.float32)
        
        # GINI coefficient formula
        gini = (2 * torch.sum(index * sorted_exposures)) / (n * torch.sum(sorted_exposures)) - (n + 1) / n
        
        return torch.clamp(gini, 0.0, 1.0)
    
    def forward(self, exposures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            exposures: [num_items] - Item exposure counts
            
        Returns:
            GINI penalty loss
        """
        current_gini = self.compute_gini_coefficient(exposures)
        
        # Penalty for deviating from target GINI
        penalty = F.mse_loss(current_gini, 
                           torch.tensor(self.target_gini, device=exposures.device))
        
        return self.weight * penalty

class ExposurePenalty(BaseLoss):
    """
    Item Exposure Distribution Penalty
    
    Encourages balanced exposure across different item categories
    """
    
    def __init__(self, weight: float = 1.0, smoothing: float = 0.1):
        super().__init__(weight)
        self.smoothing = smoothing
    
    def forward(self, predicted_exposures: torch.Tensor, 
                target_exposures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_exposures: [num_categories] - Predicted exposure distribution
            target_exposures: [num_categories] - Target exposure distribution
            
        Returns:
            Exposure distribution penalty
        """
        # Smooth the distributions
        pred_smooth = predicted_exposures + self.smoothing
        target_smooth = target_exposures + self.smoothing
        
        # Normalize to probabilities
        pred_prob = pred_smooth / pred_smooth.sum()
        target_prob = target_smooth / target_smooth.sum()
        
        # KL divergence penalty
        kl_div = F.kl_div(
            pred_prob.log(), 
            target_prob, 
            reduction='batchmean'
        )
        
        return self.weight * kl_div

class BUHSLoss(BaseLoss):
    """
    Biased User History Synthesis Loss
    
    Supports BUHS module training with contrastive objectives
    """
    
    def __init__(self, weight: float = 1.0, alpha: float = 1.0, 
                 temperature: float = 0.1):
        super().__init__(weight)
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(self, synthesized_features: torch.Tensor,
                original_features: torch.Tensor,
                item_popularities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            synthesized_features: [batch_size, feature_dim] - BUHS synthesized features
            original_features: [batch_size, feature_dim] - Original user features
            item_popularities: [batch_size] - Item popularity scores
            
        Returns:
            BUHS training loss
        """
        # Inverse popularity weighting for synthesis quality
        weights = torch.pow(1.0 / (item_popularities + 1e-8), self.alpha)
        
        # Weighted MSE between synthesized and original features
        mse_loss = F.mse_loss(synthesized_features, original_features, reduction='none')
        weighted_loss = (mse_loss.mean(dim=-1) * weights).mean()
        
        return self.weight * weighted_loss

class MultiObjectiveLoss:
    """
    Multi-Objective Loss Combiner for MARL Two-Tower System
    
    Handles complex loss weighting and combination strategies
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize individual loss functions
        self.losses = self._initialize_losses()
        
        # Loss weights from config
        self.loss_weights = config.get('loss_weights', {})
        
        # Adaptive weighting
        self.adaptive_weighting = config.get('adaptive_loss_weighting', False)
        self.weight_history = defaultdict(list)
        
        # Loss tracking
        self.loss_history = defaultdict(list)
        
    def _initialize_losses(self) -> Dict[str, BaseLoss]:
        """Initialize all loss functions"""
        losses = {}
        
        # BPR Loss
        losses['bpr'] = BPRLoss(
            weight=self.loss_weights.get('bpr_loss', 1.0),
            margin=self.config.get('bpr_margin', 1.0)
        )
        
        # PPO Loss
        losses['ppo'] = PPOLoss(
            weight=self.loss_weights.get('ppo_loss', 0.5),
            clip_epsilon=self.config['ppo']['clip_epsilon'],
            value_loss_coef=self.config['ppo']['value_loss_coef'],
            entropy_coef=self.config['ppo']['entropy_coef']
        )
        
        # Contrastive Loss
        losses['contrastive'] = ContrastiveLoss(
            weight=self.loss_weights.get('contrastive_loss', 0.3),
            temperature=self.config.get('contrastive_temperature', 0.1)
        )
        
        # Stable Rank Loss
        losses['stable_rank'] = StableRankLoss(
            weight=self.loss_weights.get('stable_rank_loss', 0.2)
        )
        
        # GINI Penalty
        losses['gini_penalty'] = GINIPenalty(
            weight=self.loss_weights.get('gini_penalty', 0.4),
            target_gini=self.config.get('target_gini', 0.5)
        )
        
        # Exposure Penalty
        losses['exposure_penalty'] = ExposurePenalty(
            weight=self.loss_weights.get('exposure_penalty', 0.1)
        )
        
        # BUHS Loss
        if self.config.get('buhs', {}).get('enabled', False):
            losses['buhs'] = BUHSLoss(
                weight=self.loss_weights.get('buhs_loss', 0.2),
                alpha=self.config['buhs']['alpha']
            )
        
        return losses
    
    def compute_bpr_loss(self, user_embeddings: torch.Tensor, 
                        pos_item_embeddings: torch.Tensor,
                        neg_item_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute BPR loss for recommendation quality"""
        # Compute scores
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=-1)
        neg_scores = torch.bmm(
            user_embeddings.unsqueeze(1),
            neg_item_embeddings.transpose(-2, -1)
        ).squeeze(1)
        
        return self.losses['bpr'](pos_scores, neg_scores)
    
    def compute_ppo_loss(self, agent_outputs: Dict[str, torch.Tensor],
                        old_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for all agents"""
        ppo_losses = {}
        
        for agent_name in agent_outputs:
            if agent_name in old_outputs:
                agent_loss = self.losses['ppo'](
                    policy_logits=agent_outputs[agent_name]['policy_logits'],
                    old_log_probs=old_outputs[agent_name]['log_probs'],
                    actions=agent_outputs[agent_name]['actions'],
                    advantages=agent_outputs[agent_name]['advantages'],
                    values=agent_outputs[agent_name]['values'],
                    returns=agent_outputs[agent_name]['returns']
                )
                ppo_losses[f'{agent_name}_ppo'] = agent_loss
        
        return ppo_losses
    
    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor],
                                popularity_mask: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for head vs tail items"""
        if 'anchor' not in embeddings or 'positive' not in embeddings or 'negative' not in embeddings:
            return torch.tensor(0.0, device=self.device)
        
        return self.losses['contrastive'](
            embeddings['anchor'],
            embeddings['positive'], 
            embeddings['negative']
        )
    
    def compute_stable_rank_loss(self, embedding_matrices: List[torch.Tensor]) -> torch.Tensor:
        """Compute stable rank loss for embedding matrices"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for embeddings in embedding_matrices:
            if embeddings.numel() > 0:
                total_loss += self.losses['stable_rank'](embeddings)
        
        return total_loss / max(len(embedding_matrices), 1)
    
    def compute_fairness_losses(self, exposure_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute fairness-related losses"""
        fairness_losses = {}
        
        # GINI penalty
        if 'item_exposures' in exposure_data:
            fairness_losses['gini_penalty'] = self.losses['gini_penalty'](
                exposure_data['item_exposures']
            )
        
        # Exposure penalty
        if 'predicted_exposures' in exposure_data and 'target_exposures' in exposure_data:
            fairness_losses['exposure_penalty'] = self.losses['exposure_penalty'](
                exposure_data['predicted_exposures'],
                exposure_data['target_exposures']
            )
        
        return fairness_losses
    
    def compute_buhs_loss(self, buhs_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute BUHS training loss"""
        if 'buhs' not in self.losses:
            return torch.tensor(0.0, device=self.device)
        
        return self.losses['buhs'](
            buhs_outputs['synthesized_features'],
            buhs_outputs['original_features'],
            buhs_outputs['item_popularities']
        )
    
    def compute_total_loss(self, loss_components: Dict[str, torch.Tensor],
                          step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total weighted loss from individual components
        
        Args:
            loss_components: Dict of individual loss values
            step: Current training step for adaptive weighting
            
        Returns:
            Dict containing total loss and individual weighted components
        """
        weighted_losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Apply weights and sum losses
        for loss_name, loss_value in loss_components.items():
            if isinstance(loss_value, dict):
                # Handle nested loss dictionaries (e.g., PPO losses)
                for sub_name, sub_value in loss_value.items():
                    weighted_loss = sub_value
                    weighted_losses[sub_name] = weighted_loss
                    total_loss += weighted_loss
            else:
                weighted_loss = loss_value
                weighted_losses[loss_name] = weighted_loss
                total_loss += weighted_loss
        
        # Update loss history
        if step is not None:
            for name, value in weighted_losses.items():
                if isinstance(value, torch.Tensor):
                    self.loss_history[name].append(value.item())
        
        # Adaptive weight adjustment
        if self.adaptive_weighting and step is not None and step % 100 == 0:
            self._adjust_weights()
        
        result = {
            'total_loss': total_loss,
            **weighted_losses
        }
        
        return result
    
    def _adjust_weights(self):
        """Adjust loss weights based on recent performance"""
        # Simple adaptive weighting based on loss magnitude
        for loss_name in self.loss_history:
            if len(self.loss_history[loss_name]) >= 10:
                recent_losses = self.loss_history[loss_name][-10:]
                avg_loss = np.mean(recent_losses)
                
                # Adjust weight inversely to loss magnitude
                if loss_name in self.loss_weights:
                    current_weight = self.loss_weights[loss_name]
                    adjustment = 0.1 if avg_loss < 0.1 else -0.05
                    new_weight = max(0.01, min(2.0, current_weight + adjustment))
                    self.loss_weights[loss_name] = new_weight
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get loss statistics for monitoring"""
        stats = {}
        
        for loss_name, history in self.loss_history.items():
            if history:
                stats[f'{loss_name}_mean'] = np.mean(history[-100:])  # Last 100 steps
                stats[f'{loss_name}_std'] = np.std(history[-100:])
                stats[f'{loss_name}_current'] = history[-1]
        
        return stats
    
    def reset_history(self):
        """Reset loss history for new training phase"""
        self.loss_history.clear()
        self.weight_history.clear()

class LossManager:
    """
    High-level Loss Manager for Enhanced MARL Two-Tower System
    
    Orchestrates all loss computations and provides unified interface
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize multi-objective loss handler
        self.multi_loss = MultiObjectiveLoss(config, device)
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Loss scaling for mixed precision
        self.loss_scale = config.get('loss_scale', 1.0)
        
        # Training phase tracking
        self.training_phase = 'pretrain'  # 'pretrain', 'marl', 'finetune'
        
    def compute_recommendation_loss(self, batch_data: Dict[str, torch.Tensor],
                                  model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute recommendation quality losses"""
        losses = {}
        
        # BPR loss
        if all(key in model_outputs for key in ['user_embeddings', 'pos_item_embeddings', 'neg_item_embeddings']):
            losses['bpr'] = self.multi_loss.compute_bpr_loss(
                model_outputs['user_embeddings'],
                model_outputs['pos_item_embeddings'],
                model_outputs['neg_item_embeddings']
            )
        
        # Contrastive loss for long-tail items
        if 'contrastive_embeddings' in model_outputs:
            losses['contrastive'] = self.multi_loss.compute_contrastive_loss(
                model_outputs['contrastive_embeddings'],
                batch_data.get('popularity_mask', torch.ones(1, device=self.device))
            )
        
        return losses
    
    def compute_rl_losses(self, agent_outputs: Dict[str, torch.Tensor],
                         old_agent_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute reinforcement learning losses for all agents"""
        return self.multi_loss.compute_ppo_loss(agent_outputs, old_agent_outputs)
    
    def compute_fairness_losses(self, exposure_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute fairness-related losses"""
        return self.multi_loss.compute_fairness_losses(exposure_data)
    
    def compute_regularization_losses(self, model_components: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute regularization losses"""
        losses = {}
        
        # Stable rank regularization
        embedding_matrices = []
        for component_name, component in model_components.items():
            if hasattr(component, 'weight') and component.weight.dim() == 2:
                embedding_matrices.append(component.weight)
        
        if embedding_matrices:
            losses['stable_rank'] = self.multi_loss.compute_stable_rank_loss(embedding_matrices)
        
        # BUHS loss if enabled
        if 'buhs_outputs' in model_components:
            losses['buhs'] = self.multi_loss.compute_buhs_loss(model_components['buhs_outputs'])
        
        return losses
    
    def compute_total_loss(self, batch_data: Dict[str, torch.Tensor],
                          model_outputs: Dict[str, torch.Tensor],
                          agent_outputs: Optional[Dict[str, torch.Tensor]] = None,
                          old_agent_outputs: Optional[Dict[str, torch.Tensor]] = None,
                          exposure_data: Optional[Dict[str, torch.Tensor]] = None,
                          model_components: Optional[Dict[str, Any]] = None,
                          step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining all components
        
        Args:
            batch_data: Input batch data
            model_outputs: Model forward pass outputs
            agent_outputs: Current agent outputs for RL
            old_agent_outputs: Previous agent outputs for PPO
            exposure_data: Exposure distribution data
            model_components: Model components for regularization
            step: Current training step
            
        Returns:
            Dict containing all loss components and total loss
        """
        all_losses = {}
        
        # Recommendation losses
        rec_losses = self.compute_recommendation_loss(batch_data, model_outputs)
        all_losses.update(rec_losses)
        
        # RL losses (only during MARL training phase)
        if self.training_phase in ['marl', 'finetune'] and agent_outputs and old_agent_outputs:
            rl_losses = self.compute_rl_losses(agent_outputs, old_agent_outputs)
            all_losses.update(rl_losses)
        
        # Fairness losses
        if exposure_data:
            fairness_losses = self.compute_fairness_losses(exposure_data)
            all_losses.update(fairness_losses)
        
        # Regularization losses
        if model_components:
            reg_losses = self.compute_regularization_losses(model_components)
            all_losses.update(reg_losses)
        
        # Combine all losses
        total_loss_dict = self.multi_loss.compute_total_loss(all_losses, step)
        
        # Apply loss scaling if needed
        if self.loss_scale != 1.0:
            total_loss_dict['total_loss'] *= self.loss_scale
        
        return total_loss_dict
    
    def backward_and_step(self, loss: torch.Tensor, optimizers: Dict[str, torch.optim.Optimizer],
                         schedulers: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Perform backward pass and optimization step with gradient clipping
        
        Args:
            loss: Total loss tensor
            optimizers: Dict of optimizers for different components
            schedulers: Optional learning rate schedulers
            
        Returns:
            Dict of gradient norms and optimization statistics
        """
        stats = {}
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping and optimization for each optimizer
        for opt_name, optimizer in optimizers.items():
            # Get parameters for this optimizer
            params = []
            for param_group in optimizer.param_groups:
                params.extend(param_group['params'])
            
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            stats[f'{opt_name}_grad_norm'] = grad_norm.item()
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad()
            
            # Learning rate scheduling
            if schedulers and opt_name in schedulers:
                schedulers[opt_name].step()
                stats[f'{opt_name}_lr'] = schedulers[opt_name].get_last_lr()[0]
        
        return stats
    
    def set_training_phase(self, phase: str):
        """Set current training phase"""
        valid_phases = ['pretrain', 'marl', 'finetune']
        if phase not in valid_phases:
            raise ValueError(f"Invalid training phase: {phase}. Must be one of {valid_phases}")
        
        self.training_phase = phase
        logger.info(f"Training phase set to: {phase}")
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return self.multi_loss.loss_weights.copy()
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights"""
        self.multi_loss.loss_weights.update(new_weights)
        logger.info(f"Updated loss weights: {new_weights}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive loss statistics"""
        return self.multi_loss.get_loss_statistics()
    
    def reset_statistics(self):
        """Reset loss statistics"""
        self.multi_loss.reset_history()

# Utility functions for loss computation
def compute_advantages_gae(rewards: torch.Tensor, values: torch.Tensor,
                          dones: torch.Tensor, gamma: float = 0.99,
                          lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: [batch_size, seq_len] - Rewards
        values: [batch_size, seq_len + 1] - Value estimates
        dones: [batch_size, seq_len] - Done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages, returns
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        gae = delta + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
    
    returns = advantages + values[:, :-1]
    return advantages, returns

def compute_discounted_returns(rewards: torch.Tensor, dones: torch.Tensor,
                              gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted returns
    
    Args:
        rewards: [batch_size, seq_len] - Rewards
        dones: [batch_size, seq_len] - Done flags
        gamma: Discount factor
        
    Returns:
        Discounted returns
    """
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)
    
    running_return = torch.zeros(batch_size, device=rewards.device)
    for t in reversed(range(seq_len)):
        running_return = rewards[:, t] + gamma * running_return * (1 - dones[:, t])
        returns[:, t] = running_return
    
    return returns

# Factory function
def create_loss_manager(config: Dict[str, Any], device: str = 'cuda') -> LossManager:
    """
    Factory function to create LossManager with configuration
    
    Args:
        config: Configuration dictionary from YAML
        device: Device to use for computation
        
    Returns:
        Configured LossManager instance
    """
    return LossManager(config, device)

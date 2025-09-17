"""
Enhanced MARL Two-Tower Recommendation System Trainer

Integrates ContextGNN, hierarchical multi-agent RL, GNN communication,
fair sampling, contrastive learning, and stable-rank regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
import time
import wandb
from dataclasses import dataclass

from src.agents.genre_agent import GenreAgentManager
from src.agents.exposure_manager import ExposureManagerController
from src.agents.marl_comm import HierarchicalMARLController
from src.models.contextgnn import ContextGNNManager
from src.models.item_tower import ItemTower
from src.training.ppo_updater import PPOUpdater
from src.training.fair_sampler import FairSampler
from src.training.loss_utils import compute_bpr_loss, compute_contrastive_loss, compute_stable_rank_penalty
from src.data.dataloader import RecommendationDataLoader
from src.utils.metrics import MetricsCalculator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class TrainingState:
    """Tracks training state and statistics"""
    episode: int = 0
    total_steps: int = 0
    best_hr10: float = 0.0
    best_ndcg10: float = 0.0
    best_gini: float = 1.0
    episodes_since_improvement: int = 0

class EnhancedMARLTrainer:
    """
    Main trainer for Enhanced MARL Two-Tower Recommendation System.
    
    Features:
    - Multi-agent coordination with genre-specific agents
    - ContextGNN user encoding
    - Hierarchical MARL with GNN communication
    - GINI fairness agent coordination
    - BUHS integration for long-tail discovery
    - PPO-based policy optimization
    - Multi-objective loss optimization
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize training state
        self.training_state = TrainingState()
        
        # Initialize components
        self._initialize_models()
        self._initialize_optimizers()
        self._initialize_training_components()
        
        # Logging and monitoring
        self._setup_logging()
        
        logger.info(f"Enhanced MARL Trainer initialized on {self.device}")
        logger.info(f"Total parameters: {self._count_parameters():,}")

    def _initialize_models(self):
        """Initialize all model components"""
        
        # ContextGNN User Tower
        self.contextgnn_manager = ContextGNNManager(self.config)
        self.contextgnn = self.contextgnn_manager.get_model()
        
        # Item Tower with Genre-Aware Encoding
        self.item_tower = ItemTower(self.config).to(self.device)
        
        # Genre Agents Manager
        self.genre_agent_manager = GenreAgentManager(self.config)
        
        # Exposure Manager (GINI Agent)
        self.exposure_controller = ExposureManagerController(self.config)
        self.exposure_manager = self.exposure_controller.get_main_manager()
        
        # Hierarchical MARL Controller
        self.marl_controller = HierarchicalMARLController(self.config).to(self.device)
        
        # Fair Sampler
        self.fair_sampler = FairSampler(self.config)
        
        # PPO Updater
        self.ppo_updater = PPOUpdater(self.config)

    def _initialize_optimizers(self):
        """Initialize optimizers for all components"""
        
        # Main model optimizers
        self.optimizers = {}
        
        # ContextGNN optimizer
        self.optimizers['contextgnn'] = AdamW(
            self.contextgnn.parameters(),
            lr=self.config['training']['base_lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Item tower optimizer
        self.optimizers['item_tower'] = AdamW(
            self.item_tower.parameters(),
            lr=self.config['training']['base_lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Genre agents optimizers
        for genre, agent in self.genre_agent_manager.get_all_agents().items():
            self.optimizers[f'agent_{genre}'] = Adam(
                agent.parameters(),
                lr=self.config['training']['agent_lr']
            )
        
        # Exposure manager optimizer
        self.optimizers['exposure_manager'] = Adam(
            self.exposure_manager.parameters(),
            lr=self.config['training']['exposure_manager_lr']
        )
        
        # MARL controller optimizer
        self.optimizers['marl_controller'] = Adam(
            self.marl_controller.parameters(),
            lr=self.config['training']['agent_lr']
        )
        
        # Learning rate schedulers
        self.schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['training']['min_lr']
            )

    def _initialize_training_components(self):
        """Initialize training-specific components"""
        
        # Experience buffers for each agent
        buffer_size = self.config['training'].get('buffer_size', 100000)
        self.experience_buffers = {}
        
        for genre in self.config['dataset']['genres']:
            self.experience_buffers[genre] = deque(maxlen=buffer_size)
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(self.config)
        
        # Training hyperparameters
        self.batch_size = self.config['training']['batch_size']
        self.num_epochs = self.config['training']['num_epochs']
        self.update_frequency = self.config['training']['update_frequency']
        self.eval_frequency = self.config['training']['eval_frequency']
        
        # Loss weights
        self.loss_weights = self.config['loss_weights']
        
        # Early stopping
        self.patience = self.config['training']['patience']

    def _setup_logging(self):
        """Setup experiment logging"""
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb'].get('entity'),
                config=self.config,
                tags=self.config['logging']['wandb']['tags']
            )

    def train(self, dataloader: RecommendationDataLoader):
        """Main training loop"""
        
        logger.info("Starting Enhanced MARL training...")
        
        for epoch in range(self.num_epochs):
            self.training_state.episode = epoch
            
            # Training phase
            train_metrics = self._train_epoch(dataloader)
            
            # Update learning rates
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Evaluation phase
            if epoch % self.eval_frequency == 0:
                val_metrics = self._evaluate_epoch(dataloader.val_loader)
                
                # Check for improvement
                improved = self._check_improvement(val_metrics)
                
                # Logging
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Early stopping check
                if not improved:
                    self.training_state.episodes_since_improvement += 1
                    if self.training_state.episodes_since_improvement >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                else:
                    self.training_state.episodes_since_improvement = 0
                    self._save_checkpoint(epoch, val_metrics)
        
        logger.info("Training completed!")

    def _train_epoch(self, dataloader: RecommendationDataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        # Set models to training mode
        self._set_training_mode(True)
        
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader.train_loader):
            batch_metrics = self._train_step(batch)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key].append(value)
            
            # Agent updates
            if batch_idx % self.update_frequency == 0 and batch_idx > 0:
                self._update_agents()
            
            self.training_state.total_steps += 1
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass through ContextGNN
        user_embeddings, attention_info = self.contextgnn(
            user_ids=batch['user_ids'],
            user_history=batch['user_history'],
            user_demographics=batch['user_demographics'],
            item_embeddings=batch['item_embeddings']
        )
        
        # Forward pass through Item Tower
        item_embeddings = self.item_tower(
            item_features=batch['item_features'],
            item_genres=batch['item_genres']
        )
        
        # MARL Controller coordination
        coordinated_output, updated_agent_states, marl_components = self.marl_controller(
            agent_states=user_embeddings,
            return_components=True
        )
        
        # Genre agents decision making
        genre_actions = {}
        genre_values = {}
        for genre, agent in self.genre_agent_manager.get_all_agents().items():
            genre_state = self._get_genre_state(batch, genre)
            action_logits, values = agent(genre_state)
            genre_actions[genre] = action_logits
            genre_values[genre] = values
        
        # Exposure Manager fairness adjustments
        exposure_adjustments, exposure_components = self.exposure_manager(
            return_components=True
        )
        
        # Compute recommendation scores
        scores = self._compute_scores(
            user_embeddings, item_embeddings, 
            genre_actions, exposure_adjustments
        )
        
        # Compute multi-objective loss
        losses = self._compute_losses(
            batch, scores, genre_actions, genre_values,
            marl_components, exposure_components
        )
        
        total_loss = self._combine_losses(losses)
        
        # Backward pass
        self.optimizers['contextgnn'].zero_grad()
        self.optimizers['item_tower'].zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.contextgnn.parameters(), 
            max_norm=self.config['training'].get('max_grad_norm', 1.0)
        )
        torch.nn.utils.clip_grad_norm_(
            self.item_tower.parameters(), 
            max_norm=self.config['training'].get('max_grad_norm', 1.0)
        )
        
        self.optimizers['contextgnn'].step()
        self.optimizers['item_tower'].step()
        
        # Store experiences for RL agents
        self._store_experiences(batch, genre_actions, genre_values, scores)
        
        # Update exposure tracking
        self.exposure_manager.update_exposure_tracking(
            recommendations=scores.topk(self.config['evaluation']['k_values'][0])[1],
            genre_assignments=batch['item_genres']
        )
        
        return {
            'total_loss': total_loss.item(),
            'bpr_loss': losses['bpr_loss'].item(),
            'ppo_loss': losses['ppo_loss'].item(),
            'contrastive_loss': losses['contrastive_loss'].item(),
            'stable_rank_loss': losses['stable_rank_loss'].item(),
            'gini_penalty': losses['gini_penalty'].item()
        }

    def _compute_losses(self, batch, scores, genre_actions, genre_values, 
                       marl_components, exposure_components) -> Dict[str, torch.Tensor]:
        """Compute all loss components"""
        
        losses = {}
        
        # BPR (Bayesian Personalized Ranking) Loss
        losses['bpr_loss'] = compute_bpr_loss(
            scores, batch['positive_items'], batch['negative_items']
        )
        
        # PPO Loss for genre agents
        losses['ppo_loss'] = self.ppo_updater.compute_ppo_loss(
            genre_actions, genre_values, batch['rewards']
        ) if 'rewards' in batch else torch.tensor(0.0, device=self.device)
        
        # Contrastive Loss for long-tail items
        losses['contrastive_loss'] = compute_contrastive_loss(
            batch['user_embeddings'], batch['tail_items'], batch['head_items']
        ) if self.config['loss_weights']['contrastive_loss'] > 0 else torch.tensor(0.0, device=self.device)
        
        # Stable Rank Regularization
        losses['stable_rank_loss'] = compute_stable_rank_penalty(
            torch.cat([batch['user_embeddings'], batch['item_embeddings']], dim=0)
        ) if self.config['loss_weights']['stable_rank_loss'] > 0 else torch.tensor(0.0, device=self.device)
        
        # GINI Penalty
        current_gini = exposure_components['genre_gini'] if exposure_components else torch.tensor(0.0, device=self.device)
        gini_threshold = self.config['marl_controller']['exposure_manager']['gini_threshold']
        losses['gini_penalty'] = F.relu(current_gini - gini_threshold)
        
        return losses

    def _combine_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses with configured weights"""
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name, 0.0)
            total_loss += weight * loss_value
        
        return total_loss

    def _update_agents(self):
        """Update RL agents using PPO"""
        
        for genre, agent in self.genre_agent_manager.get_all_agents().items():
            if len(self.experience_buffers[genre]) >= self.batch_size:
                experiences = list(self.experience_buffers[genre])
                self.ppo_updater.update_agent(agent, experiences, self.optimizers[f'agent_{genre}'])
                self.experience_buffers[genre].clear()
        
        # Update exposure manager
        exposure_reward = self.exposure_manager.compute_reward()
        self.ppo_updater.update_exposure_manager(
            self.exposure_manager, exposure_reward, self.optimizers['exposure_manager']
        )
        
        # Update MARL controller
        self.optimizers['marl_controller'].zero_grad()
        marl_loss = torch.tensor(0.0, device=self.device)  # Placeholder for MARL-specific loss
        if marl_loss.requires_grad:
            marl_loss.backward()
            self.optimizers['marl_controller'].step()

    def _evaluate_epoch(self, val_loader) -> Dict[str, float]:
        """Evaluate model performance"""
        
        self._set_training_mode(False)
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                user_embeddings, _ = self.contextgnn(
                    user_ids=batch['user_ids'],
                    user_history=batch['user_history'],
                    user_demographics=batch['user_demographics'],
                    item_embeddings=batch['item_embeddings']
                )
                
                item_embeddings = self.item_tower(
                    item_features=batch['item_features'],
                    item_genres=batch['item_genres']
                )
                
                scores = torch.mm(user_embeddings, item_embeddings.t())
                
                all_scores.append(scores.cpu())
                all_labels.append(batch['labels'].cpu())
        
        # Compute metrics
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = self.metrics_calculator.compute_metrics(all_scores, all_labels)
        
        # Add fairness metrics
        fairness_stats = self.exposure_manager.get_statistics()
        metrics.update(fairness_stats)
        
        return metrics

    def _check_improvement(self, metrics: Dict[str, float]) -> bool:
        """Check if model performance improved"""
        
        hr10 = metrics.get('hr@10', 0.0)
        ndcg10 = metrics.get('ndcg@10', 0.0)
        gini = metrics.get('current_genre_gini', 1.0)
        
        improved = (
            hr10 > self.training_state.best_hr10 or
            ndcg10 > self.training_state.best_ndcg10 or
            gini < self.training_state.best_gini
        )
        
        if improved:
            self.training_state.best_hr10 = max(self.training_state.best_hr10, hr10)
            self.training_state.best_ndcg10 = max(self.training_state.best_ndcg10, ndcg10)
            self.training_state.best_gini = min(self.training_state.best_gini, gini)
        
        return improved

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]):
        """Log metrics for current epoch"""
        
        # Console logging
        logger.info(f"Epoch {epoch:4d} | "
                   f"Train Loss: {train_metrics.get('total_loss', 0.0):.4f} | "
                   f"Val HR@10: {val_metrics.get('hr@10', 0.0):.4f} | "
                   f"Val NDCG@10: {val_metrics.get('ndcg@10', 0.0):.4f} | "
                   f"GINI: {val_metrics.get('current_genre_gini', 0.0):.4f}")
        
        # Wandb logging
        if self.config['logging']['wandb']['enabled']:
            log_dict = {
                'epoch': epoch,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()}
            }
            wandb.log(log_dict)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'contextgnn_state_dict': self.contextgnn.state_dict(),
            'item_tower_state_dict': self.item_tower.state_dict(),
            'marl_controller_state_dict': self.marl_controller.state_dict(),
            'exposure_manager_state_dict': self.exposure_manager.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'training_state': self.training_state,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save genre agents
        for genre, agent in self.genre_agent_manager.get_all_agents().items():
            checkpoint[f'agent_{genre}_state_dict'] = agent.state_dict()
        
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _set_training_mode(self, training: bool):
        """Set training mode for all models"""
        self.contextgnn.train(training)
        self.item_tower.train(training)
        self.marl_controller.train(training)
        self.exposure_manager.train(training)
        
        for agent in self.genre_agent_manager.get_all_agents().values():
            agent.train(training)

    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        total_params = 0
        
        # Count parameters in all models
        for model in [self.contextgnn, self.item_tower, self.marl_controller, self.exposure_manager]:
            total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count genre agents parameters
        for agent in self.genre_agent_manager.get_all_agents().values():
            total_params += sum(p.numel() for p in agent.parameters() if p.requires_grad)
        
        return total_params

    def _get_genre_state(self, batch: Dict[str, torch.Tensor], genre: str) -> torch.Tensor:
        """Extract genre-specific state from batch"""
        # Implementation depends on how genre-specific states are structured
        # This is a placeholder - should be implemented based on your data structure
        return batch['user_embeddings']  # Simplified

    def _compute_scores(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor,
                       genre_actions: Dict[str, torch.Tensor], 
                       exposure_adjustments: torch.Tensor) -> torch.Tensor:
        """Compute final recommendation scores with fairness adjustments"""
        
        # Base scores
        base_scores = torch.mm(user_embeddings, item_embeddings.t())
        
        # Apply genre agent preferences and fairness adjustments
        # This is simplified - actual implementation would be more complex
        adjusted_scores = base_scores * exposure_adjustments.mean()
        
        return adjusted_scores

    def _store_experiences(self, batch: Dict[str, torch.Tensor], 
                          genre_actions: Dict[str, torch.Tensor],
                          genre_values: Dict[str, torch.Tensor],
                          scores: torch.Tensor):
        """Store experiences for RL training"""
        
        # Extract rewards from environment feedback (simplified)
        rewards = self._compute_rewards(batch, scores)
        
        for genre in genre_actions.keys():
            experience = {
                'state': batch['user_embeddings'],
                'action': genre_actions[genre],
                'reward': rewards.get(genre, torch.zeros(1)),
                'value': genre_values[genre],
                'next_state': batch['user_embeddings']  # Simplified
            }
            self.experience_buffers[genre].append(experience)

    def _compute_rewards(self, batch: Dict[str, torch.Tensor], 
                        scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute rewards for RL agents"""
        
        # Compute genre-specific rewards based on HR, NDCG, fairness, etc.
        # This is a simplified implementation
        rewards = {}
        
        for genre in self.config['dataset']['genres']:
            # Placeholder reward computation
            rewards[genre] = torch.randn(batch['user_ids'].size(0))
        
        return rewards

# Training script entry point
def main():
    """Main training script"""
    import yaml
    
    # Load configuration
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = EnhancedMARLTrainer(config, device)
    
    # Initialize dataloader
    dataloader = RecommendationDataLoader(config)
    
    # Start training
    trainer.train(dataloader)

if __name__ == "__main__":
    main()

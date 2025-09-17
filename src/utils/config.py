"""
Enhanced MARL Two-Tower Recommendation System - Configuration Management

Comprehensive configuration management supporting:
- YAML-based configuration loading
- Environment-specific overrides
- Runtime configuration validation
- Dynamic parameter updates
- Multi-component configuration coordination
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from copy import deepcopy
import torch

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Supported device types"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class SamplingMethod(Enum):
    """Sampling strategies for fair sampling"""
    UNIFORM = "uniform"
    POPULARITY_AWARE = "popularity_aware"
    INVERSE_FREQUENCY = "inverse_frequency"
    GENRE_AWARE_INVERSE_FREQUENCY = "genre_aware_inverse_frequency"
    BUHS_BIASED = "buhs_biased"

class CommunicationType(Enum):
    """Agent communication strategies"""
    GAT = "gat"
    GCN = "gcn"
    TRANSFORMER = "transformer"

@dataclass
class ExperimentConfig:
    """Experiment metadata configuration"""
    name: str = "enhanced_marl_rec"
    version: str = "v2.0"
    description: str = "Enhanced MARL Two-Tower Recommendation System"
    seed: int = 42
    device: str = DeviceType.AUTO.value
    tags: List[str] = field(default_factory=lambda: ["marl", "contextgnn", "fairness", "two-tower"])

@dataclass
class DatasetConfig:
    """Dataset configuration for MovieLens and other datasets"""
    name: str = "movielens-1m"
    path: str = "data/ml-1m"
    
    # Data characteristics
    num_users: int = 6040
    num_items: int = 3706
    num_interactions: int = 1000209
    sparsity: float = 0.9553
    
    # Genre information
    genres: List[str] = field(default_factory=lambda: [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western", "Documentary"
    ])
    num_genres: int = 18
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Preprocessing parameters
    min_interactions_per_user: int = 20
    min_interactions_per_item: int = 5
    negative_sampling_ratio: int = 4
    max_sequence_length: int = 50
    implicit_threshold: int = 4
    
    # File paths
    ratings_file: str = "ratings.dat"
    users_file: str = "users.dat"
    movies_file: str = "movies.dat"

@dataclass
class ContextGNNConfig:
    """ContextGNN user tower configuration"""
    # Architecture
    input_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.2
    
    # Graph construction
    k_neighbors: int = 20
    edge_threshold: float = 0.1
    temporal_decay: float = 0.95
    
    # Feature dimensions
    user_embed_dim: int = 64
    temporal_dim: int = 16
    demographic_dim: int = 32
    
    # User demographic settings
    age_bins: int = 7
    occupation_categories: int = 21
    gender_categories: int = 2

@dataclass
class MARLControllerConfig:
    """Hierarchical MARL Controller configuration"""
    # Global coordinator
    coordinator_hidden_dim: int = 128
    coordinator_output_dim: int = 64
    coordinator_num_layers: int = 2
    
    # GNN Communication Layer
    communication_embed_dim: int = 64
    communication_num_heads: int = 4
    communication_num_layers: int = 2
    communication_message_dim: int = 32
    communication_dropout: float = 0.1
    communication_type: str = CommunicationType.GAT.value
    
    # Genre Agents
    num_genre_agents: int = 18
    genre_agent_state_dim: int = 64
    genre_agent_action_dim: int = 32
    genre_agent_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    
    # Genre-specific learning rate multipliers
    genre_lr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "Drama": 1.2, "Comedy": 1.2, "Thriller": 1.1, "Action": 1.1, "Romance": 1.1,
        "Adventure": 1.0, "Crime": 1.0, "Sci-Fi": 0.9, "Horror": 0.9, "Fantasy": 0.8,
        "Children": 0.8, "Animation": 0.8, "War": 0.7, "Musical": 0.7,
        "Western": 0.6, "Film-Noir": 0.6, "Documentary": 0.6, "Mystery": 0.6
    })

@dataclass
class ExposureManagerConfig:
    """Exposure Manager (GINI Agent) configuration"""
    input_dim: int = 18  # Number of genres
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 18
    gini_threshold: float = 0.6
    adjustment_strength: float = 1.0
    smoothing_factor: float = 0.9
    history_window: int = 1000

@dataclass
class ItemTowerConfig:
    """Item Tower configuration"""
    # Base encoder
    base_encoder_input_dim: int = 512
    base_encoder_hidden_dims: List[int] = field(default_factory=lambda: [384, 256, 128])
    base_encoder_output_dim: int = 128
    base_encoder_dropout: float = 0.2
    
    # Genre-aware refinement
    num_genre_encoders: int = 18
    genre_encoder_input_dim: int = 64
    genre_encoder_hidden_dim: int = 32
    genre_encoder_output_dim: int = 32
    
    # Feature dimensions
    item_embed_dim: int = 64
    genre_embed_dim: int = 32
    text_embed_dim: int = 384  # SBERT dimension
    year_embed_dim: int = 16
    
    # Year processing
    year_range: tuple = (1919, 2000)
    
    # Title processing
    title_processing: Dict[str, Any] = field(default_factory=lambda: {
        "max_length": 64,
        "embedding_model": "all-MiniLM-L6-v2",
        "freeze_embeddings": True
    })

@dataclass
class BUHSConfig:
    """Biased User History Synthesis configuration"""
    enabled: bool = True
    embed_dim: int = 384
    hidden_dim: int = 256
    output_dim: int = 32
    num_heads: int = 8
    dropout: float = 0.2
    alpha: float = 1.0  # Inverse popularity weighting strength
    max_history_length: int = 50
    temperature: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration"""
    # General training parameters
    num_epochs: int = 100
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    
    # Learning rates
    base_lr: float = 1e-4
    agent_lr: float = 1e-4
    exposure_manager_lr: float = 1e-4
    
    # Optimizers
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Learning rate scheduling
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Update frequencies
    update_frequency: int = 10  # episodes
    eval_frequency: int = 5   # epochs
    
    # Gradient clipping
    max_grad_norm: float = 1.0

@dataclass
class PPOConfig:
    """PPO training configuration"""
    # Core PPO parameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    
    # Update settings
    target_kl: float = 0.01
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

@dataclass
class LossWeights:
    """Loss function weights"""
    # Primary losses
    bpr_loss: float = 1.0
    ppo_loss: float = 0.5
    
    # Advanced losses
    contrastive_loss: float = 0.3
    stable_rank_loss: float = 0.2
    
    # Fairness penalties
    gini_penalty: float = 0.4
    exposure_penalty: float = 0.1

@dataclass
class FairSamplingConfig:
    """Fair sampling configuration"""
    enabled: bool = True
    method: str = SamplingMethod.INVERSE_FREQUENCY.value
    temperature: float = 1.0
    min_frequency: int = 1
    update_frequency: int = 100
    
    # Genre importance weights
    genre_importance: Dict[str, float] = field(default_factory=lambda: {
        "major_genres_boost": 1.2,
        "minor_genres_boost": 2.0
    })

@dataclass
class RewardConfig:
    """Reward models configuration"""
    # Genre agent rewards
    genre_agent: Dict[str, float] = field(default_factory=lambda: {
        "hit_rate_weight": 0.4,
        "ndcg_weight": 0.4,
        "miss_penalty": 0.15,
        "fairness_penalty": 0.05
    })
    
    # Exposure manager rewards
    exposure_manager: Dict[str, float] = field(default_factory=lambda: {
        "gini_weight": 1.0,
        "coverage_weight": 0.3,
        "diversity_weight": 0.2
    })
    
    # Long-tail promotion
    long_tail: Dict[str, float] = field(default_factory=lambda: {
        "tail_threshold": 0.2,
        "promotion_bonus": 0.5
    })

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "hr", "ndcg", "recall", "precision", "coverage", "gini", "diversity"
    ])
    
    # Top-K values
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    
    # Fairness evaluation
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "gini_coefficient", "demographic_parity", "individual_fairness"
    ])
    
    # Long-tail evaluation
    tail_evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "percentiles": [20, 50, 80]
    })

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    # Basic logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # Weights & Biases
    wandb: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "project": "enhanced-marl-rec",
        "entity": None,
        "tags": ["marl", "contextgnn", "fairness", "two-tower"]
    })
    
    # TensorBoard
    tensorboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "log_dir": "runs"
    })
    
    # Model checkpointing
    checkpoints: Dict[str, Any] = field(default_factory=lambda: {
        "save_dir": "checkpoints",
        "save_frequency": 5,
        "max_checkpoints": 5
    })

@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Serving parameters
    batch_size: int = 512
    max_candidates: int = 1000
    top_k: int = 50
    
    # Caching
    cache_user_embeddings: bool = True
    cache_item_embeddings: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Performance targets
    target_latency_ms: int = 50
    target_qps: int = 2000

@dataclass
class SystemConfig:
    """System configuration"""
    # Memory management
    max_memory_gb: int = 32
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    
    # Distributed training
    distributed: bool = False
    num_gpus: int = 1
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False

class EnhancedMARLConfig:
    """
    Main configuration class for Enhanced MARL Two-Tower Recommendation System.
    
    Provides comprehensive configuration management with:
    - YAML file loading and merging
    - Environment variable overrides
    - Runtime validation
    - Dynamic updates
    - Component-specific configurations
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional configuration overrides
        """
        self.config_path = config_path
        
        # Initialize all configuration components
        self.experiment = ExperimentConfig()
        self.dataset = DatasetConfig()
        self.contextgnn = ContextGNNConfig()
        self.marl_controller = MARLControllerConfig()
        self.exposure_manager = ExposureManagerConfig()
        self.item_tower = ItemTowerConfig()
        self.buhs = BUHSConfig()
        self.training = TrainingConfig()
        self.ppo = PPOConfig()
        self.loss_weights = LossWeights()
        self.fair_sampling = FairSamplingConfig()
        self.rewards = RewardConfig()
        self.evaluation = EvaluationConfig()
        self.logging = LoggingConfig()
        self.inference = InferenceConfig()
        self.system = SystemConfig()
        
        # Load configuration if path provided
        if config_path:
            self.load_from_file(config_path)
        
        # Apply any additional overrides
        if kwargs:
            self.update_from_dict(kwargs)
        
        # Validate and setup
        self._setup_device()
        self._validate_config()
        
        logger.info(f"Enhanced MARL Configuration initialized: {self.experiment.name} v{self.experiment.version}")

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self.update_from_dict(config_data)
            logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {section_name}.{key}")

    def save_to_file(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {output_path}: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'experiment': asdict(self.experiment),
            'dataset': asdict(self.dataset),
            'contextgnn': asdict(self.contextgnn),
            'marl_controller': asdict(self.marl_controller),
            'exposure_manager': asdict(self.exposure_manager),
            'item_tower': asdict(self.item_tower),
            'buhs': asdict(self.buhs),
            'training': asdict(self.training),
            'ppo': asdict(self.ppo),
            'loss_weights': asdict(self.loss_weights),
            'fair_sampling': asdict(self.fair_sampling),
            'rewards': asdict(self.rewards),
            'evaluation': asdict(self.evaluation),
            'logging': asdict(self.logging),
            'inference': asdict(self.inference),
            'system': asdict(self.system)
        }

    def _setup_device(self) -> None:
        """Setup and validate device configuration."""
        if self.experiment.device == DeviceType.AUTO.value:
            if torch.cuda.is_available():
                self.experiment.device = DeviceType.CUDA.value
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.experiment.device = DeviceType.MPS.value
            else:
                self.experiment.device = DeviceType.CPU.value
        
        # Validate device availability
        if self.experiment.device == DeviceType.CUDA.value and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.experiment.device = DeviceType.CPU.value
        
        logger.info(f"Device set to: {self.experiment.device}")

    def _validate_config(self) -> None:
        """Validate configuration consistency and constraints."""
        # Dataset validation
        if self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio != 1.0:
            raise ValueError("Data split ratios must sum to 1.0")
        
        if self.dataset.num_genres != len(self.dataset.genres):
            logger.warning(f"num_genres ({self.dataset.num_genres}) != len(genres) ({len(self.dataset.genres)})")
            self.dataset.num_genres = len(self.dataset.genres)
        
        # MARL Controller validation
        if self.marl_controller.num_genre_agents != self.dataset.num_genres:
            logger.warning(f"Adjusting num_genre_agents to match num_genres: {self.dataset.num_genres}")
            self.marl_controller.num_genre_agents = self.dataset.num_genres
        
        # Exposure Manager validation
        if self.exposure_manager.input_dim != self.dataset.num_genres:
            logger.warning(f"Adjusting exposure_manager input_dim to match num_genres: {self.dataset.num_genres}")
            self.exposure_manager.input_dim = self.dataset.num_genres
            self.exposure_manager.output_dim = self.dataset.num_genres
        
        # Training validation
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.training.base_lr <= 0:
            raise ValueError("learning rate must be positive")
        
        # PPO validation
        if not 0 < self.ppo.clip_epsilon < 1:
            raise ValueError("PPO clip_epsilon must be between 0 and 1")
        
        if not 0 < self.ppo.gamma <= 1:
            raise ValueError("PPO gamma must be between 0 and 1")

    def get_genre_config(self, genre: str) -> Dict[str, Any]:
        """Get genre-specific configuration."""
        lr_multiplier = self.marl_controller.genre_lr_multipliers.get(genre, 1.0)
        
        return {
            'genre_name': genre,
            'lr_multiplier': lr_multiplier,
            'state_dim': self.marl_controller.genre_agent_state_dim,
            'action_dim': self.marl_controller.genre_agent_action_dim,
            'hidden_dims': self.marl_controller.genre_agent_hidden_dims.copy(),
            'buhs_enabled': self.buhs.enabled,
            'buhs_config': asdict(self.buhs) if self.buhs.enabled else None
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration for initialization."""
        return {
            'device': self.experiment.device,
            'contextgnn_config': asdict(self.contextgnn),
            'item_tower_config': asdict(self.item_tower),
            'marl_config': asdict(self.marl_controller),
            'exposure_config': asdict(self.exposure_manager),
            'dataset_config': asdict(self.dataset)
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            'training': asdict(self.training),
            'ppo': asdict(self.ppo),
            'loss_weights': asdict(self.loss_weights),
            'fair_sampling': asdict(self.fair_sampling),
            'rewards': asdict(self.rewards)
        }

    def update_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameters dynamically."""
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested updates (e.g., 'training.batch_size')
                section, param = key.split('.', 1)
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, param):
                        setattr(section_obj, param, value)
                        logger.info(f"Updated {key}: {value}")
                    else:
                        logger.warning(f"Unknown parameter: {key}")
                else:
                    logger.warning(f"Unknown section: {section}")
            else:
                logger.warning(f"Use section.parameter format for updates: {key}")

    def create_experiment_config(self) -> Dict[str, Any]:
        """Create configuration for experiment tracking."""
        return {
            'config': self.to_dict(),
            'experiment_name': f"{self.experiment.name}_{self.experiment.version}",
            'tags': self.experiment.tags,
            'description': self.experiment.description
        }

# Configuration factory functions

def load_config(config_name: str = "base", config_dir: str = "configs") -> EnhancedMARLConfig:
    """
    Load configuration by name.
    
    Args:
        config_name: Configuration name (e.g., 'base', 'movielens', 'ablation')
        config_dir: Configuration directory
        
    Returns:
        Loaded configuration object
    """
    config_path = Path(config_dir) / f"{config_name}.yaml"
    return EnhancedMARLConfig(config_path)

def create_movielens_config() -> EnhancedMARLConfig:
    """Create MovieLens-specific configuration."""
    config = EnhancedMARLConfig()
    
    # MovieLens-specific overrides
    config.dataset.name = "movielens-1m"
    config.dataset.path = "data/ml-1m"
    config.experiment.description = "Enhanced MARL system optimized for MovieLens-1M dataset"
    config.experiment.tags.extend(["movielens-1m", "movies"])
    
    # Optimized parameters for MovieLens
    config.training.batch_size = 512
    config.training.base_lr = 2e-4
    config.ppo.clip_epsilon = 0.15
    config.ppo.gamma = 0.95
    
    return config

def create_ablation_config() -> EnhancedMARLConfig:
    """Create configuration for ablation studies."""
    config = EnhancedMARLConfig()
    
    config.experiment.name = "enhanced_marl_rec_ablation"
    config.experiment.description = "Comprehensive ablation study of MARL recommendation components"
    config.experiment.tags.extend(["ablation", "component-analysis"])
    
    return config

# Environment variable overrides

def apply_env_overrides(config: EnhancedMARLConfig) -> None:
    """Apply environment variable overrides."""
    env_mappings = {
        'BATCH_SIZE': ('training', 'batch_size', int),
        'LEARNING_RATE': ('training', 'base_lr', float),
        'NUM_EPOCHS': ('training', 'num_epochs', int),
        'DEVICE': ('experiment', 'device', str),
        'WANDB_PROJECT': ('logging', 'wandb', 'project', str),
        'GINI_THRESHOLD': ('exposure_manager', 'gini_threshold', float),
    }
    
    for env_var, (*path, type_func) in env_mappings.items():
        if env_var in os.environ:
            value = type_func(os.environ[env_var])
            
            # Navigate to the nested attribute
            obj = config
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            
            setattr(obj, path[-1], value)
            logger.info(f"Applied environment override: {env_var}={value}")

# Example usage
if __name__ == "__main__":
    # Basic configuration
    config = EnhancedMARLConfig()
    print(f"Default configuration created: {config.experiment.name}")
    
    # MovieLens configuration
    ml_config = create_movielens_config()
    print(f"MovieLens configuration: {ml_config.dataset.name}")
    
    # Save configuration
    config.save_to_file("configs/example_config.yaml")
    
    # Load from file
    loaded_config = load_config("base")
    print(f"Loaded configuration: {loaded_config.experiment.name}")
    
    # Apply environment overrides
    apply_env_overrides(config)

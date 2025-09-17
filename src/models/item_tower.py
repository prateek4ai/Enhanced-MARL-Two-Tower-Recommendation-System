"""
Item Tower Implementation for Enhanced MARL Two-Tower Recommendation System

Handles item feature encoding with genre-aware refinements and integration 
with the multi-agent RL controller for coordinated recommendation generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ItemFeatureEncoder(nn.Module):
    """
    Base item feature encoder for processing raw item features.
    """
    
    def __init__(
        self,
        item_vocab_size: int,
        item_embed_dim: int = 64,
        genre_embed_dim: int = 32,
        year_embed_dim: int = 16,
        text_embed_dim: int = 384,
        year_range: Tuple[int, int] = (1919, 2000),
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        super(ItemFeatureEncoder, self).__init__()
        
        self.item_vocab_size = item_vocab_size
        self.item_embed_dim = item_embed_dim
        self.genre_embed_dim = genre_embed_dim
        self.year_embed_dim = year_embed_dim
        self.text_embed_dim = text_embed_dim
        self.year_range = year_range
        self.device = device
        
        # Item ID embedding
        self.item_embedding = nn.Embedding(item_vocab_size, item_embed_dim)
        
        # Genre embedding (for multi-hot genre vectors)
        self.genre_projection = nn.Sequential(
            nn.Linear(18, genre_embed_dim),  # 18 genres in MovieLens
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Release year encoding
        year_span = year_range[1] - year_range[0] + 1
        self.year_embedding = nn.Embedding(year_span + 1, year_embed_dim)  # +1 for unknown
        
        # Text feature processing (SBERT title embeddings)
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, text_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_embed_dim // 2, text_embed_dim // 4)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.item_embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.year_embedding.weight, 0.0, 0.1)
        
        for module in [self.genre_projection, self.text_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        item_ids: torch.Tensor,
        genre_vectors: torch.Tensor,
        release_years: torch.Tensor,
        title_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode raw item features.
        
        Args:
            item_ids: [batch_size] - Item ID indices
            genre_vectors: [batch_size, 18] - Multi-hot genre vectors
            release_years: [batch_size] - Release years
            title_embeddings: [batch_size, 384] - SBERT title embeddings
            
        Returns:
            encoded_features: [batch_size, total_feature_dim]
        """
        # Item embeddings
        item_emb = self.item_embedding(item_ids)  # [batch_size, item_embed_dim]
        
        # Genre embeddings
        genre_emb = self.genre_projection(genre_vectors.float())  # [batch_size, genre_embed_dim]
        
        # Year embeddings (normalize years to embedding indices)
        year_indices = release_years - self.year_range[0]
        year_indices = torch.clamp(year_indices, 0, self.year_range[1] - self.year_range[0])
        year_emb = self.year_embedding(year_indices)  # [batch_size, year_embed_dim]
        
        # Text embeddings
        text_emb = self.text_projection(title_embeddings)  # [batch_size, text_embed_dim//4]
        
        # Concatenate all features
        encoded_features = torch.cat([item_emb, genre_emb, year_emb, text_emb], dim=-1)
        
        return encoded_features


class GenreAwareRefinement(nn.Module):
    """
    Genre-aware refinement layers for specialized item encoding per genre.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_genres: int = 18,
        dropout: float = 0.2
    ):
        super(GenreAwareRefinement, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_genres = num_genres
        
        # Genre-specific refinement networks
        self.genre_refiners = nn.ModuleDict()
        for i in range(num_genres):
            self.genre_refiners[str(i)] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Attention mechanism for genre importance
        self.genre_attention = nn.Sequential(
            nn.Linear(input_dim, num_genres),
            nn.Softmax(dim=-1)
        )
        
        # Final aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        item_features: torch.Tensor, 
        genre_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply genre-aware refinement.
        
        Args:
            item_features: [batch_size, input_dim] - Base item features
            genre_vectors: [batch_size, num_genres] - Multi-hot genre indicators
            
        Returns:
            refined_features: [batch_size, input_dim] - Genre-refined features
        """
        batch_size = item_features.size(0)
        
        # Compute genre attention weights
        genre_weights = self.genre_attention(item_features)  # [batch_size, num_genres]
        
        # Apply genre-specific refinements
        refinements = torch.zeros(batch_size, self.hidden_dim, device=item_features.device)
        
        for i in range(self.num_genres):
            # Get items that belong to this genre
            genre_mask = genre_vectors[:, i] > 0  # [batch_size]
            
            if genre_mask.any():
                # Apply genre-specific refinement
                masked_features = item_features[genre_mask]
                genre_refined = self.genre_refiners[str(i)](masked_features)
                
                # Weight by attention and genre membership
                attention_weight = genre_weights[genre_mask, i].unsqueeze(-1)
                genre_membership = genre_vectors[genre_mask, i].unsqueeze(-1)
                weighted_refined = genre_refined * attention_weight * genre_membership
                
                refinements[genre_mask] += weighted_refined
        
        # Aggregate base features with refinements
        combined = torch.cat([item_features, refinements], dim=-1)
        refined_features = self.aggregation(combined)
        
        return refined_features


class ItemTower(nn.Module):
    """
    Complete Item Tower for Enhanced MARL Two-Tower System.
    
    Features:
    - Multi-modal item feature encoding (ID, genres, year, text)
    - Genre-aware refinement layers for MARL integration
    - Configurable architecture for different datasets
    - Efficient caching support for inference
    """
    
    def __init__(
        self,
        # Architecture parameters
        input_dim: int = 512,
        hidden_dims: List[int] = [384, 256, 128],
        output_dim: int = 128,
        dropout: float = 0.2,
        
        # Item feature parameters
        item_vocab_size: int = 3706,
        item_embed_dim: int = 64,
        genre_embed_dim: int = 32,
        year_embed_dim: int = 16,
        text_embed_dim: int = 384,
        num_genres: int = 18,
        
        # MovieLens-specific parameters
        year_range: Tuple[int, int] = (1919, 2000),
        sbert_model: str = "all-MiniLM-L6-v2",
        freeze_sbert: bool = True,
        
        device: str = 'cuda'
    ):
        super(ItemTower, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_genres = num_genres
        self.device = device
        
        # Item feature encoder
        self.feature_encoder = ItemFeatureEncoder(
            item_vocab_size=item_vocab_size,
            item_embed_dim=item_embed_dim,
            genre_embed_dim=genre_embed_dim,
            year_embed_dim=year_embed_dim,
            text_embed_dim=text_embed_dim,
            year_range=year_range,
            dropout=dropout,
            device=device
        )
        
        # Calculate actual input dimension after feature encoding
        actual_input_dim = (item_embed_dim + genre_embed_dim + 
                          year_embed_dim + text_embed_dim // 4)
        
        # Base MLP encoder
        base_layers = []
        prev_dim = actual_input_dim
        
        for i, hdim in enumerate(hidden_dims):
            base_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hdim)
            ])
            prev_dim = hdim
        
        self.base_encoder = nn.Sequential(*base_layers)
        
        # Genre-aware refinement
        self.genre_refinement = GenreAwareRefinement(
            input_dim=hidden_dims[-1],
            hidden_dim=32,
            num_genres=num_genres,
            dropout=dropout
        )
        
        # Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(output_dim)
        )
        
        # SBERT model for text processing (optional, can be pre-computed)
        if not freeze_sbert:
            try:
                self.sbert_model = SentenceTransformer(sbert_model)
                for param in self.sbert_model.parameters():
                    param.requires_grad = not freeze_sbert
            except:
                logger.warning("Could not load SBERT model. Text features should be pre-computed.")
                self.sbert_model = None
        else:
            self.sbert_model = None
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"ItemTower initialized with output_dim={output_dim}, num_genres={num_genres}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_text_features(self, titles: List[str]) -> torch.Tensor:
        """
        Encode text features using SBERT (if available).
        
        Args:
            titles: List of item titles
            
        Returns:
            text_embeddings: [batch_size, text_embed_dim]
        """
        if self.sbert_model is not None:
            embeddings = self.sbert_model.encode(titles, convert_to_tensor=True)
            return embeddings.to(self.device)
        else:
            # Return zero embeddings if SBERT not available
            batch_size = len(titles)
            return torch.zeros(batch_size, 384, device=self.device)
    
    def forward(
        self,
        item_batch: Dict[str, torch.Tensor],
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through item tower.
        
        Args:
            item_batch: Dict containing:
                - 'item_ids': [batch_size] - Item ID indices
                - 'genre_vectors': [batch_size, num_genres] - Multi-hot genre vectors
                - 'release_years': [batch_size] - Release years
                - 'title_embeddings': [batch_size, 384] - Pre-computed SBERT embeddings
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            item_embeddings: [batch_size, output_dim] - Final item embeddings
            intermediates: Optional dict with intermediate representations
        """
        # Encode raw features
        encoded_features = self.feature_encoder(
            item_batch['item_ids'],
            item_batch['genre_vectors'],
            item_batch['release_years'],
            item_batch['title_embeddings']
        )
        
        # Base MLP encoding
        base_encoded = self.base_encoder(encoded_features)
        
        # Genre-aware refinement
        refined_features = self.genre_refinement(
            base_encoded, 
            item_batch['genre_vectors']
        )
        
        # Final projection
        item_embeddings = self.output_projection(refined_features)
        
        intermediates = None
        if return_intermediate:
            intermediates = {
                'encoded_features': encoded_features,
                'base_encoded': base_encoded,
                'refined_features': refined_features
            }
        
        return item_embeddings, intermediates
    
    def get_genre_specific_embeddings(
        self, 
        item_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get genre-specific item embeddings for MARL coordination.
        
        Args:
            item_batch: Item batch dictionary
            
        Returns:
            genre_embeddings: Dict mapping genre names to embeddings
        """
        # Get base embeddings
        item_embeddings, intermediates = self.forward(item_batch, return_intermediate=True)
        base_features = intermediates['base_encoded']
        
        genre_embeddings = {}
        genre_names = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
            "Romance", "Sci-Fi", "Thriller", "War", "Western", "Documentary"
        ]
        
        for i, genre_name in enumerate(genre_names):
            # Apply genre-specific refinement
            genre_mask = item_batch['genre_vectors'][:, i] > 0
            if genre_mask.any():
                genre_specific = self.genre_refinement.genre_refiners[str(i)](base_features[genre_mask])
                genre_embeddings[genre_name] = genre_specific
        
        return genre_embeddings
    
    def compute_similarity_matrix(
        self, 
        item_embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute item-item similarity matrix for candidate generation.
        
        Args:
            item_embeddings: [num_items, output_dim] - Item embeddings
            temperature: Temperature for softmax normalization
            
        Returns:
            similarity_matrix: [num_items, num_items] - Normalized similarities
        """
        # L2 normalize embeddings
        normalized_embeddings = F.normalize(item_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / temperature
        
        return similarity_matrix


class ItemTowerManager:
    """
    Manager class for ItemTower with utilities for batch processing and caching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize ItemTower
        item_config = config['item_tower']
        self.item_tower = ItemTower(
            input_dim=item_config['base_encoder']['input_dim'],
            hidden_dims=item_config['base_encoder']['hidden_dims'],
            output_dim=item_config['base_encoder']['output_dim'],
            dropout=item_config['base_encoder']['dropout'],
            item_vocab_size=config['dataset']['num_items'],
            num_genres=config['dataset']['num_genres'],
            year_range=item_config.get('year_range', (1919, 2000)),
            sbert_model=item_config.get('title_processing', {}).get('embedding_model', 'all-MiniLM-L6-v2'),
            device=self.device
        ).to(self.device)
        
        # Cache for pre-computed embeddings
        self.embedding_cache = {}
        
        logger.info("ItemTowerManager initialized successfully")
    
    def get_model(self) -> ItemTower:
        """Get the ItemTower model."""
        return self.item_tower
    
    def preprocess_item_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess item batch for ItemTower input.
        
        Args:
            batch: Raw batch data
            
        Returns:
            preprocessed_batch: Preprocessed batch ready for ItemTower
        """
        return {
            'item_ids': batch['item_ids'].to(self.device),
            'genre_vectors': batch['genre_vectors'].to(self.device),
            'release_years': batch['release_years'].to(self.device),
            'title_embeddings': batch['title_embeddings'].to(self.device)
        }
    
    def cache_item_embeddings(self, item_ids: List[int], embeddings: torch.Tensor):
        """Cache item embeddings for efficient inference."""
        for i, item_id in enumerate(item_ids):
            self.embedding_cache[item_id] = embeddings[i].detach().cpu()
    
    def get_cached_embeddings(self, item_ids: List[int]) -> Optional[torch.Tensor]:
        """Retrieve cached embeddings if available."""
        cached_embeddings = []
        for item_id in item_ids:
            if item_id in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[item_id])
            else:
                return None  # Not all items cached
        
        return torch.stack(cached_embeddings).to(self.device)

"""
Enhanced MARL Two-Tower Recommendation System - DataLoader

Comprehensive data loading and preprocessing for the Enhanced MARL recommendation system
supporting ContextGNN, BUHS, fair sampling, and multi-agent training.

Features:
- MovieLens-1M dataset processing with demographic and temporal features
- SBERT title embeddings integration
- Genre-specific data preparation for multi-agent RL
- BUHS-compatible user history sampling
- Fair sampling strategies for bias mitigation
- ContextGNN graph construction support
- Efficient batch processing with negative sampling
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import random
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class MovieLensDataProcessor:
    """
    Comprehensive MovieLens-1M dataset processor for Enhanced MARL system.
    """
    
    def __init__(self, data_path: str, config: Dict[str, Any]):
        self.data_path = data_path
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Dataset statistics from config
        self.num_users = config['dataset']['num_users']
        self.num_items = config['dataset']['num_items']
        self.num_genres = config['dataset']['num_genres']
        self.genres = config['dataset']['genres']
        
        # Initialize encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.occupation_encoder = LabelEncoder()
        
        # SBERT model for title embeddings
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Data containers
        self.ratings_df = None
        self.users_df = None
        self.movies_df = None
        self.interactions_graph = None
        self.user_item_matrix = None
        
        # Popularity and fairness tracking
        self.item_popularity = None
        self.genre_distribution = None
        
        logger.info("MovieLensDataProcessor initialized")

    def load_and_preprocess(self) -> Dict[str, Any]:
        """
        Load and preprocess MovieLens-1M dataset.
        
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Loading MovieLens-1M dataset...")
        
        # Load raw data files
        self._load_raw_data()
        
        # Process user features
        self._process_user_features()
        
        # Process movie features with SBERT embeddings
        self._process_movie_features()
        
        # Process interactions and create graph
        self._process_interactions()
        
        # Generate temporal features
        self._generate_temporal_features()
        
        # Compute popularity and fairness metrics
        self._compute_popularity_metrics()
        
        # Create train/val/test splits
        splits = self._create_data_splits()
        
        logger.info("Dataset preprocessing completed successfully")
        
        return {
            'users': self.users_df,
            'movies': self.movies_df,
            'ratings': self.ratings_df,
            'splits': splits,
            'metadata': self._get_metadata()
        }
    
    def _load_raw_data(self):
        """Load raw MovieLens data files."""
        # Load ratings (user_id::movie_id::rating::timestamp)
        self.ratings_df = pd.read_csv(
            f"{self.data_path}/ratings.dat",
            sep="::",
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Load users (user_id::gender::age::occupation::zip)
        self.users_df = pd.read_csv(
            f"{self.data_path}/users.dat",
            sep="::",
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python'
        )
        
        # Load movies (movie_id::title::genres)
        self.movies_df = pd.read_csv(
            f"{self.data_path}/movies.dat",
            sep="::",
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
        
        logger.info(f"Loaded {len(self.ratings_df)} ratings, {len(self.users_df)} users, {len(self.movies_df)} movies")
    
    def _process_user_features(self):
        """Process and encode user demographic features."""
        # Encode user IDs
        self.users_df['user_idx'] = self.user_encoder.fit_transform(self.users_df['user_id'])
        
        # Process gender
        self.users_df['gender_idx'] = self.gender_encoder.fit_transform(self.users_df['gender'])
        
        # Process age groups (MovieLens age groups: 1, 18, 25, 35, 45, 50, 56)
        age_bins = [0, 18, 25, 35, 45, 50, 56, 100]
        self.users_df['age_group'] = pd.cut(self.users_df['age'], bins=age_bins, labels=range(7))
        
        # Process occupation
        self.users_df['occupation_idx'] = self.occupation_encoder.fit_transform(self.users_df['occupation'])
        
        logger.info("User features processed successfully")
    
    def _process_movie_features(self):
        """Process movie features including SBERT embeddings."""
        # Encode movie IDs
        self.movies_df['movie_idx'] = self.item_encoder.fit_transform(self.movies_df['movie_id'])
        
        # Extract release year from title
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$')[0]
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
        self.movies_df['year'].fillna(1995, inplace=True)  # Fill missing with median year
        
        # Normalize year to [0, 1]
        self.movies_df['year_normalized'] = (self.movies_df['year'] - self.movies_df['year'].min()) / \
                                          (self.movies_df['year'].max() - self.movies_df['year'].min())
        
        # Process genres - create binary genre vectors
        all_genres = set()
        for genres_str in self.movies_df['genres']:
            all_genres.update(genres_str.split('|'))
        
        self.genre_list = sorted(list(all_genres))
        
        for genre in self.genre_list:
            self.movies_df[f'genre_{genre}'] = self.movies_df['genres'].str.contains(genre).astype(int)
        
        # Generate SBERT embeddings for movie titles
        logger.info("Generating SBERT embeddings for movie titles...")
        titles = self.movies_df['title'].tolist()
        title_embeddings = self.sbert_model.encode(titles, show_progress_bar=True)
        
        # Store embeddings
        self.movies_df['title_embedding'] = list(title_embeddings)
        
        logger.info("Movie features processed successfully")
    
    def _process_interactions(self):
        """Process user-item interactions and create interaction graph."""
        # Map original IDs to encoded indices
        user_id_to_idx = dict(zip(self.users_df['user_id'], self.users_df['user_idx']))
        movie_id_to_idx = dict(zip(self.movies_df['movie_id'], self.movies_df['movie_idx']))
        
        self.ratings_df['user_idx'] = self.ratings_df['user_id'].map(user_id_to_idx)
        self.ratings_df['movie_idx'] = self.ratings_df['movie_id'].map(movie_id_to_idx)
        
        # Convert to implicit feedback (rating >= 4 as positive)
        self.ratings_df['implicit_rating'] = (self.ratings_df['rating'] >= 4).astype(int)
        
        # Create interaction graph for ContextGNN
        self.interactions_graph = nx.Graph()
        
        for _, row in self.ratings_df.iterrows():
            if row['implicit_rating'] == 1:  # Only positive interactions
                self.interactions_graph.add_edge(
                    f"u_{row['user_idx']}", 
                    f"i_{row['movie_idx']}",
                    weight=row['rating'],
                    timestamp=row['timestamp']
                )
        
        logger.info("Interaction graph created successfully")
    
    def _generate_temporal_features(self):
        """Generate temporal features for ContextGNN."""
        # Convert timestamp to datetime
        self.ratings_df['datetime'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
        
        # Extract temporal features
        self.ratings_df['hour'] = self.ratings_df['datetime'].dt.hour
        self.ratings_df['day_of_week'] = self.ratings_df['datetime'].dt.dayofweek
        self.ratings_df['month'] = self.ratings_df['datetime'].dt.month
        
        # Create sinusoidal encodings for cyclic features
        self.ratings_df['hour_sin'] = np.sin(2 * np.pi * self.ratings_df['hour'] / 24)
        self.ratings_df['hour_cos'] = np.cos(2 * np.pi * self.ratings_df['hour'] / 24)
        
        # Create one-hot encoding for day of week
        for i in range(7):
            self.ratings_df[f'dow_{i}'] = (self.ratings_df['day_of_week'] == i).astype(int)
        
        logger.info("Temporal features generated successfully")
    
    def _compute_popularity_metrics(self):
        """Compute item popularity and genre distribution for fairness."""
        # Item popularity (interaction counts)
        item_counts = self.ratings_df['movie_idx'].value_counts()
        self.item_popularity = item_counts.reindex(range(self.num_items), fill_value=0)
        
        # Normalize popularity to [0, 1]
        self.item_popularity = self.item_popularity / self.item_popularity.max()
        
        # Genre distribution
        genre_cols = [col for col in self.movies_df.columns if col.startswith('genre_')]
        self.genre_distribution = self.movies_df[genre_cols].sum()
        
        logger.info("Popularity metrics computed successfully")
    
    def _create_data_splits(self) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits using leave-one-out strategy."""
        splits = {'train': [], 'val': [], 'test': []}
        
        # Group by user and sort by timestamp
        user_groups = self.ratings_df.groupby('user_idx')
        
        for user_idx, group in user_groups:
            group_sorted = group.sort_values('timestamp')
            
            if len(group_sorted) >= 3:  # Minimum 3 interactions for splits
                # Last interaction for test, second-to-last for validation
                splits['test'].append(group_sorted.iloc[-1:])
                splits['val'].append(group_sorted.iloc[-2:-1])
                splits['train'].append(group_sorted.iloc[:-2])
            elif len(group_sorted) >= 2:
                splits['test'].append(group_sorted.iloc[-1:])
                splits['train'].append(group_sorted.iloc[:-1])
            else:
                splits['train'].append(group_sorted)
        
        # Concatenate splits
        for split in splits:
            if splits[split]:
                splits[split] = pd.concat(splits[split], ignore_index=True)
            else:
                splits[split] = pd.DataFrame()
        
        logger.info(f"Data splits created: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_genres': len(self.genre_list),
            'genres': self.genre_list,
            'sparsity': 1 - (len(self.ratings_df) / (self.num_users * self.num_items)),
            'item_popularity': self.item_popularity.values,
            'genre_distribution': self.genre_distribution.to_dict(),
            'encoders': {
                'user': self.user_encoder,
                'item': self.item_encoder,
                'gender': self.gender_encoder,
                'occupation': self.occupation_encoder
            }
        }


class EnhancedMARLDataset(Dataset):
    """
    Enhanced Dataset for MARL Two-Tower Recommendation System.
    Supports ContextGNN, BUHS, and multi-agent training requirements.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        users_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        mode: str = 'train'
    ):
        self.data = data
        self.users_df = users_df
        self.movies_df = movies_df
        self.metadata = metadata
        self.config = config
        self.mode = mode
        
        self.device = config.get('device', 'cuda')
        self.max_seq_length = config['dataset']['max_sequence_length']
        self.negative_sampling_ratio = config['dataset']['negative_sampling_ratio']
        
        # Build user interaction histories
        self.user_histories = self._build_user_histories()
        
        # Genre information
        self.genres = metadata['genres']
        self.num_genres = len(self.genres)
        
        # Fair sampling setup
        self.fair_sampler = FairNegativeSampler(
            metadata['item_popularity'],
            config['fair_sampling']
        )
        
        logger.info(f"EnhancedMARLDataset initialized for {mode} with {len(data)} samples")
    
    def _build_user_histories(self) -> Dict[int, Dict[str, Any]]:
        """Build comprehensive user interaction histories."""
        histories = defaultdict(lambda: {
            'items': [],
            'ratings': [],
            'timestamps': [],
            'genres': [],
            'temporal_features': []
        })
        
        # Sort by user and timestamp
        sorted_data = self.data.sort_values(['user_idx', 'timestamp'])
        
        for _, row in sorted_data.iterrows():
            user_idx = row['user_idx']
            movie_idx = row['movie_idx']
            
            # Get movie genres
            movie_row = self.movies_df[self.movies_df['movie_idx'] == movie_idx].iloc[0]
            genre_vector = []
            for genre in self.genres:
                if f'genre_{genre}' in movie_row:
                    genre_vector.append(movie_row[f'genre_{genre}'])
                else:
                    genre_vector.append(0)
            
            # Temporal features
            temporal_features = [
                row['hour_sin'], row['hour_cos'],
                row['dow_0'], row['dow_1'], row['dow_2'], row['dow_3'],
                row['dow_4'], row['dow_5'], row['dow_6']
            ]
            
            histories[user_idx]['items'].append(movie_idx)
            histories[user_idx]['ratings'].append(row['rating'])
            histories[user_idx]['timestamps'].append(row['timestamp'])
            histories[user_idx]['genres'].append(genre_vector)
            histories[user_idx]['temporal_features'].append(temporal_features)
        
        return histories
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single training sample with all required features for Enhanced MARL.
        """
        row = self.data.iloc[idx]
        user_idx = row['user_idx']
        movie_idx = row['movie_idx']
        rating = row['rating']
        implicit_rating = row['implicit_rating']
        
        # User features
        user_features = self._get_user_features(user_idx)
        
        # Item features
        item_features = self._get_item_features(movie_idx)
        
        # User history for ContextGNN and BUHS
        user_history = self._get_user_history(user_idx, exclude_item=movie_idx)
        
        # Genre-specific features for multi-agent RL
        genre_features = self._get_genre_features(movie_idx)
        
        # Temporal context
        temporal_context = self._get_temporal_context(row)
        
        # Negative samples for training
        negative_samples = []
        if self.mode == 'train':
            negative_samples = self._get_negative_samples(user_idx, movie_idx)
        
        sample = {
            # Basic features
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(movie_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float32),
            'implicit_rating': torch.tensor(implicit_rating, dtype=torch.float32),
            
            # Enhanced features
            'user_features': user_features,
            'item_features': item_features,
            'user_history': user_history,
            'genre_features': genre_features,
            'temporal_context': temporal_context,
            
            # Training specific
            'negative_samples': negative_samples,
            
            # Metadata
            'item_popularity': torch.tensor(
                self.metadata['item_popularity'][movie_idx], 
                dtype=torch.float32
            ),
            'timestamp': torch.tensor(row['timestamp'], dtype=torch.long)
        }
        
        return sample
    
    def _get_user_features(self, user_idx: int) -> torch.Tensor:
        """Get comprehensive user features."""
        user_row = self.users_df[self.users_df['user_idx'] == user_idx].iloc[0]
        
        features = [
            user_row['gender_idx'],
            user_row['age_group'],
            user_row['occupation_idx']
        ]
        
        return torch.tensor(features, dtype=torch.long)
    
    def _get_item_features(self, movie_idx: int) -> Dict[str, torch.Tensor]:
        """Get comprehensive item features."""
        movie_row = self.movies_df[self.movies_df['movie_idx'] == movie_idx].iloc[0]
        
        # Genre vector
        genre_vector = []
        for genre in self.genres:
            if f'genre_{genre}' in movie_row:
                genre_vector.append(movie_row[f'genre_{genre}'])
            else:
                genre_vector.append(0)
        
        # SBERT title embedding
        title_embedding = torch.tensor(movie_row['title_embedding'], dtype=torch.float32)
        
        return {
            'genre_vector': torch.tensor(genre_vector, dtype=torch.float32),
            'title_embedding': title_embedding,
            'year': torch.tensor(movie_row['year_normalized'], dtype=torch.float32),
            'movie_idx': torch.tensor(movie_idx, dtype=torch.long)
        }
    
    def _get_user_history(self, user_idx: int, exclude_item: int = None) -> Dict[str, torch.Tensor]:
        """Get user interaction history for ContextGNN and BUHS."""
        if user_idx not in self.user_histories:
            return self._get_empty_history()
        
        history = self.user_histories[user_idx]
        
        # Filter out the current item if specified
        if exclude_item is not None:
            filtered_indices = [i for i, item in enumerate(history['items']) if item != exclude_item]
        else:
            filtered_indices = list(range(len(history['items'])))
        
        # Limit to max sequence length (most recent)
        if len(filtered_indices) > self.max_seq_length:
            filtered_indices = filtered_indices[-self.max_seq_length:]
        
        if not filtered_indices:
            return self._get_empty_history()
        
        # Extract filtered history
        items = [history['items'][i] for i in filtered_indices]
        ratings = [history['ratings'][i] for i in filtered_indices]
        timestamps = [history['timestamps'][i] for i in filtered_indices]
        genres = [history['genres'][i] for i in filtered_indices]
        temporal_features = [history['temporal_features'][i] for i in filtered_indices]
        
        # Get item embeddings for BUHS
        item_embeddings = []
        item_popularities = []
        
        for item in items:
            movie_row = self.movies_df[self.movies_df['movie_idx'] == item].iloc[0]
            item_embeddings.append(movie_row['title_embedding'])
            item_popularities.append(self.metadata['item_popularity'][item])
        
        return {
            'items': torch.tensor(items, dtype=torch.long),
            'ratings': torch.tensor(ratings, dtype=torch.float32),
            'timestamps': torch.tensor(timestamps, dtype=torch.long),
            'genres': torch.tensor(genres, dtype=torch.float32),
            'temporal_features': torch.tensor(temporal_features, dtype=torch.float32),
            'item_embeddings': torch.tensor(item_embeddings, dtype=torch.float32),
            'item_popularities': torch.tensor(item_popularities, dtype=torch.float32),
            'sequence_length': torch.tensor(len(items), dtype=torch.long)
        }
    
    def _get_empty_history(self) -> Dict[str, torch.Tensor]:
        """Get empty history for cold-start users."""
        return {
            'items': torch.zeros(1, dtype=torch.long),
            'ratings': torch.zeros(1, dtype=torch.float32),
            'timestamps': torch.zeros(1, dtype=torch.long),
            'genres': torch.zeros(1, self.num_genres, dtype=torch.float32),
            'temporal_features': torch.zeros(1, 9, dtype=torch.float32),  # 2 (sin, cos) + 7 (dow)
            'item_embeddings': torch.zeros(1, 384, dtype=torch.float32),  # SBERT dimension
            'item_popularities': torch.zeros(1, dtype=torch.float32),
            'sequence_length': torch.tensor(0, dtype=torch.long)
        }
    
    def _get_genre_features(self, movie_idx: int) -> Dict[str, torch.Tensor]:
        """Get genre-specific features for multi-agent RL."""
        movie_row = self.movies_df[self.movies_df['movie_idx'] == movie_idx].iloc[0]
        
        # Primary genre (most relevant for agent selection)
        genre_scores = []
        for genre in self.genres:
            if f'genre_{genre}' in movie_row:
                genre_scores.append(movie_row[f'genre_{genre}'])
            else:
                genre_scores.append(0)
        
        primary_genre_idx = np.argmax(genre_scores) if any(genre_scores) else 0
        
        return {
            'genre_vector': torch.tensor(genre_scores, dtype=torch.float32),
            'primary_genre': torch.tensor(primary_genre_idx, dtype=torch.long),
            'is_multi_genre': torch.tensor(sum(genre_scores) > 1, dtype=torch.bool)
        }
    
    def _get_temporal_context(self, row: pd.Series) -> torch.Tensor:
        """Get temporal context features."""
        temporal_features = [
            row['hour_sin'], row['hour_cos'],
            row['dow_0'], row['dow_1'], row['dow_2'], row['dow_3'],
            row['dow_4'], row['dow_5'], row['dow_6']
        ]
        
        return torch.tensor(temporal_features, dtype=torch.float32)
    
    def _get_negative_samples(self, user_idx: int, positive_item: int) -> torch.Tensor:
        """Get negative samples using fair sampling strategy."""
        # Get user's historical items to avoid sampling
        user_items = set()
        if user_idx in self.user_histories:
            user_items = set(self.user_histories[user_idx]['items'])
        
        # Use fair sampler to get negative items
        negative_items = self.fair_sampler.sample_negatives(
            user_items,
            positive_item,
            self.negative_sampling_ratio
        )
        
        return torch.tensor(negative_items, dtype=torch.long)


class FairNegativeSampler:
    """
    Fair negative sampling to address popularity bias and improve long-tail coverage.
    """
    
    def __init__(self, item_popularity: np.ndarray, config: Dict[str, Any]):
        self.item_popularity = item_popularity
        self.config = config
        self.num_items = len(item_popularity)
        
        # Sampling weights based on inverse popularity
        self.sampling_weights = self._compute_sampling_weights()
        
        logger.info("FairNegativeSampler initialized")
    
    def _compute_sampling_weights(self) -> np.ndarray:
        """Compute sampling weights for fair negative sampling."""
        if self.config['method'] == 'uniform':
            return np.ones(self.num_items)
        
        elif self.config['method'] == 'inverse_frequency':
            # Inverse popularity weighting
            weights = 1.0 / (self.item_popularity + 1e-8)
            weights = weights / weights.sum()
            return weights
        
        elif self.config['method'] == 'genre_aware_inverse_frequency':
            # Enhanced with temperature scaling
            temp = self.config['temperature']
            weights = (1.0 / (self.item_popularity + 1e-8)) ** (1.0 / temp)
            weights = weights / weights.sum()
            return weights
        
        else:
            return np.ones(self.num_items)
    
    def sample_negatives(
        self, 
        user_items: set, 
        positive_item: int, 
        num_samples: int
    ) -> List[int]:
        """Sample negative items for a user."""
        # Items to exclude
        exclude_items = user_items.union({positive_item})
        
        # Available items for sampling
        available_items = [i for i in range(self.num_items) if i not in exclude_items]
        
        if len(available_items) < num_samples:
            # If not enough items, sample with replacement
            return random.choices(available_items, k=num_samples)
        
        # Sample based on weights
        available_weights = [self.sampling_weights[i] for i in available_items]
        available_weights = np.array(available_weights)
        available_weights = available_weights / available_weights.sum()
        
        sampled_items = np.random.choice(
            available_items,
            size=num_samples,
            replace=False,
            p=available_weights
        )
        
        return sampled_items.tolist()


class MARLBatchSampler(Sampler):
    """
    Custom batch sampler for MARL training with genre balance and fairness considerations.
    """
    
    def __init__(self, dataset: EnhancedMARLDataset, batch_size: int, config: Dict[str, Any]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = config
        
        # Group samples by genre for balanced sampling
        self.genre_groups = self._group_by_genre()
        
        logger.info("MARLBatchSampler initialized")
    
    def _group_by_genre(self) -> Dict[int, List[int]]:
        """Group samples by primary genre."""
        genre_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            row = self.dataset.data.iloc[idx]
            movie_idx = row['movie_idx']
            
            # Get primary genre
            movie_row = self.dataset.movies_df[
                self.dataset.movies_df['movie_idx'] == movie_idx
            ].iloc[0]
            
            genre_scores = []
            for genre in self.dataset.genres:
                if f'genre_{genre}' in movie_row:
                    genre_scores.append(movie_row[f'genre_{genre}'])
                else:
                    genre_scores.append(0)
            
            primary_genre = np.argmax(genre_scores) if any(genre_scores) else 0
            genre_groups[primary_genre].append(idx)
        
        return genre_groups
    
    def __iter__(self):
        """Generate batches with genre balance."""
        all_indices = list(range(len(self.dataset)))
        random.shuffle(all_indices)
        
        for i in range(0, len(all_indices), self.batch_size):
            batch_indices = all_indices[i:i + self.batch_size]
            yield batch_indices
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for Enhanced MARL batches.
    """
    collated = {}
    
    # Handle tensor fields
    tensor_fields = [
        'user_idx', 'item_idx', 'rating', 'implicit_rating',
        'user_features', 'temporal_context', 'item_popularity', 'timestamp'
    ]
    
    for field in tensor_fields:
        if field in batch[0]:
            collated[field] = torch.stack([item[field] for item in batch])
    
    # Handle item features (dict of tensors)
    if 'item_features' in batch[0]:
        collated['item_features'] = {}
        for key in batch[0]['item_features'].keys():
            collated['item_features'][key] = torch.stack([
                item['item_features'][key] for item in batch
            ])
    
    # Handle genre features (dict of tensors)
    if 'genre_features' in batch[0]:
        collated['genre_features'] = {}
        for key in batch[0]['genre_features'].keys():
            collated['genre_features'][key] = torch.stack([
                item['genre_features'][key] for item in batch
            ])
    
    # Handle user history (variable length sequences)
    if 'user_history' in batch[0]:
        collated['user_history'] = {}
        
        # Get max sequence length in batch
        max_seq_len = max(item['user_history']['sequence_length'].item() for item in batch)
        max_seq_len = max(max_seq_len, 1)  # At least 1
        
        for key in batch[0]['user_history'].keys():
            if key == 'sequence_length':
                collated['user_history'][key] = torch.stack([
                    item['user_history'][key] for item in batch
                ])
                continue
            
            # Pad sequences to max length
            padded_seqs = []
            for item in batch:
                seq = item['user_history'][key]
                if seq.dim() == 1:
                    # 1D sequence (e.g., items, ratings)
                    if len(seq) < max_seq_len:
                        pad_size = max_seq_len - len(seq)
                        seq = F.pad(seq, (0, pad_size), value=0)
                    padded_seqs.append(seq[:max_seq_len])
                else:
                    # 2D sequence (e.g., genres, embeddings)
                    if seq.size(0) < max_seq_len:
                        pad_size = max_seq_len - seq.size(0)
                        seq = F.pad(seq, (0, 0, 0, pad_size), value=0)
                    padded_seqs.append(seq[:max_seq_len])
            
            collated['user_history'][key] = torch.stack(padded_seqs)
    
    # Handle negative samples
    if 'negative_samples' in batch[0]:
        collated['negative_samples'] = torch.stack([
            item['negative_samples'] for item in batch
        ])
    
    return collated


class EnhancedMARLDataLoader:
    """
    Main DataLoader class for Enhanced MARL Two-Tower Recommendation System.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize data processor
        self.processor = MovieLensDataProcessor(
            config['dataset']['path'], 
            config
        )
        
        # Load and preprocess data
        self.processed_data = self.processor.load_and_preprocess()
        
        # Create datasets
        self.datasets = self._create_datasets()
        
        logger.info("EnhancedMARLDataLoader initialized successfully")
    
    def _create_datasets(self) -> Dict[str, EnhancedMARLDataset]:
        """Create train/val/test datasets."""
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            if split in self.processed_data['splits'] and len(self.processed_data['splits'][split]) > 0:
                datasets[split] = EnhancedMARLDataset(
                    data=self.processed_data['splits'][split],
                    users_df=self.processed_data['users'],
                    movies_df=self.processed_data['movies'],
                    metadata=self.processed_data['metadata'],
                    config=self.config,
                    mode=split
                )
        
        return datasets
    
    def get_dataloader(
        self, 
        split: str, 
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: int = 4
    ) -> DataLoader:
        """Get DataLoader for specified split."""
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available")
        
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        dataset = self.datasets[split]
        
        # Use custom batch sampler for training
        if split == 'train' and self.config.get('use_balanced_sampling', True):
            batch_sampler = MARLBatchSampler(dataset, batch_size, self.config)
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True
            )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return self.processed_data['metadata']
    
    def save_processed_data(self, filepath: str):
        """Save processed data for future use."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.processed_data, f)
        logger.info(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str):
        """Load previously processed data."""
        with open(filepath, 'rb') as f:
            self.processed_data = pickle.load(f)
        
        # Recreate datasets
        self.datasets = self._create_datasets()
        logger.info(f"Processed data loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'dataset': {
            'path': 'data/ml-1m',
            'num_users': 6040,
            'num_items': 3706,
            'num_genres': 18,
            'genres': [
                "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western", "Documentary"
            ],
            'max_sequence_length': 50,
            'negative_sampling_ratio': 4
        },
        'training': {
            'batch_size': 256,
            'num_workers': 4
        },
        'fair_sampling': {
            'enabled': True,
            'method': 'genre_aware_inverse_frequency',
            'temperature': 0.8
        },
        'device': 'cuda'
    }
    
    # Initialize dataloader
    dataloader = EnhancedMARLDataLoader(config)
    
    # Get training dataloader
    train_loader = dataloader.get_dataloader('train')
    
    # Test batch processing
    for batch in train_loader:
        print("Batch keys:", list(batch.keys()))
        print("User features shape:", batch['user_features'].shape)
        print("Item features genre vector shape:", batch['item_features']['genre_vector'].shape)
        print("User history items shape:", batch['user_history']['items'].shape)
        print("Negative samples shape:", batch['negative_samples'].shape)
        break
    
    logger.info("DataLoader testing completed successfully")

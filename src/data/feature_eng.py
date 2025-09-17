"""
Enhanced MARL Two-Tower Recommendation System - Feature Engineering

Comprehensive feature engineering utilities supporting:
- ContextGNN user modeling with graph-based features
- Multi-agent RL feature preparation
- SBERT embeddings and text processing
- Temporal and demographic feature engineering
- BUHS-compatible user history synthesis
- Fair sampling feature support
- Genre-aware item encoding
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import logging
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

# NLP and embeddings
from sentence_transformers import SentenceTransformer
import transformers

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Graph processing
import networkx as nx
from torch_geometric.utils import to_undirected, add_self_loops

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """
    Comprehensive Feature Engineering Pipeline for Enhanced MARL Two-Tower System.
    
    Features:
    - User demographic and behavioral feature engineering
    - Item content and metadata processing with SBERT
    - Temporal feature extraction and encoding
    - Graph-based feature construction for ContextGNN
    - Genre-aware feature preparation for multi-agent RL
    - BUHS-compatible user history synthesis features
    - Fair sampling support features
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize encoders
        self.encoders = {
            'user': LabelEncoder(),
            'item': LabelEncoder(),
            'gender': LabelEncoder(),
            'occupation': LabelEncoder()
        }
        
        # Initialize scalers
        self.scalers = {
            'age': MinMaxScaler(),
            'year': MinMaxScaler(),
            'popularity': MinMaxScaler()
        }
        
        # SBERT model for text embeddings
        self.text_encoder = None
        self.tfidf_vectorizer = None
        
        # Feature statistics
        self.feature_stats = {}
        
        # Cache for expensive computations
        self.cache = {}
        
        logger.info("FeatureEngineeringPipeline initialized")
    
    def initialize_text_encoder(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize SBERT text encoder"""
        try:
            self.text_encoder = SentenceTransformer(model_name)
            logger.info(f"SBERT model '{model_name}' loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.text_encoder = None
    
    def process_user_features(self, users_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process user demographic and behavioral features.
        
        Args:
            users_df: Raw users dataframe
            
        Returns:
            Dictionary containing processed user features and metadata
        """
        logger.info("Processing user features...")
        
        processed_users = users_df.copy()
        
        # Basic ID encoding
        processed_users['user_idx'] = self.encoders['user'].fit_transform(processed_users['user_id'])
        
        # Gender encoding
        processed_users['gender_idx'] = self.encoders['gender'].fit_transform(processed_users['gender'])
        processed_users['gender_onehot'] = pd.get_dummies(processed_users['gender'], prefix='gender')
        
        # Age processing
        processed_users['age_normalized'] = self.scalers['age'].fit_transform(
            processed_users['age'].values.reshape(-1, 1)
        ).flatten()
        
        # Age group binning (MovieLens specific)
        age_bins = [0, 18, 25, 35, 45, 50, 56, 100]
        processed_users['age_group'] = pd.cut(
            processed_users['age'], 
            bins=age_bins, 
            labels=range(len(age_bins)-1),
            include_lowest=True
        ).astype(int)
        
        # Occupation encoding
        processed_users['occupation_idx'] = self.encoders['occupation'].fit_transform(
            processed_users['occupation']
        )
        processed_users['occupation_onehot'] = pd.get_dummies(
            processed_users['occupation'], prefix='occupation'
        )
        
        # Zip code processing (extract geographic features)
        processed_users['zip_region'] = processed_users['zip_code'].astype(str).str[:2]
        processed_users['zip_region_idx'] = LabelEncoder().fit_transform(processed_users['zip_region'])
        
        # Create user demographic embeddings
        demographic_features = self._create_demographic_embeddings(processed_users)
        
        # Store feature statistics
        self.feature_stats['users'] = {
            'num_users': len(processed_users),
            'age_range': (processed_users['age'].min(), processed_users['age'].max()),
            'num_occupations': processed_users['occupation'].nunique(),
            'num_genders': processed_users['gender'].nunique(),
            'num_zip_regions': processed_users['zip_region'].nunique()
        }
        
        logger.info(f"Processed {len(processed_users)} user features")
        
        return {
            'users_df': processed_users,
            'demographic_embeddings': demographic_features,
            'user_encoders': {k: v for k, v in self.encoders.items() if k in ['user', 'gender', 'occupation']},
            'user_scalers': {k: v for k, v in self.scalers.items() if k == 'age'},
            'metadata': self.feature_stats['users']
        }
    
    def process_item_features(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process item content and metadata features.
        
        Args:
            movies_df: Raw movies dataframe
            
        Returns:
            Dictionary containing processed item features and metadata
        """
        logger.info("Processing item features...")
        
        processed_movies = movies_df.copy()
        
        # Basic ID encoding
        processed_movies['item_idx'] = self.encoders['item'].fit_transform(processed_movies['movie_id'])
        
        # Extract and process release year
        processed_movies['year'] = processed_movies['title'].str.extract(r'\((\d{4})\)$')[0]
        processed_movies['year'] = pd.to_numeric(processed_movies['year'], errors='coerce')
        processed_movies['year'].fillna(processed_movies['year'].median(), inplace=True)
        
        # Normalize year
        processed_movies['year_normalized'] = self.scalers['year'].fit_transform(
            processed_movies['year'].values.reshape(-1, 1)
        ).flatten()
        
        # Year binning (decades)
        processed_movies['decade'] = (processed_movies['year'] // 10) * 10
        processed_movies['decade_idx'] = LabelEncoder().fit_transform(processed_movies['decade'])
        
        # Process genres
        genre_features = self._process_genre_features(processed_movies)
        processed_movies.update(genre_features['genre_df'])
        
        # Process titles
        title_features = self._process_title_features(processed_movies)
        processed_movies.update(title_features['title_df'])
        
        # Create item content embeddings
        content_embeddings = self._create_item_content_embeddings(processed_movies, genre_features, title_features)
        
        # Store feature statistics
        self.feature_stats['items'] = {
            'num_items': len(processed_movies),
            'year_range': (processed_movies['year'].min(), processed_movies['year'].max()),
            'num_genres': len(genre_features['genre_list']),
            'genres': genre_features['genre_list'],
            'avg_genres_per_item': genre_features['genre_matrix'].sum(axis=1).mean()
        }
        
        logger.info(f"Processed {len(processed_movies)} item features")
        
        return {
            'movies_df': processed_movies,
            'genre_features': genre_features,
            'title_features': title_features,
            'content_embeddings': content_embeddings,
            'item_encoders': {k: v for k, v in self.encoders.items() if k == 'item'},
            'item_scalers': {k: v for k, v in self.scalers.items() if k == 'year'},
            'metadata': self.feature_stats['items']
        }
    
    def process_interaction_features(self, interactions_df: pd.DataFrame,
                                   users_df: pd.DataFrame,
                                   movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process user-item interaction features and create derived features.
        
        Args:
            interactions_df: Raw interactions dataframe
            users_df: Processed users dataframe
            movies_df: Processed movies dataframe
            
        Returns:
            Dictionary containing processed interaction features
        """
        logger.info("Processing interaction features...")
        
        processed_interactions = interactions_df.copy()
        
        # Map to encoded indices
        user_mapping = dict(zip(users_df['user_id'], users_df['user_idx']))
        item_mapping = dict(zip(movies_df['movie_id'], movies_df['item_idx']))
        
        processed_interactions['user_idx'] = processed_interactions['user_id'].map(user_mapping)
        processed_interactions['item_idx'] = processed_interactions['movie_id'].map(item_mapping)
        
        # Convert to implicit feedback
        implicit_threshold = self.config.get('implicit_threshold', 4)
        processed_interactions['implicit_rating'] = (
            processed_interactions['rating'] >= implicit_threshold
        ).astype(int)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(processed_interactions)
        processed_interactions.update(temporal_features)
        
        # User behavior features
        user_behavior = self._extract_user_behavior_features(processed_interactions)
        
        # Item popularity features
        item_popularity = self._extract_item_popularity_features(processed_interactions)
        
        # Interaction context features
        context_features = self._extract_interaction_context_features(
            processed_interactions, users_df, movies_df
        )
        
        # Create interaction graph for ContextGNN
        interaction_graph = self._create_interaction_graph(processed_interactions)
        
        # Store feature statistics
        self.feature_stats['interactions'] = {
            'num_interactions': len(processed_interactions),
            'sparsity': 1 - (len(processed_interactions) / (len(users_df) * len(movies_df))),
            'implicit_positive_ratio': processed_interactions['implicit_rating'].mean(),
            'rating_distribution': processed_interactions['rating'].value_counts().to_dict(),
            'temporal_span': (
                processed_interactions['timestamp'].min(),
                processed_interactions['timestamp'].max()
            )
        }
        
        logger.info(f"Processed {len(processed_interactions)} interactions")
        
        return {
            'interactions_df': processed_interactions,
            'user_behavior': user_behavior,
            'item_popularity': item_popularity,
            'context_features': context_features,
            'interaction_graph': interaction_graph,
            'temporal_features': temporal_features,
            'metadata': self.feature_stats['interactions']
        }
    
    def create_buhs_features(self, user_histories: Dict[int, List[int]], 
                           item_popularity: np.ndarray) -> Dict[str, Any]:
        """
        Create BUHS-compatible features for user history synthesis.
        
        Args:
            user_histories: User interaction histories
            item_popularity: Item popularity scores
            
        Returns:
            BUHS feature dictionary
        """
        logger.info("Creating BUHS features...")
        
        buhs_features = {}
        
        # Popularity-based sampling weights
        inv_popularity = 1.0 / (item_popularity + 1e-8)
        buhs_features['sampling_weights'] = inv_popularity / inv_popularity.sum()
        
        # User history statistics
        history_stats = {}
        for user_id, history in user_histories.items():
            if len(history) > 0:
                hist_popularity = item_popularity[history]
                history_stats[user_id] = {
                    'avg_popularity': hist_popularity.mean(),
                    'popularity_std': hist_popularity.std(),
                    'tail_ratio': (hist_popularity < 0.2).mean(),  # Bottom 20%
                    'head_ratio': (hist_popularity > 0.8).mean(),  # Top 20%
                    'diversity_score': len(set(history)) / len(history)
                }
        
        buhs_features['history_stats'] = history_stats
        
        # Tail item promotion factors
        tail_threshold = self.config.get('tail_threshold', 0.2)
        buhs_features['tail_mask'] = item_popularity < tail_threshold
        buhs_features['head_mask'] = item_popularity > (1 - tail_threshold)
        
        logger.info("BUHS features created successfully")
        
        return buhs_features
    
    def create_contextgnn_features(self, processed_interactions: pd.DataFrame,
                                 users_df: pd.DataFrame,
                                 movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create features specifically for ContextGNN user tower.
        
        Args:
            processed_interactions: Processed interactions
            users_df: Processed users dataframe
            movies_df: Processed movies dataframe
            
        Returns:
            ContextGNN feature dictionary
        """
        logger.info("Creating ContextGNN features...")
        
        contextgnn_features = {}
        
        # User interaction sequences
        user_sequences = self._create_user_sequences(processed_interactions)
        contextgnn_features['user_sequences'] = user_sequences
        
        # Temporal adjacency matrices
        temporal_graphs = self._create_temporal_adjacency_matrices(processed_interactions)
        contextgnn_features['temporal_graphs'] = temporal_graphs
        
        # User-item interaction embeddings
        interaction_embeddings = self._create_interaction_embeddings(
            processed_interactions, users_df, movies_df
        )
        contextgnn_features['interaction_embeddings'] = interaction_embeddings
        
        # Graph construction parameters
        contextgnn_features['graph_params'] = {
            'k_neighbors': self.config.get('k_neighbors', 20),
            'edge_threshold': self.config.get('edge_threshold', 0.1),
            'temporal_decay': self.config.get('temporal_decay', 0.95)
        }
        
        logger.info("ContextGNN features created successfully")
        
        return contextgnn_features
    
    def create_marl_features(self, processed_interactions: pd.DataFrame,
                           movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create features for multi-agent RL system.
        
        Args:
            processed_interactions: Processed interactions
            movies_df: Processed movies dataframe
            
        Returns:
            MARL feature dictionary
        """
        logger.info("Creating MARL features...")
        
        marl_features = {}
        
        # Genre-specific interaction data
        genre_interactions = self._create_genre_interaction_data(processed_interactions, movies_df)
        marl_features['genre_interactions'] = genre_interactions
        
        # Agent state features
        agent_features = self._create_agent_state_features(processed_interactions, movies_df)
        marl_features['agent_features'] = agent_features
        
        # Exposure tracking features
        exposure_features = self._create_exposure_tracking_features(processed_interactions, movies_df)
        marl_features['exposure_features'] = exposure_features
        
        # Reward computation features
        reward_features = self._create_reward_features(processed_interactions, movies_df)
        marl_features['reward_features'] = reward_features
        
        logger.info("MARL features created successfully")
        
        return marl_features
    
    def _create_demographic_embeddings(self, users_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Create demographic embedding features"""
        embeddings = {}
        
        # Gender embeddings
        embeddings['gender'] = torch.tensor(
            pd.get_dummies(users_df['gender']).values, dtype=torch.float32
        )
        
        # Age group embeddings
        embeddings['age_group'] = torch.tensor(
            pd.get_dummies(users_df['age_group']).values, dtype=torch.float32
        )
        
        # Occupation embeddings
        embeddings['occupation'] = torch.tensor(
            pd.get_dummies(users_df['occupation']).values, dtype=torch.float32
        )
        
        return embeddings
    
    def _process_genre_features(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Process movie genre features"""
        genre_features = {}
        
        # Extract all unique genres
        all_genres = set()
        for genres_str in movies_df['genres']:
            all_genres.update(genres_str.split('|'))
        
        genre_list = sorted(list(all_genres))
        genre_features['genre_list'] = genre_list
        
        # Create binary genre matrix
        genre_matrix = np.zeros((len(movies_df), len(genre_list)))
        genre_df_cols = {}
        
        for i, genre in enumerate(genre_list):
            genre_col = movies_df['genres'].str.contains(genre).astype(int)
            genre_matrix[:, i] = genre_col.values
            genre_df_cols[f'genre_{genre.lower()}'] = genre_col
        
        genre_features['genre_matrix'] = genre_matrix
        genre_features['genre_df'] = genre_df_cols
        
        # Genre statistics
        genre_features['genre_counts'] = {
            genre: genre_matrix[:, i].sum() for i, genre in enumerate(genre_list)
        }
        
        # Multi-genre indicators
        genre_features['num_genres_per_item'] = genre_matrix.sum(axis=1)
        genre_features['is_single_genre'] = (genre_matrix.sum(axis=1) == 1)
        genre_features['is_multi_genre'] = (genre_matrix.sum(axis=1) > 1)
        
        return genre_features
    
    def _process_title_features(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Process movie title features"""
        title_features = {}
        
        # Clean titles (remove year)
        clean_titles = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
        title_features['clean_titles'] = clean_titles
        
        # Title length features
        title_features['title_length'] = clean_titles.str.len()
        title_features['title_word_count'] = clean_titles.str.split().str.len()
        
        # SBERT embeddings if available
        if self.text_encoder is not None:
            logger.info("Generating SBERT embeddings for movie titles...")
            title_embeddings = self.text_encoder.encode(
                clean_titles.tolist(), 
                show_progress_bar=True,
                batch_size=32
            )
            title_features['sbert_embeddings'] = title_embeddings
        else:
            # Fallback to TF-IDF
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            
            tfidf_embeddings = self.tfidf_vectorizer.fit_transform(clean_titles).toarray()
            title_features['tfidf_embeddings'] = tfidf_embeddings
        
        # Title DataFrame columns
        title_df = {
            'clean_title': clean_titles,
            'title_length': title_features['title_length'],
            'title_word_count': title_features['title_word_count']
        }
        
        title_features['title_df'] = title_df
        
        return title_features
    
    def _create_item_content_embeddings(self, movies_df: pd.DataFrame,
                                      genre_features: Dict[str, Any],
                                      title_features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Create comprehensive item content embeddings"""
        embeddings = {}
        
        # Genre embeddings
        embeddings['genre_binary'] = torch.tensor(
            genre_features['genre_matrix'], dtype=torch.float32
        )
        
        # Year embeddings
        embeddings['year_normalized'] = torch.tensor(
            movies_df['year_normalized'].values, dtype=torch.float32
        ).unsqueeze(-1)
        
        # Title embeddings
        if 'sbert_embeddings' in title_features:
            embeddings['title_sbert'] = torch.tensor(
                title_features['sbert_embeddings'], dtype=torch.float32
            )
        else:
            embeddings['title_tfidf'] = torch.tensor(
                title_features['tfidf_embeddings'], dtype=torch.float32
            )
        
        # Content diversity features
        embeddings['content_diversity'] = torch.tensor([
            genre_features['num_genres_per_item'],
            title_features['title_length'],
            title_features['title_word_count']
        ]).T.float()
        
        return embeddings
    
    def _extract_temporal_features(self, interactions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract temporal features from interactions"""
        temporal_features = {}
        
        # Convert timestamp to datetime
        interactions_df['datetime'] = pd.to_datetime(interactions_df['timestamp'], unit='s')
        
        # Basic temporal components
        temporal_features['hour'] = interactions_df['datetime'].dt.hour
        temporal_features['day_of_week'] = interactions_df['datetime'].dt.dayofweek
        temporal_features['month'] = interactions_df['datetime'].dt.month
        temporal_features['year'] = interactions_df['datetime'].dt.year
        
        # Sinusoidal encodings for cyclic features
        temporal_features['hour_sin'] = np.sin(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * temporal_features['hour'] / 24)
        
        temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['month'] / 12)
        
        temporal_features['dow_sin'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['dow_cos'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        
        # One-hot encodings
        for i in range(7):
            temporal_features[f'dow_{i}'] = (temporal_features['day_of_week'] == i).astype(int)
        
        for i in range(24):
            temporal_features[f'hour_{i}'] = (temporal_features['hour'] == i).astype(int)
        
        return temporal_features
    
    def _extract_user_behavior_features(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract user behavioral features"""
        user_behavior = {}
        
        # User interaction statistics
        user_stats = interactions_df.groupby('user_idx').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'implicit_rating': 'mean',
            'timestamp': ['min', 'max']
        }).round(3)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_behavior['user_stats'] = user_stats
        
        # User activity patterns
        user_temporal = interactions_df.groupby(['user_idx', 'hour']).size().unstack(fill_value=0)
        user_behavior['hourly_activity'] = user_temporal
        
        # User genre preferences
        user_genre_counts = interactions_df.groupby('user_idx')['item_idx'].apply(list)
        user_behavior['genre_preferences'] = user_genre_counts
        
        # User interaction sequences (for ContextGNN)
        user_sequences = interactions_df.sort_values(['user_idx', 'timestamp']).groupby('user_idx').apply(
            lambda x: {
                'items': x['item_idx'].tolist(),
                'ratings': x['rating'].tolist(),
                'timestamps': x['timestamp'].tolist(),
                'temporal_features': list(zip(x['hour_sin'], x['hour_cos']))
            }
        ).to_dict()
        
        user_behavior['interaction_sequences'] = user_sequences
        
        return user_behavior
    
    def _extract_item_popularity_features(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract item popularity and fairness features"""
        item_popularity = {}
        
        # Basic popularity metrics
        item_counts = interactions_df['item_idx'].value_counts()
        item_popularity['interaction_counts'] = item_counts
        
        # Normalize popularity
        normalized_popularity = item_counts / item_counts.max()
        item_popularity['normalized_popularity'] = normalized_popularity
        
        # Popularity percentiles
        item_popularity['popularity_percentiles'] = normalized_popularity.rank(pct=True)
        
        # Long-tail classification
        tail_threshold = self.config.get('tail_threshold', 0.2)
        head_threshold = self.config.get('head_threshold', 0.8)
        
        item_popularity['tail_items'] = normalized_popularity < tail_threshold
        item_popularity['head_items'] = normalized_popularity > head_threshold
        item_popularity['medium_items'] = ~(item_popularity['tail_items'] | item_popularity['head_items'])
        
        # GINI coefficient computation helpers
        sorted_popularity = np.sort(normalized_popularity.values)
        n = len(sorted_popularity)
        index = np.arange(1, n + 1)
        gini_coef = (2 * np.sum(index * sorted_popularity)) / (n * np.sum(sorted_popularity)) - (n + 1) / n
        
        item_popularity['gini_coefficient'] = gini_coef
        
        return item_popularity
    
    def _extract_interaction_context_features(self, interactions_df: pd.DataFrame,
                                            users_df: pd.DataFrame,
                                            movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract contextual features from interactions"""
        context_features = {}
        
        # User-item context matrices
        user_item_matrix = interactions_df.pivot_table(
            index='user_idx', 
            columns='item_idx', 
            values='rating', 
            fill_value=0
        )
        context_features['user_item_matrix'] = user_item_matrix
        
        # Implicit feedback matrix
        implicit_matrix = interactions_df.pivot_table(
            index='user_idx',
            columns='item_idx',
            values='implicit_rating',
            fill_value=0
        )
        context_features['implicit_matrix'] = implicit_matrix
        
        # Co-occurrence matrices
        user_cooccurrence = np.dot(user_item_matrix.T, user_item_matrix)
        item_cooccurrence = np.dot(user_item_matrix, user_item_matrix.T)
        
        context_features['user_cooccurrence'] = user_cooccurrence
        context_features['item_cooccurrence'] = item_cooccurrence
        
        return context_features
    
    def _create_interaction_graph(self, interactions_df: pd.DataFrame) -> nx.Graph:
        """Create interaction graph for ContextGNN"""
        graph = nx.Graph()
        
        # Add nodes
        users = interactions_df['user_idx'].unique()
        items = interactions_df['item_idx'].unique()
        
        for user in users:
            graph.add_node(f"u_{user}", type='user')
        
        for item in items:
            graph.add_node(f"i_{item}", type='item')
        
        # Add edges (only positive interactions)
        positive_interactions = interactions_df[interactions_df['implicit_rating'] == 1]
        
        for _, row in positive_interactions.iterrows():
            graph.add_edge(
                f"u_{row['user_idx']}", 
                f"i_{row['item_idx']}",
                weight=row['rating'],
                timestamp=row['timestamp']
            )
        
        return graph
    
    def _create_user_sequences(self, interactions_df: pd.DataFrame) -> Dict[int, Dict[str, List]]:
        """Create user interaction sequences for ContextGNN"""
        sequences = {}
        
        # Sort by user and timestamp
        sorted_interactions = interactions_df.sort_values(['user_idx', 'timestamp'])
        
        for user_idx, group in sorted_interactions.groupby('user_idx'):
            sequences[user_idx] = {
                'items': group['item_idx'].tolist(),
                'ratings': group['rating'].tolist(),
                'timestamps': group['timestamp'].tolist(),
                'temporal_sin_cos': list(zip(group['hour_sin'], group['hour_cos'])),
                'day_of_week_onehot': [
                    [int(group[f'dow_{i}'].iloc[j]) for i in range(7)] 
                    for j in range(len(group))
                ]
            }
        
        return sequences
    
    def _create_temporal_adjacency_matrices(self, interactions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create temporal adjacency matrices"""
        temporal_graphs = {}
        
        # Time-based user similarity
        user_time_features = interactions_df.groupby('user_idx').agg({
            'hour_sin': 'mean',
            'hour_cos': 'mean',
            'dow_sin': 'mean',
            'dow_cos': 'mean'
        })
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        temporal_similarity = cosine_similarity(user_time_features.values)
        
        temporal_graphs['user_temporal_similarity'] = temporal_similarity
        
        return temporal_graphs
    
    def _create_interaction_embeddings(self, interactions_df: pd.DataFrame,
                                     users_df: pd.DataFrame,
                                     movies_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Create interaction embeddings"""
        embeddings = {}
        
        # User-item interaction features
        interaction_features = []
        
        for _, row in interactions_df.iterrows():
            features = [
                row['rating'],
                row['implicit_rating'],
                row['hour_sin'],
                row['hour_cos'],
                row['dow_sin'],
                row['dow_cos']
            ]
            interaction_features.append(features)
        
        embeddings['interaction_features'] = torch.tensor(
            interaction_features, dtype=torch.float32
        )
        
        return embeddings
    
    def _create_genre_interaction_data(self, interactions_df: pd.DataFrame,
                                     movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Create genre-specific interaction data for MARL"""
        genre_data = {}
        
        # Map items to genres
        item_to_genres = {}
        for _, movie in movies_df.iterrows():
            item_idx = movie['item_idx']
            genres = [col.replace('genre_', '') for col in movies_df.columns 
                     if col.startswith('genre_') and movie[col] == 1]
            item_to_genres[item_idx] = genres
        
        # Create genre-specific interaction datasets
        for genre in self.feature_stats['items']['genres']:
            genre_items = [item for item, genres in item_to_genres.items() if genre in genres]
            genre_interactions = interactions_df[interactions_df['item_idx'].isin(genre_items)]
            
            genre_data[f'{genre.lower()}_interactions'] = genre_interactions
            genre_data[f'{genre.lower()}_items'] = genre_items
        
        return genre_data
    
    def _create_agent_state_features(self, interactions_df: pd.DataFrame,
                                   movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Create features for agent state representation"""
        agent_features = {}
        
        # User genre preferences
        user_genre_preferences = {}
        for user_idx in interactions_df['user_idx'].unique():
            user_interactions = interactions_df[interactions_df['user_idx'] == user_idx]
            genre_counts = defaultdict(int)
            
            for _, interaction in user_interactions.iterrows():
                item_idx = interaction['item_idx']
                movie = movies_df[movies_df['item_idx'] == item_idx].iloc[0]
                
                for col in movies_df.columns:
                    if col.startswith('genre_') and movie[col] == 1:
                        genre = col.replace('genre_', '')
                        genre_counts[genre] += 1
            
            user_genre_preferences[user_idx] = dict(genre_counts)
        
        agent_features['user_genre_preferences'] = user_genre_preferences
        
        return agent_features
    
    def _create_exposure_tracking_features(self, interactions_df: pd.DataFrame,
                                         movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Create features for exposure tracking"""
        exposure_features = {}
        
        # Item exposure counts
        item_exposures = interactions_df['item_idx'].value_counts()
        exposure_features['item_exposures'] = item_exposures
        
        # Genre exposure distribution
        genre_exposures = defaultdict(int)
        for _, interaction in interactions_df.iterrows():
            item_idx = interaction['item_idx']
            movie = movies_df[movies_df['item_idx'] == item_idx].iloc[0]
            
            for col in movies_df.columns:
                if col.startswith('genre_') and movie[col] == 1:
                    genre = col.replace('genre_', '')
                    genre_exposures[genre] += 1
        
        exposure_features['genre_exposures'] = dict(genre_exposures)
        
        return exposure_features
    
    def _create_reward_features(self, interactions_df: pd.DataFrame,
                               movies_df: pd.DataFrame) -> Dict[str, Any]:
        """Create features for reward computation"""
        reward_features = {}
        
        # Hit rate features
        positive_interactions = interactions_df[interactions_df['implicit_rating'] == 1]
        reward_features['positive_interactions'] = positive_interactions
        
        # NDCG computation helpers
        user_rankings = {}
        for user_idx in interactions_df['user_idx'].unique():
            user_data = interactions_df[interactions_df['user_idx'] == user_idx]
            sorted_items = user_data.sort_values('rating', ascending=False)['item_idx'].tolist()
            user_rankings[user_idx] = sorted_items
        
        reward_features['user_rankings'] = user_rankings
        
        return reward_features
    
    def save_features(self, features_dict: Dict[str, Any], filepath: str):
        """Save processed features to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(features_dict, f)
        
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, filepath: str) -> Dict[str, Any]:
        """Load processed features from disk"""
        with open(filepath, 'rb') as f:
            features_dict = pickle.load(f)
        
        logger.info(f"Features loaded from {filepath}")
        return features_dict
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feature statistics"""
        return {
            'feature_stats': self.feature_stats,
            'encoders_info': {k: len(v.classes_) for k, v in self.encoders.items() if hasattr(v, 'classes_')},
            'cache_size': len(self.cache)
        }


class FeatureEngineeringManager:
    """
    Manager class for coordinating all feature engineering processes.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.pipeline = FeatureEngineeringPipeline(config, device)
        
        # Initialize text encoder
        model_name = config.get('text_embedding_model', 'all-MiniLM-L6-v2')
        self.pipeline.initialize_text_encoder(model_name)
    
    def process_all_features(self, users_df: pd.DataFrame, 
                           movies_df: pd.DataFrame, 
                           interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process all features in the correct order.
        
        Returns:
            Comprehensive feature dictionary
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Process users
        user_features = self.pipeline.process_user_features(users_df)
        
        # Process items
        item_features = self.pipeline.process_item_features(movies_df)
        
        # Process interactions
        interaction_features = self.pipeline.process_interaction_features(
            interactions_df, user_features['users_df'], item_features['movies_df']
        )
        
        # Create specialized features
        buhs_features = self.pipeline.create_buhs_features(
            interaction_features['user_behavior']['interaction_sequences'],
            interaction_features['item_popularity']['normalized_popularity'].values
        )
        
        contextgnn_features = self.pipeline.create_contextgnn_features(
            interaction_features['interactions_df'],
            user_features['users_df'],
            item_features['movies_df']
        )
        
        marl_features = self.pipeline.create_marl_features(
            interaction_features['interactions_df'],
            item_features['movies_df']
        )
        
        # Combine all features
        all_features = {
            'user_features': user_features,
            'item_features': item_features,
            'interaction_features': interaction_features,
            'buhs_features': buhs_features,
            'contextgnn_features': contextgnn_features,
            'marl_features': marl_features,
            'feature_statistics': self.pipeline.get_feature_statistics()
        }
        
        logger.info("Feature engineering completed successfully")
        
        return all_features
    
    def save_all_features(self, features_dict: Dict[str, Any], base_path: str):
        """Save all features to organized directory structure"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual feature sets
        for feature_type, features in features_dict.items():
            if feature_type != 'feature_statistics':
                filepath = base_path / f"{feature_type}.pkl"
                self.pipeline.save_features(features, str(filepath))
        
        # Save statistics separately
        stats_path = base_path / "feature_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(features_dict['feature_statistics'], f, indent=2)
        
        logger.info(f"All features saved to {base_path}")


# Factory function for easy integration
def create_feature_engineering_manager(config: Dict[str, Any], device: str = 'cuda') -> FeatureEngineeringManager:
    """
    Factory function to create FeatureEngineeringManager.
    
    Args:
        config: Configuration dictionary
        device: Device for computation
        
    Returns:
        Configured FeatureEngineeringManager instance
    """
    return FeatureEngineeringManager(config, device)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'text_embedding_model': 'all-MiniLM-L6-v2',
        'implicit_threshold': 4,
        'tail_threshold': 0.2,
        'head_threshold': 0.8,
        'k_neighbors': 20,
        'edge_threshold': 0.1,
        'temporal_decay': 0.95,
        'year_range': (1919, 2000)
    }
    
    # Initialize feature engineering
    fe_manager = create_feature_engineering_manager(config)
    
    logger.info("Feature engineering module ready for use")

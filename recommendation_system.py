# recommendation_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import random
from collections import defaultdict, deque
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
import pickle
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# ==================== CONFIGURATION ====================
@dataclass
class ModelConfig:
    """Configuration for the Two-Tower Model"""
    # Text embedding dimensions
    text_embedding_dim: int = 384  # Sentence transformer dimension
    
    # Hidden dimensions for different towers
    item_hidden_dim: int = 256
    user_hidden_dim: int = 256
    query_hidden_dim: int = 256
    
    # Image encoder settings
    image_encoder_type: str = 'resnet18'  # 'resnet18' or 'resnet50'
    freeze_image_layers: int = 20  # Number of layers to freeze from the end
    
    # Projection dimensions
    image_projection_dim: int = 512
    text_projection_dim: int = 512
    
    # Attention settings
    attention_heads: int = 4
    
    # Dropout rates
    image_dropout: float = 0.3
    text_dropout: float = 0.3
    user_dropout: float = 0.3
    query_dropout: float = 0.3
    
    # Scoring weights
    item_similarity_weight: float = 0.8  # Weight for item-to-item similarity
    user_preference_weight: float = 0.2  # Weight for user preference in similar items
    semantic_weight: float = 0.7  # Weight for sentence transformer similarity
    
    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    negative_samples: int = 5  # Number of negative samples per positive
    margin: float = 0.2  # Margin for triplet loss
    
    # Cache settings
    max_cache_size: int = 10000
    cache_ttl: int = 3600  # Cache time-to-live in seconds

# ==================== DATABASE INTERFACES ====================
class DatabaseInterface:
    """Interface for database operations"""
    
    async def get_user_history(self, user_id: str) -> Dict[str, List[str]]:
        """
        Get user interaction history from database
        
        Returns:
            {
                'viewed': List[str],  # List of viewed item IDs
                'liked': List[str],   # List of liked item IDs
                'timestamp': List[datetime]  # Optional: timestamps for interactions
            }
        """
        raise NotImplementedError("Implement this method for your database")
    
    async def get_user_histories_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Get multiple user histories in batch"""
        raise NotImplementedError("Implement this method for your database")
    
    async def record_interaction(self, user_id: str, item_id: str, 
                                interaction_type: str, metadata: Optional[Dict] = None):
        """Record a new user interaction"""
        raise NotImplementedError("Implement this method for your database")
    
    async def get_item_metadata(self, item_id: str) -> Optional[Dict]:
        """Get item metadata from database"""
        raise NotImplementedError("Implement this method for your database")

# Example implementation for PostgreSQL
class PostgreSQLInterface(DatabaseInterface):
    """PostgreSQL implementation of database interface"""
    
    def __init__(self, connection_pool):
        self.pool = connection_pool
    
    async def get_user_history(self, user_id: str) -> Dict[str, List[str]]:
        async with self.pool.acquire() as conn:
            # Get viewed items
            viewed_query = """
                SELECT item_id, created_at 
                FROM user_interactions 
                WHERE user_id = $1 AND interaction_type = 'view'
                ORDER BY created_at DESC
                LIMIT 1000
            """
            viewed_rows = await conn.fetch(viewed_query, user_id)
            
            # Get liked items
            liked_query = """
                SELECT item_id, created_at 
                FROM user_interactions 
                WHERE user_id = $1 AND interaction_type = 'like'
                ORDER BY created_at DESC
                LIMIT 500
            """
            liked_rows = await conn.fetch(liked_query, user_id)
            
            return {
                'viewed': [row['item_id'] for row in viewed_rows],
                'liked': [row['item_id'] for row in liked_rows],
                'viewed_timestamps': [row['created_at'] for row in viewed_rows],
                'liked_timestamps': [row['created_at'] for row in liked_rows]
            }
    
    async def record_interaction(self, user_id: str, item_id: str, 
                                interaction_type: str, metadata: Optional[Dict] = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_interactions (user_id, item_id, interaction_type, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, item_id, interaction_type, json.dumps(metadata or {}), datetime.utcnow())

# ==================== CACHING LAYER ====================
class CacheManager:
    """Manages caching for embeddings and user data"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.item_cache = {}
        self.user_cache = {}
        self.cache_timestamps = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, *args) -> str:
        """Generate cache key from arguments"""
        return hashlib.md5('_'.join(str(arg) for arg in args).encode()).hexdigest()
    
    async def get_or_compute_item_embedding(self, item_id: str, 
                                          compute_func, *args, **kwargs):
        """Get item embedding from cache or compute it"""
        cache_key = self._generate_key(item_id, *args, **kwargs)
        
        async with self._lock:
            # Check cache
            if cache_key in self.item_cache:
                timestamp = self.cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < self.config.cache_ttl:
                    return self.item_cache[cache_key]
            
            # Compute embedding
            embedding = await compute_func(item_id, *args, **kwargs)
            
            # Update cache
            self.item_cache[cache_key] = embedding
            self.cache_timestamps[cache_key] = time.time()
            
            # Evict old entries if cache is full
            if len(self.item_cache) > self.config.max_cache_size:
                oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
                del self.item_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            return embedding

# ==================== DATA STRUCTURES ====================
class UserInteractionBuffer:
    """Buffer to store recent user interactions for online learning"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.interactions = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        
    async def add_interaction(self, user_id: str, item_id: str, 
                            interaction_type: str, timestamp: float = None):
        """Add a new interaction to the buffer"""
        if timestamp is None:
            timestamp = time.time()
            
        async with self._lock:
            self.interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'type': interaction_type,  # 'view', 'like', 'dislike'
                'timestamp': timestamp
            })
    
    async def get_batch(self, batch_size: int):
        """Get a batch of interactions for training"""
        async with self._lock:
            if len(self.interactions) < batch_size:
                return list(self.interactions)
            
            # Sample random batch
            indices = np.random.choice(len(self.interactions), batch_size, replace=False)
            return [self.interactions[i] for i in indices]

# ==================== MODEL COMPONENTS ====================
class MultimodalItemEncoder(nn.Module):
    """Encoder that combines image and text features"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Image encoder
        if config.image_encoder_type == 'resnet18':
            self.image_encoder = models.resnet18(pretrained=True)
            image_features = 512
        else:
            self.image_encoder = models.resnet50(pretrained=True)
            image_features = 2048
            
        # Remove final layer
        self.image_encoder.fc = nn.Identity()
        
        # Freeze early layers
        params = list(self.image_encoder.parameters())
        for param in params[:-config.freeze_image_layers]:
            param.requires_grad = False
        
        # Image projection
        self.image_projection = nn.Sequential(
            nn.Linear(image_features, config.image_projection_dim),
            nn.ReLU(),
            nn.Dropout(config.image_dropout),
            nn.Linear(config.image_projection_dim, config.item_hidden_dim),
            nn.LayerNorm(config.item_hidden_dim)
        )
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.text_projection_dim),
            nn.ReLU(),
            nn.Dropout(config.text_dropout),
            nn.Linear(config.text_projection_dim, config.item_hidden_dim),
            nn.LayerNorm(config.item_hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            config.item_hidden_dim, 
            num_heads=config.attention_heads, 
            batch_first=True
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(config.item_hidden_dim, config.item_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.item_hidden_dim)
        )
        
    def forward(self, image_tensor=None, text_tensor=None):
        embeddings = []
        
        if image_tensor is not None:
            img_features = self.image_encoder(image_tensor)
            img_embedding = self.image_projection(img_features)
            embeddings.append(img_embedding.unsqueeze(1))
        
        if text_tensor is not None:
            text_embedding = self.text_projection(text_tensor)
            embeddings.append(text_embedding.unsqueeze(1))
        
        if len(embeddings) == 0:
            raise ValueError("At least one modality required")
        
        # Combine embeddings
        if len(embeddings) == 1:
            combined = embeddings[0].squeeze(1)
        else:
            stacked = torch.cat(embeddings, dim=1)
            attended, _ = self.attention(stacked, stacked, stacked)
            combined = attended.mean(dim=1)
        
        return self.final_projection(combined)

class EnhancedTwoTowerModel(nn.Module):
    """Complete two-tower recommendation model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Item encoder
        self.item_encoder = MultimodalItemEncoder(config)
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(config.item_hidden_dim, config.user_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.user_dropout),
            nn.Linear(config.user_hidden_dim * 2, config.user_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.user_dropout / 2),
            nn.Linear(config.user_hidden_dim, config.item_hidden_dim),
            nn.LayerNorm(config.item_hidden_dim)
        )
        
        # Query tower
        self.query_tower = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.query_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.query_dropout),
            nn.Linear(config.query_hidden_dim * 2, config.query_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.query_dropout / 2),
            nn.Linear(config.query_hidden_dim, config.item_hidden_dim),
            nn.LayerNorm(config.item_hidden_dim)
        )
        
    def get_item_embedding(self, image_tensor=None, text_tensor=None):
        """Get embedding for an item"""
        return self.item_encoder(image_tensor, text_tensor)
    
    def get_user_embedding(self, user_history_embeddings):
        """Get embedding for a user based on their history"""
        batch_size = user_history_embeddings.shape[0] if len(user_history_embeddings.shape) > 2 else 1
        if batch_size == 1 and len(user_history_embeddings.shape) == 2:
            # Single user case - weighted by recency
            weights = torch.linspace(0.5, 1.0, user_history_embeddings.shape[0])
            weights = weights / weights.sum()
            weights = weights.to(user_history_embeddings.device).unsqueeze(1)
            user_profile = (user_history_embeddings * weights).sum(dim=0, keepdim=True)
        else:
            # Batch case
            user_profile = user_history_embeddings.mean(dim=1)
        
        return self.user_tower(user_profile)
    
    def get_query_embedding(self, query_tensor):
        """Get embedding for a search query"""
        return self.query_tower(query_tensor)
    
    def compute_similarity(self, embedding1, embedding2, method='cosine'):
        """Compute similarity between two embeddings"""
        if method == 'cosine':
            return F.cosine_similarity(embedding1, embedding2, dim=-1)
        elif method == 'dot':
            return (embedding1 * embedding2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

# ==================== RECOMMENDATION SYSTEM ====================
class ProductionRecommendationSystem:
    """Production-ready recommendation system with database integration"""
    
    def __init__(self, 
                 model: EnhancedTwoTowerModel,
                 sentence_model: SentenceTransformer,
                 db_interface: DatabaseInterface,
                 data: Dict,
                 image_dir: str,
                 text_features: np.ndarray,
                 image_ids: List[str],
                 config: ModelConfig,
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.sentence_model = sentence_model
        self.db = db_interface
        self.data = data
        self.image_dir = image_dir
        self.text_features = text_features
        self.image_ids = image_ids
        self.config = config
        self.device = device
        
        # Cache manager
        self.cache_manager = CacheManager(config)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def load_image(self, image_path: str):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image).unsqueeze(0)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    async def get_item_embedding(self, item_id: str, use_image: bool = True, 
                               use_text: bool = True) -> Optional[torch.Tensor]:
        """Get embedding for a single item with caching"""
        
        async def compute_embedding():
            # Get text features
            text_tensor = None
            if use_text and item_id in self.image_ids:
                idx = self.image_ids.index(item_id)
                text_tensor = torch.tensor(
                    self.text_features[idx], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
            
            # Get image features
            image_tensor = None
            if use_image:
                image_path = os.path.join(self.image_dir, item_id)
                # Run image loading in thread pool
                loop = asyncio.get_event_loop()
                image_tensor = await loop.run_in_executor(
                    self.executor, self.load_image, image_path
                )
                if image_tensor is not None:
                    image_tensor = image_tensor.to(self.device)
            
            # Get embedding
            self.model.eval()
            with torch.no_grad():
                embedding = self.model.get_item_embedding(image_tensor, text_tensor)
            
            return embedding
        
        # Use cache manager
        return await self.cache_manager.get_or_compute_item_embedding(
            item_id, compute_embedding, use_image, use_text
        )
    
    async def get_user_history_embeddings(self, user_id: str, 
                                        max_items: int = 30) -> Optional[torch.Tensor]:
        """Get embeddings for user's history from database"""
        try:
            # Get user history from database
            user_history = await self.db.get_user_history(user_id)
            
            liked_items = user_history.get('liked', [])[-max_items:]
            
            if not liked_items:
                # Fall back to viewed items if no likes
                viewed_items = user_history.get('viewed', [])[-max_items:]
                if not viewed_items:
                    return None
                liked_items = viewed_items[:max_items//2]  # Use recent views as proxy
            
            # Get embeddings for liked items
            embeddings = []
            for item_id in liked_items:
                if item_id in self.image_ids:
                    emb = await self.get_item_embedding(item_id)
                    if emb is not None:
                        embeddings.append(emb)
            
            if not embeddings:
                return None
            
            return torch.cat(embeddings, dim=0)
            
        except Exception as e:
            self.logger.error(f"Error getting user history for {user_id}: {e}")
            return None
    
    async def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Advanced semantic search combining sentence transformers and learned embeddings"""
        
        # Get sentence transformer embedding
        query_embedding_st = self.sentence_model.encode([query], convert_to_numpy=True)[0]
        
        # Compute semantic scores in parallel
        semantic_scores = []
        for idx, item_id in enumerate(self.image_ids):
            item_st_embedding = self.text_features[idx]
            semantic_sim = np.dot(query_embedding_st, item_st_embedding) / (
                np.linalg.norm(query_embedding_st) * np.linalg.norm(item_st_embedding)
            )
            semantic_scores.append(semantic_sim)
        
        # Get two-tower model scores
        query_tensor = torch.tensor(query_embedding_st, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            query_embedding = self.model.get_query_embedding(query_tensor)
        
        # Score items in batches for efficiency
        final_scores = []
        batch_size = 100
        
        for i in range(0, len(self.image_ids), batch_size):
            batch_ids = self.image_ids[i:i+batch_size]
            batch_embeddings = []
            
            # Get embeddings for batch
            for item_id in batch_ids:
                emb = await self.get_item_embedding(item_id)
                if emb is not None:
                    batch_embeddings.append(emb)
            
            if batch_embeddings:
                batch_tensor = torch.cat(batch_embeddings, dim=0)
                with torch.no_grad():
                    model_sims = self.model.compute_similarity(
                        query_embedding.expand(batch_tensor.shape[0], -1), 
                        batch_tensor
                    ).cpu().numpy()
                
                # Combine scores
                for j, item_id in enumerate(batch_ids[:len(batch_embeddings)]):
                    idx = self.image_ids.index(item_id)
                    combined_score = (
                        self.config.semantic_weight * semantic_scores[idx] + 
                        (1 - self.config.semantic_weight) * model_sims[j]
                    )
                    final_scores.append((item_id, float(combined_score)))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
    
    async def find_similar_to_current(self, current_item_id: str, 
                                    user_id: Optional[str] = None, 
                                    top_k: int = 20) -> List[Tuple[str, float]]:
        """Find items similar to current item with optional user personalization"""
        
        if current_item_id not in self.image_ids:
            self.logger.error(f"Item {current_item_id} not found")
            return []
        
        # Get embedding of current item
        current_embedding = await self.get_item_embedding(current_item_id)
        if current_embedding is None:
            return []
        
        # Get user embedding if user_id provided
        user_embedding = None
        if user_id:
            user_history_embeddings = await self.get_user_history_embeddings(user_id)
            if user_history_embeddings is not None:
                with torch.no_grad():
                    user_embedding = self.model.get_user_embedding(user_history_embeddings)
        
        # Score items
        scores = []
        for item_id in self.image_ids:
            if item_id == current_item_id:
                continue
            
            item_embedding = await self.get_item_embedding(item_id)
            if item_embedding is None:
                continue
            
            with torch.no_grad():
                similarity = self.model.compute_similarity(
                    current_embedding, item_embedding
                ).item()
                
                # Boost with user preference if available
                if user_embedding is not None:
                    user_score = self.model.compute_similarity(
                        user_embedding, item_embedding
                    ).item()
                    similarity = (
                        self.config.item_similarity_weight * similarity + 
                        self.config.user_preference_weight * user_score
                    )
            
            scores.append((item_id, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    async def get_user_recommendations(self, user_id: str, 
                                     top_k: int = 20,
                                     filter_viewed: bool = True) -> List[Tuple[str, float]]:
        """Get personalized recommendations for a user based on their history"""
        
        # Get user history embeddings
        user_history_embeddings = await self.get_user_history_embeddings(user_id)
        
        if user_history_embeddings is None:
            self.logger.warning(f"No history found for user {user_id}")
            # Return popular items as fallback
            return await self.get_popular_items(top_k)
        
        # Get user embedding
        self.model.eval()
        with torch.no_grad():
            user_embedding = self.model.get_user_embedding(user_history_embeddings)
        
        # Get viewed items to filter if requested
        viewed_set = set()
        if filter_viewed:
            user_history = await self.db.get_user_history(user_id)
            viewed_set = set(user_history.get('viewed', []))
        
        # Score items
        scores = []
        for item_id in self.image_ids:
            if item_id in viewed_set:
                continue
            
            item_embedding = await self.get_item_embedding(item_id)
            if item_embedding is None:
                continue
            
            with torch.no_grad():
                score = self.model.compute_similarity(
                    user_embedding, item_embedding
                ).item()
            
            scores.append((item_id, float(score)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    async def get_popular_items(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get popular items as fallback recommendations"""
        # This is a placeholder - implement based on your popularity metrics
        popular_items = random.sample(self.image_ids, min(top_k, len(self.image_ids)))
        return [(item_id, 1.0) for item_id in popular_items]
    
    async def record_interaction(self, user_id: str, item_id: str, 
                               interaction_type: str, metadata: Optional[Dict] = None):
        """Record user interaction to database"""
        try:
            await self.db.record_interaction(user_id, item_id, interaction_type, metadata)
        except Exception as e:
            self.logger.error(f"Error recording interaction: {e}")
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'image_ids': self.image_ids,
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, db_interface: DatabaseInterface, 
                   data: Dict, image_dir: str, device: str = 'cpu'):
        """Load model from saved state"""
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint['config']
        model = EnhancedTwoTowerModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load text features
        text_features_path = path.replace('.pt', '_text_features.pkl')
        with open(text_features_path, 'rb') as f:
            text_features = pickle.load(f)
        
        # Initialize sentence transformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return cls(
            model=model,
            sentence_model=sentence_model,
            db_interface=db_interface,
            data=data,
            image_dir=image_dir,
            text_features=text_features,
            image_ids=checkpoint['image_ids'],
            config=config,
            device=device
        )

# ==================== USAGE EXAMPLE ====================
async def main():
    """Example usage of the production recommendation system"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database interface (implement for your database)
    db = PostgreSQLInterface(connection_pool)
    # OR
    # db = MongoDBInterface(client)
    # OR
    # db = MySQLInterface(connection)
    
    # Load data and model
    config = ModelConfig()
    
    # For production, load from your trained model
    rec_system = ProductionRecommendationSystem.load_model(
        'path/to/saved_model.pt', 
        db, 
        data, 
        image_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example API endpoints
    
    # 1. Get user recommendations
    user_id = "user123"
    recommendations = await rec_system.get_user_recommendations(user_id, top_k=20)
    print(f"Recommendations for {user_id}:", recommendations[:5])
    
    # 2. Semantic search
    search_results = await rec_system.semantic_search(
        "modern minimalist bedroom", 
        top_k=10
    )
    print("Search results:", search_results[:5])
    
    # 3. Find similar items
    current_item = "image123.jpg"
    similar_items = await rec_system.find_similar_to_current(
        current_item, 
        user_id=user_id,
        top_k=15
    )
    print(f"Items similar to {current_item}:", similar_items[:5])
    
    # 4. Record interaction
    await rec_system.record_interaction(
        user_id, 
        "image456.jpg", 
        "view",
        metadata={'source': 'homepage', 'duration': 5.2}
    )

if __name__ == "__main__":
    asyncio.run(main())
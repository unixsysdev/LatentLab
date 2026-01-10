"""
Latent Cartographer for LatentLab

Handles dimensionality reduction and space building for visualization.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Optional UMAP import (better for non-linear, but slower)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.info("UMAP not available, using PCA only")


class LatentCartographer:
    """
    Maps high-dimensional activation space to 3D for visualization.
    
    The cartographer maintains a "reference space" built from seed concepts
    to ensure consistent projections across different prompts.
    """
    
    def __init__(self, hidden_size: int = 3584, n_components: int = 3):
        self.hidden_size = hidden_size
        self.n_components = n_components
        
        # PCA for fast projection  
        self.pca = PCA(n_components=n_components)
        self.pca_fitted = False
        
        # Optional UMAP for better clustering
        self.umap_reducer = None
        if HAS_UMAP:
            self.umap_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
        self.umap_fitted = False
        
        # Store reference vectors for consistent mapping
        self.reference_vectors: Optional[np.ndarray] = None
        self.reference_labels: List[str] = []
        
    def build_reference_space(
        self,
        vectors: List[np.ndarray],
        labels: Optional[List[str]] = None
    ):
        """
        Build the reference coordinate system from seed vectors.
        
        This ensures that the same concepts always map to similar
        locations in 3D space, making comparisons meaningful.
        """
        if len(vectors) < self.n_components:
            logger.warning(f"Need at least {self.n_components} vectors, got {len(vectors)}")
            # Pad with random vectors
            while len(vectors) < self.n_components:
                vectors.append(np.random.randn(self.hidden_size))
                
        matrix = np.vstack([v.reshape(1, -1) for v in vectors])
        self.reference_vectors = matrix
        self.reference_labels = labels or [f"ref_{i}" for i in range(len(vectors))]
        
        # Fit PCA
        self.pca.fit(matrix)
        self.pca_fitted = True
        logger.info(f"Built reference space with {len(vectors)} vectors")
        
        # Fit UMAP if available (slower)
        if self.umap_reducer is not None and len(vectors) >= 15:
            try:
                self.umap_reducer.fit(matrix)
                self.umap_fitted = True
                logger.info("UMAP reference space built")
            except Exception as e:
                logger.warning(f"UMAP fitting failed: {e}")
                
    def project(
        self,
        vectors: np.ndarray,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Project high-dimensional vectors to 3D.
        
        Args:
            vectors: Array of shape (n_samples, hidden_size) or (hidden_size,)
            method: 'pca' or 'umap'
            
        Returns:
            Array of shape (n_samples, 3) or (3,)
        """
        # Handle single vector
        single = vectors.ndim == 1
        if single:
            vectors = vectors.reshape(1, -1)
            
        if method == "umap" and self.umap_fitted:
            coords = self.umap_reducer.transform(vectors)
        elif self.pca_fitted:
            coords = self.pca.transform(vectors)
        else:
            # Fallback: just take first 3 components
            logger.warning("Projector not fitted, using raw slicing")
            coords = vectors[:, :self.n_components]
            
        return coords[0] if single else coords
    
    def interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        steps: int = 7,
        method: str = "linear"
    ) -> np.ndarray:
        """
        Interpolate between two vectors.
        
        Args:
            start: Starting vector
            end: Ending vector  
            steps: Number of interpolation points
            method: 'linear' (LERP) or 'spherical' (SLERP)
            
        Returns:
            Array of shape (steps, hidden_size)
        """
        alphas = np.linspace(0, 1, steps)
        
        if method == "spherical":
            # Spherical interpolation (better for normalized embeddings)
            start_norm = start / np.linalg.norm(start)
            end_norm = end / np.linalg.norm(end)
            
            omega = np.arccos(np.clip(np.dot(start_norm, end_norm), -1, 1))
            if omega < 1e-10:
                # Vectors are nearly identical
                return np.tile(start, (steps, 1))
                
            result = []
            for alpha in alphas:
                t = (np.sin((1 - alpha) * omega) / np.sin(omega)) * start_norm
                t += (np.sin(alpha * omega) / np.sin(omega)) * end_norm
                result.append(t * (np.linalg.norm(start) * (1 - alpha) + np.linalg.norm(end) * alpha))
            return np.array(result)
        else:
            # Linear interpolation
            return np.array([(1 - a) * start + a * end for a in alphas])
    
    def compute_trajectory(
        self,
        layer_activations: Dict[str, np.ndarray],
        token_idx: int = -1
    ) -> List[List[float]]:
        """
        Compute the 3D trajectory of a token through layers.
        
        Args:
            layer_activations: Dict of layer_name -> activation tensor
            token_idx: Which token position to trace (-1 for last)
            
        Returns:
            List of [x, y, z] coordinates, one per layer
        """
        trajectory = []
        
        # Sort layers by number
        layer_names = sorted(
            layer_activations.keys(),
            key=lambda x: int(x.split("_")[1])
        )
        
        for name in layer_names:
            act = layer_activations[name]
            
            # Handle different shapes
            if len(act.shape) == 3:
                # [batch, seq, hidden]
                vec = act[0, token_idx, :].float().cpu().numpy()
            elif len(act.shape) == 2:
                # [seq, hidden]
                vec = act[token_idx, :].float().cpu().numpy()
            else:
                vec = act.float().cpu().numpy().flatten()
                
            # Project to 3D
            coords = self.project(vec)
            trajectory.append(coords.tolist())
            
        return trajectory
    
    def vector_arithmetic(
        self,
        positive: List[np.ndarray],
        negative: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute vector arithmetic: sum(positive) - sum(negative)
        
        Useful for steering vectors like Love - Hate.
        """
        result = np.zeros(self.hidden_size)
        for p in positive:
            result += p
        for n in negative:
            result -= n
        return result

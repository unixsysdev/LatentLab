"""
LatentLab Engine

The core orchestrator that combines model, cartographer, and experiments.
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Any, Callable
import logging

from .model_loader import HookedModel, load_model, DEFAULT_MODEL
from .latent_cartographer import LatentCartographer

logger = logging.getLogger(__name__)


class LatentSpaceEngine:
    """
    Main engine for LatentLab.
    
    Provides high-level operations for:
    - Embedding text
    - Generating text
    - Tracing thoughts through layers
    - Vector arithmetic
    """
    
    def __init__(
        self,
        model: Optional[HookedModel] = None,
        model_name: str = DEFAULT_MODEL,
        seed_prompts: Optional[List[str]] = None
    ):
        # Load model if not provided
        if model is None:
            self.model = load_model(model_name=model_name)
        else:
            self.model = model
            
        # Initialize cartographer
        self.cartographer = LatentCartographer(
            hidden_size=self.model.hidden_size
        )
        
        # Build reference space from seed prompts
        if seed_prompts is None:
            seed_prompts = [
                "The quick brown fox jumps over the lazy dog.",
                "In the beginning, the universe was created.",
                "Love is patient, love is kind.",
                "The stock market crashed today.",
                "Quantum mechanics describes nature at the atomic scale.",
                "The chef prepared a delicious meal.",
                "War and peace are two sides of the same coin.",
                "The child laughed with pure joy.",
                "Mathematics is the language of the universe.",
                "The ancient ruins tell stories of forgotten civilizations.",
            ]
        
        self._build_reference_space(seed_prompts)
        
    def _build_reference_space(self, prompts: List[str]):
        """Build the 3D projection space from seed prompts"""
        logger.info(f"Building reference space from {len(prompts)} prompts...")
        vectors = []
        for prompt in prompts:
            vec = self.embed(prompt)
            vectors.append(vec)
        
        self.cartographer.build_reference_space(vectors, prompts)
        logger.info("Reference space built successfully")
        
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        vec = self.model.embed(text)
        return vec.float().cpu().numpy()  # Convert to float32 for numpy
    
    def unembed(self, vector: np.ndarray, top_k: int = 10) -> List[str]:
        """Project vector back to vocabulary"""
        # Convert numpy to torch
        vec_tensor = torch.from_numpy(vector)
        return self.model.unembed(vec_tensor, top_k=top_k)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        return self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def trace_thought(
        self,
        text: str,
        layers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Trace how a thought evolves through the model layers.
        
        Returns dict with:
        - tokens: List of tokens
        - trajectories: Dict of token_idx -> layer trajectory
        - coords_3d: 3D coordinates for visualization
        """
        if layers is None:
            # Sample every other layer for performance
            layers = list(range(0, self.model.n_layers, 2))
            
        result = self.model.forward_with_cache(text, layers=layers)
        
        trajectories = {}
        for token_idx in range(len(result["tokens"])):
            trajectory = self.cartographer.compute_trajectory(
                result["activations"],
                token_idx=token_idx
            )
            trajectories[token_idx] = trajectory
            
        return {
            "tokens": result["tokens"],
            "trajectories": trajectories,
            "activations": result["activations"]  # Raw activations for advanced use
        }
    
    def interpolate_concepts(
        self,
        start: str,
        end: str,
        steps: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Interpolate between two concepts in embedding space.
        
        Returns list of points with coords_3d and labels.
        """
        v_start = self.embed(start)
        v_end = self.embed(end)
        
        vectors = self.cartographer.interpolate(v_start, v_end, steps)
        coords = self.cartographer.project(vectors)
        
        points = []
        for i, (vec, coord) in enumerate(zip(vectors, coords)):
            alpha = i / (steps - 1)
            points.append({
                "index": i,
                "alpha": alpha,
                "vector": vec,
                "coords_3d": coord.tolist()
            })
            
        return points
    
    def compute_steering_vector(
        self,
        positive: str,
        negative: str
    ) -> np.ndarray:
        """Compute a steering vector: embed(positive) - embed(negative)"""
        v_pos = self.embed(positive)
        v_neg = self.embed(negative)
        return v_pos - v_neg
    
    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: np.ndarray,
        layer: int = 15,
        strength: float = 1.0,
        max_tokens: int = 50
    ) -> Dict[str, str]:
        """
        Generate text with activation steering.
        
        Returns both original and steered outputs for comparison.
        """
        # Original generation
        original = self.generate(prompt, max_tokens=max_tokens)
        
        # Steered generation
        steering_tensor = torch.from_numpy(steering_vector).to(
            self.model.model.device
        ).to(self.model.dtype)
        
        # Add steering hook
        handle = self.model.inject_at_layer(
            layer,
            steering_tensor,
            strength=strength
        )
        
        try:
            steered = self.generate(prompt, max_tokens=max_tokens)
        finally:
            handle.remove()
            
        return {
            "original": original,
            "steered": steered
        }
    
    @property
    def n_layers(self) -> int:
        return self.model.n_layers
    
    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size
    
    @property
    def model_name(self) -> str:
        return self.model.model_name
    
    @property
    def device(self) -> str:
        return self.model.device


# Singleton engine instance
_engine: Optional[LatentSpaceEngine] = None


def get_engine(model_name: str = DEFAULT_MODEL) -> LatentSpaceEngine:
    """Get or create the global engine instance"""
    global _engine
    if _engine is None:
        _engine = LatentSpaceEngine(model_name=model_name)
    return _engine


def reset_engine():
    """Reset the global engine (for testing)"""
    global _engine
    _engine = None

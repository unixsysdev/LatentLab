"""
Steering Experiment

Activation steering (ActAdd) - inject vectors to alter model behavior.
"""

from typing import Any, Dict
import numpy as np
import torch

from .base import Experiment
from ..models import ExperimentResult, SteeringInput, VectorPoint


class SteeringExperiment(Experiment):
    """
    The Steering Vector: Inject concepts into the model's brain.
    
    This is the most "magical" experiment - we calculate a direction
    (e.g., Love - Hate) and inject it during generation to alter output.
    """
    
    name = "steering"
    description = "Inject activation vectors to alter model behavior"
    input_model = SteeringInput
    
    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)
        
        prompt = validated.prompt
        positive = validated.positive_concept
        negative = validated.negative_concept
        layer = validated.layer
        strength = validated.strength
        
        # Clamp layer to valid range
        max_layer = self.engine.n_layers - 1
        layer = min(layer, max_layer)
        
        # Calculate steering vector
        v_positive = self.engine.embed(positive)
        v_negative = self.engine.embed(negative)
        steering_vec = v_positive - v_negative
        
        # Normalize and scale
        steering_norm = np.linalg.norm(steering_vec)
        
        # Generate original and steered outputs
        results = self.engine.generate_with_steering(
            prompt=prompt,
            steering_vector=steering_vec,
            layer=layer,
            strength=strength,
            max_tokens=80
        )
        
        # Also trace the thought paths for visualization
        # (This shows where in 3D space the thought moves)
        
        # Embed the concepts and outputs for visualization
        prompt_vec = self.engine.embed(prompt)
        original_vec = self.engine.embed(prompt + results["original"])
        steered_vec = self.engine.embed(prompt + results["steered"])
        
        # Project all to 3D
        all_vecs = np.vstack([
            prompt_vec.reshape(1, -1),
            v_positive.reshape(1, -1),
            v_negative.reshape(1, -1),
            original_vec.reshape(1, -1),
            steered_vec.reshape(1, -1)
        ])
        coords_3d = self.engine.cartographer.project(all_vecs)
        
        points = [
            VectorPoint(
                label=f"Prompt: {prompt[:30]}...",
                coords_3d=coords_3d[0].tolist(),
                metadata={"type": "prompt"}
            ),
            VectorPoint(
                label=f"+: {positive}",
                coords_3d=coords_3d[1].tolist(),
                metadata={"type": "positive"}
            ),
            VectorPoint(
                label=f"-: {negative}",
                coords_3d=coords_3d[2].tolist(),
                metadata={"type": "negative"}
            ),
            VectorPoint(
                label="Original Output",
                coords_3d=coords_3d[3].tolist(),
                metadata={
                    "type": "original",
                    "text": results["original"][:200]
                }
            ),
            VectorPoint(
                label="Steered Output",
                coords_3d=coords_3d[4].tolist(),
                metadata={
                    "type": "steered",
                    "text": results["steered"][:200]
                }
            )
        ]
        
        # Connections showing the steering direction
        connections = [
            (0, 3),  # Prompt → Original
            (0, 4),  # Prompt → Steered
            (2, 1),  # Negative → Positive (steering direction)
        ]
        
        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Steering '{negative}' → '{positive}' at layer {layer} with strength {strength}",
            metadata={
                "prompt": prompt,
                "positive": positive,
                "negative": negative,
                "layer": layer,
                "strength": strength,
                "steering_magnitude": float(steering_norm),
                "original_output": results["original"],
                "steered_output": results["steered"]
            }
        )

"""
Wormhole Experiment

Visualizes the semantic trajectory between two distant concepts.
"""

import time
import logging
from typing import Any, Dict
import numpy as np

from .base import Experiment
from ..models import ExperimentResult, WormholeInput, VectorPoint

logger = logging.getLogger(__name__)


class WormholeExperiment(Experiment):
    """
    The Semantic Wormhole: Visualize the path between two concepts.
    
    Instead of just showing the endpoints, we interpolate through
    the latent space and ask the LLM to name the midpoints.
    """
    
    name = "wormhole"
    description = "Visualize the semantic trajectory between two distant concepts"
    input_model = WormholeInput
    
    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)
        
        start_concept = validated.start
        end_concept = validated.end
        steps = validated.steps
        device = self.engine.device
        
        logger.info(f"[WORMHOLE] Starting: '{start_concept}' -> '{end_concept}' ({steps} steps, device: {device})")
        total_start = time.time()
        
        # Get embeddings
        logger.info(f"[WORMHOLE] Step 1/4: Embedding endpoints... (GPU)")
        t0 = time.time()
        v_start = self.engine.embed(start_concept)
        v_end = self.engine.embed(end_concept)
        logger.info(f"[WORMHOLE] Step 1/4: Done in {time.time()-t0:.2f}s")
        
        # Linear interpolation
        logger.info(f"[WORMHOLE] Step 2/4: Interpolating {steps} points... (CPU)")
        t0 = time.time()
        vectors = self.engine.cartographer.interpolate(v_start, v_end, steps)
        logger.info(f"[WORMHOLE] Step 2/4: Done in {time.time()-t0:.2f}s")
        
        # Project to 3D
        logger.info(f"[WORMHOLE] Step 3/4: Projecting to 3D... (CPU PCA)")
        t0 = time.time()
        coords_3d = self.engine.cartographer.project(vectors)
        logger.info(f"[WORMHOLE] Step 3/4: Done in {time.time()-t0:.2f}s")
        
        # Labels: known anchors at start and end, LLM-generated midpoints
        labels = [start_concept] + ["?"] * (steps - 2) + [end_concept]
        
        # Generate stepping stone concepts using LLM
        logger.info(f"[WORMHOLE] Step 4/4: Generating {steps-2} waypoint labels... (LLM)")
        current_context = start_concept
        for i in range(1, steps - 1):
            t0 = time.time()
            alpha = i / (steps - 1)
            prompt = f"""You are identifying concepts that lie on a semantic path.

Start concept: "{start_concept}"
End concept: "{end_concept}"
Current position: {alpha:.0%} of the way from start to end.
Previous waypoint: "{current_context}"

What single concept or idea exists at this point in the semantic journey?
Respond with ONLY the concept name (1-5 words), nothing else."""

            try:
                mid_concept = self.engine.generate(
                    prompt,
                    max_tokens=20,
                    temperature=0.7
                ).strip().split('\n')[0]
                
                # Clean up the response
                mid_concept = mid_concept.strip("\"'")[:50]
                if not mid_concept:
                    mid_concept = f"Point {i}"
            except Exception as e:
                mid_concept = f"Point {i}"
                
            labels[i] = mid_concept
            current_context = mid_concept
            logger.info(f"[WORMHOLE] Waypoint {i}/{steps-2}: '{mid_concept}' ({time.time()-t0:.2f}s)")
        
        logger.info(f"[WORMHOLE] Total time: {time.time()-total_start:.2f}s")
        
        # Build result
        points = []
        for i, (label, vec, coord) in enumerate(zip(labels, vectors, coords_3d)):
            points.append(VectorPoint(
                label=label,
                coords_3d=coord.tolist(),
                vector=vec.tolist() if i in [0, steps-1] else None,  # Only store endpoints
                metadata={
                    "alpha": i / (steps - 1),
                    "is_anchor": i == 0 or i == steps - 1
                }
            ))
        
        # Connect sequentially
        connections = [(i, i + 1) for i in range(steps - 1)]
        
        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Semantic trajectory from '{start_concept}' to '{end_concept}'",
            metadata={
                "start": start_concept,
                "end": end_concept,
                "steps": steps,
                "distance": float(np.linalg.norm(v_end - v_start))
            }
        )

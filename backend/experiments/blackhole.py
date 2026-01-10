"""
Blackhole Experiment

Finds multiple semantic paths between two concepts by:
1. Extracting activation neighborhoods (hidden state + MLP residual) 
2. Finding bridge concepts that appear in both concept's neighborhoods
3. Asking the LLM to generate paths through different semantic "lenses"
4. Visualizing all paths with different colors
"""

import re
import time
import logging
from typing import Any, Dict, List, Set, Tuple
import numpy as np

from .base import Experiment
from ..models import ExperimentResult, VectorPoint

logger = logging.getLogger(__name__)


# Pydantic model for input validation
from pydantic import BaseModel, Field


class BlackholeInput(BaseModel):
    """Input for Blackhole experiment"""
    start: str = Field(..., description="Starting concept")
    end: str = Field(..., description="Ending concept")
    num_paths: int = Field(default=3, ge=1, le=5, description="Number of semantic paths to find")
    steps_per_path: int = Field(default=3, ge=2, le=6, description="Steps per path")


class BlackholeExperiment(Experiment):
    """
    The Semantic Blackhole: Multiple paths converging between two concepts.
    
    Unlike Wormhole (single interpolated path), Blackhole discovers multiple
    semantic routes using:
    1. Activation-based bridge concept discovery
    2. LLM-guided path generation with different semantic lenses
    3. Multi-path visualization with color coding
    """
    
    name = "blackhole"
    description = "Find multiple semantic paths between two concepts"
    input_model = BlackholeInput
    
    # Stop words for filtering
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'it', 'to', 'of', 'in', 'on', 'at', 'by', 'for',
        'and', 'or', 'but', 'not', 'be', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'your', 'his', 'her',
        'its', 'our', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    }
    
    # Semantic lenses for different path types
    PATH_LENSES = [
        {
            "name": "semantic",
            "color": "primary",  # cyan
            "prompt": "Find words that bridge the MEANING between these concepts. Focus on synonyms, definitions, and direct semantic relationships."
        },
        {
            "name": "emotional", 
            "color": "secondary",  # pink
            "prompt": "Find words that bridge the EMOTIONAL or AFFECTIVE connection between these concepts. Focus on feelings, moods, and emotional associations."
        },
        {
            "name": "categorical",
            "color": "tertiary",  # green
            "prompt": "Find words that bridge the CATEGORICAL relationship between these concepts. Focus on hypernyms, hyponyms, and taxonomic relationships."
        },
        {
            "name": "associative",
            "color": "accent",  # orange
            "prompt": "Find words that bridge through FREE ASSOCIATION. What concepts connect these through culture, memory, or common context?"
        },
        {
            "name": "metaphorical",
            "color": "positive",  # bright green
            "prompt": "Find words that bridge through METAPHOR or ANALOGY. What abstract connections link these concepts?"
        },
    ]

    def _extract_neighborhood(self, concept: str) -> Tuple[np.ndarray, List[str]]:
        """
        Extract semantic neighborhood using both hidden state and MLP residual.
        Returns the embedding and list of neighbor concepts.
        """
        last_layer = self.engine.n_layers - 1
        
        # Get full embedding
        full_embedding = self.engine.embed(concept)
        
        # Get MLP residual
        mlp_result = self.engine.model.forward_with_mlp_residuals(concept, layers=[last_layer])
        layer_key = f"layer_{last_layer}"
        
        if layer_key in mlp_result["mlp_residuals"]:
            mlp_residual = mlp_result["mlp_residuals"][layer_key]
            mlp_vec = mlp_residual.mean(dim=1).squeeze(0).float().cpu().numpy()
        else:
            mlp_vec = None
        
        # Unembed both
        raw_from_full = self.engine.unembed(full_embedding, top_k=30)
        raw_from_mlp = self.engine.unembed(mlp_vec, top_k=30) if mlp_vec is not None else []
        
        # Merge and filter
        seen = set()
        neighbors = []
        for c in raw_from_mlp + raw_from_full:
            c_lower = c.lower()
            if (c_lower not in seen and 
                c_lower != concept.lower() and 
                c_lower not in self.STOP_WORDS and 
                len(c) >= 3):
                neighbors.append(c)
                seen.add(c_lower)
        
        return full_embedding, neighbors[:25]
    
    def _find_bridge_concepts(self, neighbors_start: List[str], neighbors_end: List[str]) -> List[str]:
        """Find concepts that appear in both neighborhoods (potential bridges)."""
        start_set = {n.lower() for n in neighbors_start}
        end_set = {n.lower() for n in neighbors_end}
        
        bridges = start_set & end_set
        
        # Return original casing
        result = []
        for n in neighbors_start + neighbors_end:
            if n.lower() in bridges and n not in result:
                result.append(n)
        return result[:5]
    
    def _generate_path(
        self, 
        start: str, 
        end: str, 
        lens: Dict[str, str],
        bridges: List[str],
        num_steps: int
    ) -> List[str]:
        """
        Ask the LLM to generate a semantic path using a specific lens.
        """
        bridges_str = ", ".join(bridges[:5]) if bridges else "none found"
        
        prompt = f"""Find a semantic path from "{start}" to "{end}".

Lens: {lens['prompt']}

Bridge concepts found in both neighborhoods: {bridges_str}

Generate EXACTLY {num_steps} intermediate concepts that form a path from "{start}" to "{end}".
Each concept should logically connect to the previous one.
Use the bridge concepts if they fit naturally.

Output format: One concept per line, no numbering, no explanations.
Start your response with the first intermediate concept (not "{start}")."""

        try:
            response = self.engine.generate(prompt, max_tokens=30, temperature=0.5)
            
            # Parse response
            lines = response.strip().split('\n')
            path = []
            for line in lines:
                word = line.strip().strip('-').strip('•').strip('*').strip()
                if not word or len(word) < 3 or len(word) > 40:
                    continue
                if word.lower() in self.STOP_WORDS:
                    continue
                if word.lower() == start.lower() or word.lower() == end.lower():
                    continue
                # Allow multi-word but check first word
                first_word = word.split()[0].lower()
                if first_word in self.STOP_WORDS:
                    continue
                path.append(word)
                if len(path) >= num_steps:
                    break
            
            return path[:num_steps]
            
        except Exception as e:
            return []
    
    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)
        
        start_concept = validated.start
        end_concept = validated.end
        num_paths = validated.num_paths
        steps_per_path = validated.steps_per_path
        device = self.engine.device
        
        logger.info(f"[BLACKHOLE] Starting: '{start_concept}' -> '{end_concept}' (device: {device})")
        logger.info(f"[BLACKHOLE] Config: {num_paths} paths, {steps_per_path} steps each")
        total_start = time.time()
        
        # 1. Extract neighborhoods for both concepts
        logger.info(f"[BLACKHOLE] Step 1/4: Extracting neighborhood for '{start_concept}'... (GPU)")
        t0 = time.time()
        start_embedding, start_neighbors = self._extract_neighborhood(start_concept)
        logger.info(f"[BLACKHOLE] Found {len(start_neighbors)} neighbors in {time.time()-t0:.2f}s")
        
        t0 = time.time()
        logger.info(f"[BLACKHOLE] Step 1/4: Extracting neighborhood for '{end_concept}'... (GPU)")
        end_embedding, end_neighbors = self._extract_neighborhood(end_concept)
        logger.info(f"[BLACKHOLE] Found {len(end_neighbors)} neighbors in {time.time()-t0:.2f}s")
        
        # 2. Find bridge concepts
        logger.info(f"[BLACKHOLE] Step 2/4: Finding bridge concepts... (CPU)")
        t0 = time.time()
        bridges = self._find_bridge_concepts(start_neighbors, end_neighbors)
        logger.info(f"[BLACKHOLE] Found {len(bridges)} bridges: {bridges} in {time.time()-t0:.2f}s")
        
        # 3. Generate multiple paths using different lenses
        logger.info(f"[BLACKHOLE] Step 3/4: Generating {num_paths} paths with LLM...")
        all_paths: List[Tuple[Dict, List[str]]] = []
        
        for i, lens in enumerate(self.PATH_LENSES[:num_paths]):
            t0 = time.time()
            logger.info(f"[BLACKHOLE] Generating path {i+1}/{num_paths}: {lens['name']} lens...")
            path = self._generate_path(
                start_concept, 
                end_concept, 
                lens,
                bridges,
                steps_per_path
            )
            if path:
                all_paths.append((lens, path))
                logger.info(f"[BLACKHOLE] Path {i+1}: {path} ({time.time()-t0:.2f}s)")
            else:
                logger.info(f"[BLACKHOLE] Path {i+1}: Failed ({time.time()-t0:.2f}s)")
        
        # 4. Build points and connections
        points = []
        connections = []
        
        # Start point (index 0) - prominent
        start_coord = self.engine.cartographer.project(start_embedding).tolist()
        points.append(VectorPoint(
            label=start_concept,
            coords_3d=start_coord,
            metadata={"type": "start", "size": 3.0}
        ))
        
        # End point (index 1) - prominent
        end_coord = self.engine.cartographer.project(end_embedding).tolist()
        points.append(VectorPoint(
            label=end_concept,
            coords_3d=end_coord,
            metadata={"type": "end", "size": 3.0}
        ))
        
        # Add path points
        path_metadata = []
        for path_idx, (lens, path_concepts) in enumerate(all_paths):
            path_start_idx = len(points)
            prev_idx = 0  # Start connects to start_concept
            
            for step_idx, concept in enumerate(path_concepts):
                # Embed and project
                try:
                    vec = self.engine.embed(concept)
                    coord = self.engine.cartographer.project(vec).tolist()
                except:
                    # Skip if embedding fails
                    continue
                
                point_idx = len(points)
                points.append(VectorPoint(
                    label=concept,
                    coords_3d=coord,
                    metadata={
                        "type": f"path_{lens['name']}",
                        "path_index": path_idx,
                        "step": step_idx,
                        "lens": lens['name'],
                        "color": lens['color']
                    }
                ))
                
                # Connect to previous point in path
                connections.append((prev_idx, point_idx))
                prev_idx = point_idx
            
            # Connect last path point to end
            if prev_idx != 0:  # Only if we added points
                connections.append((prev_idx, 1))  # 1 is end_concept
            
            path_metadata.append({
                "lens": lens['name'],
                "color": lens['color'],
                "concepts": path_concepts
            })
        
        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Blackhole: {len(all_paths)} semantic paths from '{start_concept}' to '{end_concept}'",
            metadata={
                "start": start_concept,
                "end": end_concept,
                "bridges": bridges,
                "paths": path_metadata,
                "start_neighbors": start_neighbors[:10],
                "end_neighbors": end_neighbors[:10]
            }
        )

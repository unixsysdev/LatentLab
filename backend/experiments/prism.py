"""
Concept Prism Experiment

Uses the "Reverse Embedding" technique to spectrally analyze a concept
by projecting its input embedding back onto the vocabulary.
"""

import re
import time
import logging
from typing import Any, Dict, List
import numpy as np

from .base import Experiment
from ..models import ExperimentResult, ConceptPrismInput, VectorPoint

logger = logging.getLogger(__name__)


class ConceptPrismExperiment(Experiment):
    """
    The Concept Prism: Spectrally analyze a concept using reverse embedding.

    This experiment takes the input embedding of a concept and projects it
    back onto the vocabulary (using the LM Head) to find its constituent
    semantic components (nearest neighbors in latent space).
    
    Then uses the model itself to refine and validate the raw candidates.
    """

    name = "prism"
    description = "Spectrally analyze a concept using reverse embedding"
    input_model = ConceptPrismInput

    # Common stop words / articles that should never appear
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
        'if', 'as', 'with', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'under', 'again', 'further',
    }

    def _refine_with_model(self, concept: str, raw_candidates: List[str]) -> List[str]:
        """
        Ask the model to refine the raw candidates into semantically meaningful words.
        
        The raw extraction from activations is noisy - the model itself can
        tell us which words actually make sense as semantic components.
        """
        # Pre-filter raw candidates
        filtered_raw = [c for c in raw_candidates if c.lower() not in self.STOP_WORDS and len(c) >= 3]
        candidates_str = ", ".join(filtered_raw[:25])
        
        prompt = f"""I extracted these tokens from a neural network's hidden state for the concept "{concept}":

Raw tokens: {candidates_str}

Your task: Select 10-12 MEANINGFUL words that are semantically related to "{concept}".
Rules:
- Only nouns, adjectives, or verbs (NO articles, pronouns, prepositions)
- Each word must have 3+ letters
- Words must be genuinely related to "{concept}" (synonyms, associations, related concepts)
- Replace any noise with better semantic associations

Output format: One word per line, no numbering, no explanations, no punctuation."""

        try:
            response = self.engine.generate(prompt, max_tokens=30, temperature=0.3)
            
            # Parse response - one word per line
            lines = response.strip().split('\n')
            refined = []
            for line in lines:
                word = line.strip().strip('-').strip('•').strip('*').strip()
                # Strict filtering
                if not word:
                    continue
                # Must be 3+ chars
                if len(word) < 3 or len(word) > 30:
                    continue
                # Must be alphabetic (allow spaces for two-word phrases)
                if not re.match(r'^[a-zA-Z]+(\s[a-zA-Z]+)?$', word):
                    continue
                # Not a stop word
                if word.lower() in self.STOP_WORDS:
                    continue
                refined.append(word)
            
            # Fallback if parsing failed
            if len(refined) < 5:
                return filtered_raw[:12]
                
            return refined[:12]
            
        except Exception as e:
            # Fallback to raw if generation fails
            return filtered_raw[:12]

    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)
        concept = validated.concept
        device = self.engine.device
        
        logger.info(f"[PRISM] Starting experiment for '{concept}' (device: {device})")
        total_start = time.time()

        # 1. Get the full embedding (single forward pass)
        logger.info(f"[PRISM] Step 1/5: Embedding concept... (GPU forward pass)")
        t0 = time.time()
        center_vec = self.engine.embed(concept)
        logger.info(f"[PRISM] Step 1/5: Done in {time.time()-t0:.2f}s")

        # 2. Unembed to find raw semantic components
        logger.info(f"[PRISM] Step 2/5: Unembedding to find candidates... (CPU matmul)")
        t0 = time.time()
        raw_components = self.engine.unembed(center_vec, top_k=30)
        raw_components = [c for c in raw_components if c.lower() != concept.lower()]
        logger.info(f"[PRISM] Step 2/5: Found {len(raw_components)} candidates in {time.time()-t0:.2f}s")

        # 3. Use the model to refine the candidates (main bottleneck)
        logger.info(f"[PRISM] Step 3/5: LLM refinement (generating ~30 tokens)...")
        t0 = time.time()
        components = self._refine_with_model(concept, raw_components[:20])
        logger.info(f"[PRISM] Step 3/5: Refined to {len(components)} components in {time.time()-t0:.2f}s")

        # 4. Embed ALL components
        logger.info(f"[PRISM] Step 4/5: Embedding {len(components)} components... (GPU forward passes)")
        t0 = time.time()
        comp_vecs = [self.engine.embed(c) for c in components]
        logger.info(f"[PRISM] Step 4/5: Done in {time.time()-t0:.2f}s")

        # 5. Project to 3D
        logger.info(f"[PRISM] Step 5/5: Projecting to 3D... (CPU PCA)")
        t0 = time.time()
        all_vecs = np.vstack([center_vec.reshape(1, -1)] + [v.reshape(1, -1) for v in comp_vecs])
        coords_3d = self.engine.cartographer.project(all_vecs)
        logger.info(f"[PRISM] Step 5/5: Done in {time.time()-t0:.2f}s")
        
        logger.info(f"[PRISM] Total time: {time.time()-total_start:.2f}s")

        # 7. Build Points
        points = []

        # Center (Prism source) - the original concept
        points.append(VectorPoint(
            label=concept,
            coords_3d=coords_3d[0].tolist(),
            metadata={"type": "source", "size": 2.5}
        ))

        # Components (Spectral lines) - refined semantic relatives
        for i, comp in enumerate(components):
            points.append(VectorPoint(
                label=comp,
                coords_3d=coords_3d[i + 1].tolist(),
                metadata={
                    "type": "component",
                    "rank": i + 1,
                    "distance": float(np.linalg.norm(comp_vecs[i] - center_vec))
                }
            ))

        # Connections: Source (index 0) to all components
        connections = [(0, i + 1) for i in range(len(components))]

        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Prism analysis of '{concept}': Found {len(components)} semantic components",
            metadata={
                "concept": concept,
                "raw_candidates": raw_components[:15],
                "refined_components": components
            }
        )

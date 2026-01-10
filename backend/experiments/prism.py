"""
Concept Prism Experiment

Uses the "Reverse Embedding" technique to spectrally analyze a concept
by projecting its input embedding back onto the vocabulary.
"""

from typing import Any, Dict
import numpy as np

from .base import Experiment
from ..models import ExperimentResult, ConceptPrismInput, VectorPoint


class ConceptPrismExperiment(Experiment):
    """
    The Concept Prism: Spectrally analyze a concept using reverse embedding.

    This experiment takes the input embedding of a concept and projects it
    back onto the vocabulary (using the LM Head) to find its constituent
    semantic components (nearest neighbors in latent space).
    """

    name = "prism"
    description = "Spectrally analyze a concept using reverse embedding"
    input_model = ConceptPrismInput

    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)

        concept = validated.concept

        # 1. Get the embedding of the concept
        # We use the engine's embed method which gets the contextual embedding
        # But for "Prism", we arguably want the *static* input embedding or just the
        # contextual one. Engine.embed gives contextual (last layer) or specific layer.
        # However, as found in our exploration, Input Embedding (layer 0) often works best
        # for simple synonym lookup.
        # But `engine.embed` defaults to last layer.
        # Let's try to capture the *first* layer activations if possible,
        # or just use the standard embedding which is a robust representation.

        # Actually, let's stick to the engine.embed (last layer) first.
        # If that's too abstract, we might need to expose layer-0 embedding.
        # BUT, in my exploration `explore_activations.py`, I saw that
        # "Input Embedding Nearest Neighbors" (Method 1) was best.
        # "Method 1" used `wte(token)`.

        # `engine.embed` does: `forward_with_cache` -> `activations[last_layer]`.
        # We might want `activations[layer_0]`.

        # Let's request Layer 0 explicitly
        trace = self.engine.trace_thought(concept, layers=[0])
        # activations is dict {layer_name: tensor}
        # We need to find layer 0

        # Find the key for layer 0
        layer_0_key = "layer_0"
        if layer_0_key in trace["activations"]:
            # [batch, seq, hidden]
            # Take mean over sequence
            act = trace["activations"][layer_0_key]
            center_vec = act.mean(dim=1).squeeze(0).cpu().numpy()
        else:
            # Fallback to standard embed (last layer)
            center_vec = self.engine.embed(concept)

        # 2. Unembed to find components
        # This is the "Prism" effect - splitting the white light (vector) into colors (words)
        components = self.engine.unembed(center_vec, top_k=20)

        # Filter: Remove the concept itself if it appears
        components = [c for c in components if c.lower() != concept.lower()]
        components = components[:15] # Take top 15

        # 3. Embed components for visualization
        comp_vecs = [self.engine.embed(c) for c in components]

        # 4. Project to 3D
        all_vecs = np.vstack([center_vec.reshape(1, -1)] + [v.reshape(1, -1) for v in comp_vecs])
        coords_3d = self.engine.cartographer.project(all_vecs)

        # 5. Build Points
        points = []

        # Center (Prism source)
        points.append(VectorPoint(
            label=concept,
            coords_3d=coords_3d[0].tolist(),
            metadata={"type": "source", "size": 2.0}
        ))

        # Components (Spectral lines)
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

        # Connections: Source to all components
        connections = [(0, i + 1) for i in range(len(components))]

        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Prism analysis of '{concept}': Found {len(components)} spectral components",
            metadata={
                "concept": concept,
                "components": components
            }
        )

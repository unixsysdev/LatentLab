"""
Supernova Experiment

Explodes a concept into its semantic dimensions and finds the anti-concept.
"""

from typing import Any, Dict
import numpy as np

from .base import Experiment
from ..models import ExperimentResult, SupernovaInput, VectorPoint


class SupernovaExperiment(Experiment):
    """
    The Concept Supernova: Explode a concept into orthogonal dimensions.
    
    This visualizes Mike's "72 variables" idea - showing how the model
    sees a simple concept as a high-dimensional starburst of attributes.
    """
    
    name = "supernova"
    description = "Explode a concept into semantic dimensions and find its opposite"
    input_model = SupernovaInput
    
    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)
        
        concept = validated.concept
        num_attributes = validated.num_attributes
        
        # Get center embedding
        center_vec = self.engine.embed(concept)
        
        # Ask LLM to generate semantic attributes
        attr_prompt = f"""List {num_attributes} distinct, high-level semantic attributes or associations of the concept "{concept}".

Examples might include: cultural meaning, physical properties, emotional associations, historical significance, etc.

Return ONLY a comma-separated list of short attribute phrases (2-4 words each).
No numbering, no explanations."""

        try:
            attributes_str = self.engine.generate(
                attr_prompt,
                max_tokens=200,
                temperature=0.8
            )
            attributes = [x.strip().strip('"\'') for x in attributes_str.split(',')]
            attributes = [a for a in attributes if a and len(a) < 50][:num_attributes]
        except Exception as e:
            attributes = [f"Attribute {i+1}" for i in range(num_attributes)]
        
        # Pad if needed
        while len(attributes) < num_attributes:
            attributes.append(f"Dimension {len(attributes)+1}")
        
        # Embed all attributes
        attr_vecs = [self.engine.embed(a) for a in attributes]
        
        # Calculate the "Anti-Concept" (semantic opposite)
        anti_prompt = f"""What is the exact conceptual opposite of "{concept}"?

Consider: if "{concept}" has certain qualities, what concept completely lacks or inverts ALL of them?

Respond with ONLY the concept name (1-5 words), nothing else."""

        try:
            anti_label = self.engine.generate(
                anti_prompt,
                max_tokens=20,
                temperature=0.6
            ).strip().split('\n')[0].strip('"\'')[:50]
        except Exception:
            anti_label = f"Anti-{concept}"
        
        anti_vec = self.engine.embed(anti_label)
        
        # Mathematical anti-vector (for comparison)
        math_anti_vec = center_vec * -1
        
        # Combine all vectors for projection
        all_vecs = np.vstack([
            center_vec.reshape(1, -1),
            np.array(attr_vecs),
            anti_vec.reshape(1, -1)
        ])
        
        coords_3d = self.engine.cartographer.project(all_vecs)
        
        # Build points
        points = []
        
        # Center point (the concept)
        points.append(VectorPoint(
            label=concept,
            coords_3d=coords_3d[0].tolist(),
            metadata={"type": "center", "size": 2.0}
        ))
        
        # Attribute points (the rays)
        for i, attr in enumerate(attributes):
            points.append(VectorPoint(
                label=attr,
                coords_3d=coords_3d[i + 1].tolist(),
                metadata={
                    "type": "attribute",
                    "index": i,
                    "distance": float(np.linalg.norm(attr_vecs[i] - center_vec))
                }
            ))
        
        # Anti-concept point
        points.append(VectorPoint(
            label=f"ANTI: {anti_label}",
            coords_3d=coords_3d[-1].tolist(),
            metadata={
                "type": "anti",
                "distance": float(np.linalg.norm(anti_vec - center_vec))
            }
        ))
        
        # Connections: center to each attribute (starburst pattern)
        connections = [(0, i + 1) for i in range(len(attributes))]
        
        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Semantic explosion of '{concept}' into {len(attributes)} dimensions",
            metadata={
                "concept": concept,
                "anti_concept": anti_label,
                "num_attributes": len(attributes)
            }
        )

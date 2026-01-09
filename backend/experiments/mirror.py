"""
Mirror Experiment

Maps relationship structure from one domain onto another.
"""

from typing import Any, Dict, List
import numpy as np

from .base import Experiment
from ..models import ExperimentResult, MirrorInput, VectorPoint


class MirrorExperiment(Experiment):
    """
    The Structure Mirror: Map relationships across domains.
    
    This demonstrates Mike's "cross-domain linkage" - showing how
    the model sees identical structure in vastly different topics.
    
    Example: Map "Rome Rise → Rome Peak → Rome Fall" onto "Dubstep Track"
    """
    
    name = "mirror"
    description = "Map relationship structure from one domain onto another"
    input_model = MirrorInput
    
    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        validated = self.validate_inputs(inputs)
        
        source_chain = validated.source_chain
        target_domain = validated.target_domain
        
        # Embed source chain
        source_vecs = [self.engine.embed(s) for s in source_chain]
        
        # Calculate deltas (the "shape" of the progression)
        deltas = []
        for i in range(len(source_vecs) - 1):
            deltas.append(source_vecs[i + 1] - source_vecs[i])
        
        # Ask LLM for equivalent starting point in target domain
        start_prompt = f"""In the domain of "{target_domain}", what would be the equivalent of "{source_chain[0]}"?

The source domain shows this progression: {' → '.join(source_chain)}

What is the starting point of an equivalent progression in "{target_domain}"?
Respond with ONLY the concept (2-6 words), nothing else."""

        try:
            target_start = self.engine.generate(
                start_prompt,
                max_tokens=30,
                temperature=0.6
            ).strip().split('\n')[0].strip('"\'')[:60]
        except Exception:
            target_start = f"{target_domain} Start"
        
        # Build target chain by asking LLM to continue the pattern
        target_chain = [target_start]
        target_vecs = [self.engine.embed(target_start)]
        
        for i in range(len(deltas)):
            # Ask for next step that mirrors the source transition
            next_prompt = f"""We are mapping this progression: {' → '.join(source_chain)}
onto the domain of "{target_domain}".

In the source, we went from "{source_chain[i]}" to "{source_chain[i+1]}".
In the target, we are currently at "{target_chain[-1]}".

What comes next in the target domain, mirroring the same type of transition?
Respond with ONLY the next concept (2-6 words), nothing else."""

            try:
                next_label = self.engine.generate(
                    next_prompt,
                    max_tokens=30,
                    temperature=0.6
                ).strip().split('\n')[0].strip('"\'')[:60]
            except Exception:
                next_label = f"{target_domain} Step {i+2}"
            
            target_chain.append(next_label)
            target_vecs.append(self.engine.embed(next_label))
        
        # Project all to 3D
        all_vecs = np.vstack(source_vecs + target_vecs)
        coords_3d = self.engine.cartographer.project(all_vecs)
        
        n = len(source_chain)
        points = []
        
        # Source chain points
        for i, label in enumerate(source_chain):
            points.append(VectorPoint(
                label=label,
                coords_3d=coords_3d[i].tolist(),
                metadata={
                    "type": "source",
                    "domain": "source",
                    "step": i,
                    "side": "left"
                }
            ))
        
        # Target chain points (offset in visualization)
        for i, label in enumerate(target_chain):
            points.append(VectorPoint(
                label=label,
                coords_3d=coords_3d[n + i].tolist(),
                metadata={
                    "type": "target",
                    "domain": target_domain,
                    "step": i,
                    "side": "right"
                }
            ))
        
        # Connections
        connections = []
        
        # Internal chain connections (solid lines)
        for i in range(n - 1):
            connections.append((i, i + 1))  # Source chain
            connections.append((n + i, n + i + 1))  # Target chain
        
        # Mirror connections between equivalent positions (dashed in UI)
        for i in range(n):
            connections.append((i, n + i))
        
        return ExperimentResult(
            experiment_type=self.name,
            points=points,
            connections=connections,
            description=f"Mapping '{' → '.join(source_chain)}' onto '{target_domain}'",
            metadata={
                "source_chain": source_chain,
                "target_chain": target_chain,
                "target_domain": target_domain
            }
        )

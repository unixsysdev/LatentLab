"""
Base Experiment Class

Abstract base for all LatentLab experiments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from ..models import ExperimentResult, VectorPoint


class Experiment(ABC):
    """
    Abstract base class for experiments.
    
    Each experiment defines:
    - name: Unique identifier
    - description: Human-readable description
    - input_model: Pydantic model for validation
    - run(): Execute the experiment
    """
    
    name: str = "base"
    description: str = "Base experiment"
    input_model: Type[BaseModel] = BaseModel
    
    def __init__(self, engine: Any):
        """
        Initialize with a LatentSpaceEngine.
        
        Args:
            engine: The LatentSpaceEngine instance
        """
        self.engine = engine
        
    @abstractmethod
    async def run(self, inputs: Dict[str, Any]) -> ExperimentResult:
        """
        Execute the experiment.
        
        Args:
            inputs: Validated input dictionary
            
        Returns:
            ExperimentResult with points and connections
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> BaseModel:
        """Validate inputs against the input model"""
        return self.input_model(**inputs)
    
    def _make_point(
        self,
        label: str,
        coords_3d: List[float],
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        layer: Optional[int] = None
    ) -> VectorPoint:
        """Helper to create a VectorPoint"""
        return VectorPoint(
            label=label,
            coords_3d=coords_3d,
            vector=vector,
            metadata=metadata or {},
            layer=layer
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Return experiment metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema()
        }

"""
Experiment Registry

Discovers and manages available experiments.
"""

from typing import Dict, Type, Optional, List, Any

from .base import Experiment


class ExperimentRegistry:
    """
    Registry for discovering and instantiating experiments.
    """
    
    _experiments: Dict[str, Type[Experiment]] = {}
    
    @classmethod
    def register(cls, experiment_cls: Type[Experiment]) -> Type[Experiment]:
        """Register an experiment class (can be used as decorator)"""
        cls._experiments[experiment_cls.name] = experiment_cls
        return experiment_cls
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Experiment]]:
        """Get experiment class by name"""
        return cls._experiments.get(name)
    
    @classmethod
    def list_all(cls) -> List[Dict[str, str]]:
        """List all registered experiments"""
        return [
            {
                "name": exp_cls.name,
                "description": exp_cls.description
            }
            for exp_cls in cls._experiments.values()
        ]
    
    @classmethod
    def create(cls, name: str, engine: Any) -> Optional[Experiment]:
        """Create an experiment instance"""
        exp_cls = cls.get(name)
        if exp_cls is None:
            return None
        return exp_cls(engine)
    
    @classmethod
    def get_schema(cls, name: str) -> Optional[Dict]:
        """Get input schema for an experiment"""
        exp_cls = cls.get(name)
        if exp_cls is None:
            return None
        return exp_cls.input_model.model_json_schema()


# Auto-register all experiments
def _register_all():
    from .wormhole import WormholeExperiment
    from .supernova import SupernovaExperiment
    from .mirror import MirrorExperiment
    from .steering import SteeringExperiment
    from .prism import ConceptPrismExperiment
    from .blackhole import BlackholeExperiment
    
    ExperimentRegistry.register(WormholeExperiment)
    ExperimentRegistry.register(SupernovaExperiment)
    ExperimentRegistry.register(MirrorExperiment)
    ExperimentRegistry.register(SteeringExperiment)
    ExperimentRegistry.register(ConceptPrismExperiment)
    ExperimentRegistry.register(BlackholeExperiment)


_register_all()

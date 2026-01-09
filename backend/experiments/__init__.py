"""
LatentLab Experiments Package
"""

from .base import Experiment, ExperimentResult
from .registry import ExperimentRegistry

__all__ = ['Experiment', 'ExperimentResult', 'ExperimentRegistry']

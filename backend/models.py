"""
Pydantic models for LatentLab API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class ExperimentType(str, Enum):
    WORMHOLE = "wormhole"
    SUPERNOVA = "supernova"
    MIRROR = "mirror"
    STEERING = "steering"
    LIVE_TRACE = "live_trace"


class VectorPoint(BaseModel):
    """A point in semantic space with label and coordinates"""
    label: str
    vector: Optional[List[float]] = None  # Full vector (optional, can be large)
    coords_3d: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    metadata: Dict[str, Any] = Field(default_factory=dict)
    layer: Optional[int] = None  # Which layer this came from


class ExperimentResult(BaseModel):
    """Result from running an experiment"""
    experiment_type: str
    points: List[VectorPoint]
    connections: List[Tuple[int, int]] = Field(default_factory=list)
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Experiment Inputs ---

class WormholeInput(BaseModel):
    """Input for Wormhole experiment"""
    start: str = Field(..., description="Starting concept")
    end: str = Field(..., description="Ending concept")
    steps: int = Field(default=7, ge=3, le=20, description="Number of interpolation steps")


class SupernovaInput(BaseModel):
    """Input for Supernova experiment"""
    concept: str = Field(..., description="Concept to explode")
    num_attributes: int = Field(default=15, ge=5, le=30)


class MirrorInput(BaseModel):
    """Input for Mirror experiment"""
    source_chain: List[str] = Field(..., min_length=2, max_length=10)
    target_domain: str


class SteeringInput(BaseModel):
    """Input for Steering experiment"""
    prompt: str
    positive_concept: str = Field(..., description="Concept to steer towards")
    negative_concept: str = Field(..., description="Concept to steer away from")
    layer: int = Field(default=15, ge=1, description="Layer to inject at")
    strength: float = Field(default=1.0, ge=0.0, le=10.0)


class LiveTraceInput(BaseModel):
    """Input for live thought tracing"""
    prompt: str
    max_tokens: int = Field(default=20, ge=1, le=100)


# --- API Models ---

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    experiments_available: List[str]


class ExperimentListResponse(BaseModel):
    experiments: List[Dict[str, str]]


class TokenGeneration(BaseModel):
    """WebSocket message for token generation"""
    type: str = "token_generation"
    token: str
    layer_path: List[List[float]]  # [layer][x,y,z]
    token_index: int

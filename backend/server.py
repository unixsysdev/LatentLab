"""
LatentLab API Server

FastAPI server with REST and WebSocket endpoints.
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    HealthResponse,
    ExperimentListResponse,
    ExperimentResult,
    WormholeInput,
    SupernovaInput,
    ConceptPrismInput,
    MirrorInput,
    SteeringInput,
    LiveTraceInput,
    TokenGeneration
)
from .engine import get_engine, LatentSpaceEngine
from .model_loader import AVAILABLE_MODELS
from .experiments.registry import ExperimentRegistry
from .visualizers.static import StaticVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine - loaded at startup
engine: Optional[LatentSpaceEngine] = None
visualizer = StaticVisualizer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events - load model ONCE here"""
    global engine
    logger.info("Starting LatentLab server...")
    logger.info("Loading model (this may take a minute)...")
    
    # Load model at startup - only once!
    engine = get_engine()
    logger.info("Model loaded successfully!")
    
    yield
    
    logger.info("Shutting down LatentLab server...")


app = FastAPI(
    title="LatentLab",
    description="Local LLM latent space visualization",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_load_engine() -> LatentSpaceEngine:
    """Get the global engine (already loaded at startup)"""
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, server still starting")
    return engine


# --- REST Endpoints ---

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        eng = get_or_load_engine()
        return HealthResponse(
            status="ok",
            model=eng.model_name,
            device=eng.device,
            experiments_available=[e["name"] for e in ExperimentRegistry.list_all()]
        )
    except Exception as e:
        return HealthResponse(
            status="loading",
            model="loading...",
            device="unknown",
            experiments_available=[]
        )


@app.get("/api/experiments", response_model=ExperimentListResponse)
async def list_experiments():
    """List available experiments"""
    return ExperimentListResponse(
        experiments=ExperimentRegistry.list_all()
    )


@app.get("/api/models")
async def list_models():
    """List available models for switching"""
    try:
        eng = get_or_load_engine()
        current_model = eng.model_name
    except:
        current_model = None
    
    return {
        "models": AVAILABLE_MODELS,
        "current": current_model
    }


@app.post("/api/models/switch")
async def switch_model(body: dict):
    """Switch to a different model"""
    global engine
    model_id = body.get("model_id")
    
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    
    # Check if model is in available list
    valid_models = [m["id"] for m in AVAILABLE_MODELS]
    if model_id not in valid_models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    
    logger.info(f"Switching to model: {model_id}")
    
    # Reset the cached engine
    from .engine import reset_engine
    reset_engine()
    
    # Free old model
    if engine is not None:
        del engine
        engine = None
    
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load new model (get_engine will create fresh since we reset)
    engine = get_engine(model_name=model_id)
    
    logger.info(f"Successfully loaded model: {model_id}")
    return {"status": "ok", "model": model_id}


@app.get("/api/experiment/{name}/schema")
async def get_experiment_schema(name: str):
    """Get input schema for an experiment"""
    schema = ExperimentRegistry.get_schema(name)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    return schema


@app.post("/api/experiment/wormhole", response_model=ExperimentResult)
async def run_wormhole(inputs: WormholeInput):
    """Run the Wormhole experiment"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create("wormhole", eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Wormhole experiment not found")
    return await experiment.run(inputs.model_dump())


@app.post("/api/experiment/supernova", response_model=ExperimentResult)
async def run_supernova(inputs: SupernovaInput):
    """Run the Supernova experiment"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create("supernova", eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Supernova experiment not found")
    return await experiment.run(inputs.model_dump())


@app.post("/api/experiment/prism", response_model=ExperimentResult)
async def run_prism(inputs: ConceptPrismInput):
    """Run the Concept Prism experiment"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create("prism", eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Prism experiment not found")
    return await experiment.run(inputs.model_dump())


@app.post("/api/experiment/mirror", response_model=ExperimentResult)
async def run_mirror(inputs: MirrorInput):
    """Run the Mirror experiment"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create("mirror", eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Mirror experiment not found")
    return await experiment.run(inputs.model_dump())


@app.post("/api/experiment/steering", response_model=ExperimentResult)
async def run_steering(inputs: SteeringInput):
    """Run the Steering experiment"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create("steering", eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Steering experiment not found")
    return await experiment.run(inputs.model_dump())


@app.post("/api/experiment/{name}", response_model=ExperimentResult)
async def run_experiment(name: str, inputs: dict):
    """Run any experiment by name"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create(name, eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    
    try:
        return await experiment.run(inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Visualization Endpoints ---

@app.post("/api/visualize/{name}")
async def visualize_experiment(name: str, inputs: dict):
    """Run experiment and return PNG visualization"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create(name, eng)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    
    try:
        result = await experiment.run(inputs)
        image_bytes = visualizer.visualize(result)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/visualize/wormhole")
async def visualize_wormhole(
    start: str = Query(...),
    end: str = Query(...),
    steps: int = Query(default=7)
):
    """Visualize wormhole experiment as PNG"""
    eng = get_or_load_engine()
    experiment = ExperimentRegistry.create("wormhole", eng)
    result = await experiment.run({"start": start, "end": end, "steps": steps})
    image_bytes = visualizer.visualize(result)
    return Response(content=image_bytes, media_type="image/png")


# --- WebSocket for Live Thought Tracing ---

@app.websocket("/ws/live")
async def websocket_live_trace(websocket: WebSocket):
    """
    WebSocket endpoint for real-time thought tracing.
    
    Client sends: {"prompt": "...", "max_tokens": 20}
    Server sends: TokenGeneration messages as tokens are generated
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 20)
            
            if not prompt:
                await websocket.send_json({"error": "No prompt provided"})
                continue
            
            eng = get_or_load_engine()
            
            # Trace the initial prompt
            trace = eng.trace_thought(prompt)
            
            # Send initial state
            await websocket.send_json({
                "type": "prompt_trace",
                "tokens": trace["tokens"],
                "trajectories": {
                    str(k): v for k, v in trace["trajectories"].items()
                }
            })
            
            # Generate tokens one by one (simplified - actual streaming is complex)
            # For full streaming, you'd need to hook into the generation loop
            current_text = prompt
            for i in range(max_tokens):
                # Generate next token
                new_text = eng.generate(current_text, max_tokens=1)
                if not new_text:
                    break
                
                current_text += new_text
                
                # Trace the new state
                trace = eng.trace_thought(current_text)
                
                # Send the new token
                await websocket.send_json({
                    "type": "token_generation",
                    "token": new_text,
                    "token_index": i,
                    "layer_path": trace["trajectories"].get(len(trace["tokens"]) - 1, [])
                })
                
                await asyncio.sleep(0.1)  # Small delay for visualization
            
            await websocket.send_json({"type": "generation_complete"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# --- Run Server ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

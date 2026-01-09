# LatentLab

<div align="center">
  <video src="media/demo.mp4" width="100%" controls autoplay loop muted></video>
  <br/>
  <em>A 3D visualization of Large Language Model latent spaces</em>
</div>

Local LLM latent space visualization tool for exploring how thoughts evolve in transformer models.

## Features

- **Local Model Inference**: Qwen3-4B-Instruct-2507 (easy to swap to larger models)
- **4 Experiments**:
  - 🌀 **Wormhole**: Semantic trajectory between distant concepts
  - 💥 **Supernova**: Explode a concept into semantic dimensions
  - 🪞 **Mirror**: Map relationship structures across domains
  - 🧭 **Steering**: Inject activation vectors to alter model behavior
- **3D Interactive Visualization**: React + Three.js
- **Static Plots**: Matplotlib PNG exports

## Quick Start

### Backend (requires ROCm toolbox)

```bash
cd backend
pip install -r ../requirements.txt
python -m backend.server
```

Server starts at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API Endpoints

- `GET /api/health` - Check server status
- `GET /api/experiments` - List available experiments
- `POST /api/experiment/wormhole` - Run wormhole experiment
- `POST /api/experiment/supernova` - Run supernova experiment  
- `POST /api/experiment/mirror` - Run mirror experiment
- `POST /api/experiment/steering` - Run steering experiment
- `GET /api/visualize/wormhole?start=X&end=Y` - Get matplotlib PNG
- `WS /ws/live` - Real-time thought tracing

## Changing Models

Edit `backend/model_loader.py`:

```python
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Change this
```

Tested models:
- `Qwen/Qwen3-4B-Instruct-2507` (default, ~8GB VRAM)
- `Qwen/Qwen3-8B-Instruct-2507` (~16GB VRAM)
- `Qwen/Qwen3-14B-Instruct-2507` (~28GB VRAM)

## Project Structure

```
semantic-viz/
├── backend/
│   ├── model_loader.py      # Model with activation hooks
│   ├── latent_cartographer.py # PCA/UMAP projection
│   ├── engine.py            # Main engine
│   ├── server.py            # FastAPI server
│   ├── models.py            # Pydantic models
│   ├── experiments/
│   │   ├── base.py          # Abstract experiment
│   │   ├── wormhole.py      # Concept interpolation
│   │   ├── supernova.py     # Feature explosion
│   │   ├── mirror.py        # Cross-domain mapping
│   │   ├── steering.py      # Activation steering
│   │   └── registry.py      # Experiment discovery
│   └── visualizers/
│       └── static.py        # Matplotlib plots
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app
│   │   ├── components/
│   │   │   ├── ThoughtVisualizer.jsx
│   │   │   ├── ExperimentForm.jsx
│   │   │   └── ResultsPanel.jsx
│   │   └── index.css        # Dark theme
│   └── package.json
└── requirements.txt
```

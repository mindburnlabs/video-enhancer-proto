# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: Next‑Gen AI Video Enhancer (Gradio UI + SOTA model handlers)

Common commands

- Create a virtualenv and install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For a leaner setup (e.g., CI or CPU-only dev):
# pip install -r requirements-minimal.txt
```

- Run the Gradio UI (local, defaults to port 7860)

```bash
python app.py
# Open http://localhost:7860
```

- Quick analysis (no heavy models): run the Degradation Router on a sample video

```bash
python models/analysis/degradation_router.py path/to/input.mp4
```

- Run tests

```bash
# Install test tooling if missing
pip install pytest

# All tests
python -m pytest -q
# Single file
python -m pytest tests/test_sota_models_comprehensive.py -q
# Single test via expression
python -m pytest -k "VSRMHandler and initialization" -q
```

- Lint, format, and type‑check (optional dev tooling)

```bash
pip install black isort flake8 mypy
# Format
black . && isort .
# Lint
flake8 .
# Type-check (no config required by default)
mypy .
```

Environment toggles

- Device selection in UI: app.py chooses CUDA when the CUDA_AVAILABLE env var is set, else CPU.

```bash
# Request CUDA if available
export CUDA_AVAILABLE=1
python app.py
```

Architecture overview

- User interface (app.py)
  - Gradio Blocks UI exposes a single “enhance” flow. On load it attempts to initialize a VideoEnhancerSOTAAgent (agents/enhancer) and processes uploads via a TaskSpecification describing target fps, quality tier, latency class, etc. Results include metadata (primary_model, quality_score, beats_topaz flag) for the summary panel.

- Analysis and routing (models/analysis/degradation_router.py)
  - Samples frames and computes fused metrics: compression (block-DCT energy), motion blur (Laplacian variance), low light (brightness + histogram), noise (blur residual), temporal inconsistency (frame diffs), and simple face prominence (Haar cascade).
  - Produces a routing plan with SOTA model selection among: vsrm, seedvr2, ditvr, fast_mamba_vsr based on unknown-degradation score, motion complexity, and latency class. Also toggles pre/post experts (compression cleanup, denoising, face restoration, temporal consistency) and whether to apply HFR interpolation.

- SOTA model handlers (models/enhancement/*)
  - vsr/vsrm_handler.py: Mamba-based super-resolution backbone. Sliding window + tile processing; returns enhanced frames at scale factor. Uses simplified deformable alignment and temporal Mamba blocks.
  - diffusion/seedvr2_handler.py: One-step diffusion restoration with temporal attention/flow fusion and quality-aware conditioning. Processes windows conditionally based on measured segment quality.
  - zeroshot/ditvr_handler.py: Transformer restoration using 3D patching, degradation-conditioned adaptive layer norm, and meta-adaptation for zero-shot scenarios; reconstructs frames from patches.

- Interpolation (models/interpolation/enhanced_rife_handler.py)
  - Prototype wrapper intended to fetch and run RIFE. Not used by the UI path directly; may attempt to clone an external repo on first use.

- Agents (agents/*)
  - Coordinator and related agent scaffolding exist (e.g., agents/coordinator/coordinator_agent.py) but reference external dependencies and pipeline components that are not included here. Treat these as prototypes/skeletons for future orchestration—focus development on UI + model handlers + router in this repo.

Notes and pitfalls

- Avoid outdated commands from README or prior docs that reference server.py, pipeline/topaz_killer_pipeline.py, Dockerfile, or docker-compose.yml—those artifacts are not present in this repository.
- The UI path (app.py) relies on agents/enhancer/video_enhancer_sota.py and TaskSpecification from agents.*; some agent abstractions may be incomplete. If the UI fails to initialize the agent, validate core modules by running the Degradation Router and individual handlers in isolation.
- Model weights are not bundled; handlers initialize randomly unless you provide weights. Quality/visual outputs from prototypes are for development and may not reflect final model performance.

# One-Click AI Live2D Asset Compiler

Phase 1 MVP for compiling a single character illustration into a Live2D-oriented
asset package.

## Quick Start

1. Put `master.png`, `master.jpg`, or `master.webp` into `input/`.
2. Run:

```powershell
python app.py build
```

If your default Python does not have Pillow installed, use the bundled Codex
runtime or install the dependencies from `requirements.txt`.

The build writes a unique run directory under `output/` with:

- original source copy
- 2x upscaled image
- coarse decomposition layers
- contact sheet
- material PNGs
- `manifest.json`
- PSD files
- HTML report

The default configuration is cloud-first. Heavy reasoning runs through
OpenRouter, and Qwen Image Layered decomposition is expected to run on Vast.ai.
The deterministic local fallback still exists for development, but it is
disabled by default.

## Commands

```powershell
python app.py build
python app.py build --input input/master.png
python app.py build --quality standard
python app.py prepare-vast
python app.py validate-run <run_id>
python app.py archive-run <run_id>
python app.py resume <run_id>
```

## Cloud-First Flow

First publish this project as a GitHub repository and let GitHub Actions build
the Vast.ai Docker image:

```powershell
git init
git add .
git commit -m "Initial Live2D compiler MVP"
git branch -M main
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

Then set `vast.docker_image` in `config.json` to:

```text
ghcr.io/<owner>/live2d-asset-compiler:latest
```

Use:

```powershell
python app.py prepare-vast
```

This creates `output/<run_id>/vast/` and `output/<run_id>.zip`. Upload the zip
to a Vast.ai instance running the GitHub-built image, then run:

```bash
python -m src.remote.worker --run-zip /workspace/<run_id>.zip --preflight
python -m src.remote.worker --run-zip /workspace/<run_id>.zip
```

The worker creates `/workspace/<run_id>_done.zip`. Download and extract it over
`output/<run_id>/`, then run:

```powershell
python app.py validate-run <run_id>
python app.py resume <run_id>
```

## Phase 1 Scope

Implemented:

- input validation
- run management
- source preservation
- faithful 2x upscale
- OpenRouter analysis adapter
- Qwen-layered adapter interface with Vast.ai handoff
- contact sheet generation
- basic Live2D-oriented manifest
- material PNG export
- PSD export
- HTML report
- stage metadata for resume-ready runs
- remote artifact validation

Not implemented yet:

- real Qwen Image Layered inference
- segmentation models
- hidden-area reconstruction
- advanced validation and QA metrics
- one-click web UI

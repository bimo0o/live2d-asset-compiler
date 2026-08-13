from __future__ import annotations

import json
from pathlib import Path

from src.schemas.config import AppConfig


def write_vast_job_package(run_dir: Path, config: AppConfig) -> list[Path]:
    package_dir = run_dir / "vast"
    package_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_readme(package_dir, config),
        _write_requirements(package_dir),
        _write_remote_runner(package_dir),
        _write_job_config(package_dir, config),
        _write_cli_notes(package_dir, config),
    ]
    return files


def _write_readme(package_dir: Path, config: AppConfig) -> Path:
    path = package_dir / "README_VAST.md"
    path.write_text(
        f"""# Vast.ai Decomposition Job

This run is cloud-first. Do not run Qwen Image Layered on the local machine.

Use the Docker image built from this GitHub repository:

```text
{config.vast.docker_image}
```

Upload `output/{package_dir.parent.name}.zip` to the Vast.ai instance, then run:

```bash
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --preflight
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --plan-only
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip
```

Expected output contract:

```text
decomposition/raw/layer_00_<semantic>.png
decomposition/raw/layer_01_<semantic>.png
...
decomposition/layers.json
decomposition/preview.png
```

The worker writes:

```text
/workspace/{package_dir.parent.name}_done.zip
```

Download that `*_done.zip`, extract it over `output/{package_dir.parent.name}/`,
then resume locally:

```powershell
python app.py resume {package_dir.parent.name}
```

The generated remote runner uses the Qwen Image Layered pipeline contract.
Keep the output paths unchanged so the local compiler can continue after the
run directory is downloaded back.

If the image is private, make the GHCR package public or run Docker login on
the Vast instance before starting the container.
""",
        encoding="utf-8",
    )
    return path


def _write_requirements(package_dir: Path) -> Path:
    path = package_dir / "requirements-vast.txt"
    root_requirements = Path("requirements-vast.txt")
    if root_requirements.exists():
        path.write_text(root_requirements.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        path.write_text("pillow>=10.0\nnumpy>=2.0\n", encoding="utf-8")
    return path


def _write_job_config(package_dir: Path, config: AppConfig) -> Path:
    path = package_dir / "vast_job.json"
    path.write_text(
        json.dumps(
            {
                "docker_image": config.vast.docker_image,
                "min_vram_gb": config.vast.min_vram_gb,
                "disk_gb": config.vast.disk_gb,
                "workdir": config.vast.workdir,
                "model": config.decomposition.model,
                "layers": config.decomposition.layers,
                "qwen_resolution": config.vast.qwen_resolution,
                "qwen_steps": config.vast.qwen_steps,
                "reconstruction_model": config.vast.reconstruction_model,
                "reconstruction_resolution": config.vast.reconstruction_resolution,
                "reconstruction_steps": config.vast.reconstruction_steps,
                "max_reconstruction_tasks": config.vast.max_reconstruction_tasks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_cli_notes(package_dir: Path, config: AppConfig) -> Path:
    path = package_dir / "vast_cli_notes.md"
    path.write_text(
        f"""# Vast.ai CLI Notes

Install and authenticate:

```bash
pip install vastai
export VAST_API_KEY="..."
vastai show user
```

Search for a suitable instance:

```bash
vastai search offers 'gpu_ram>={config.vast.min_vram_gb} disk_space>={config.vast.disk_gb} verified=true rentable=true'
```

Create one from an offer id:

```bash
vastai create instance <offer_id> --image {config.vast.docker_image} --disk {config.vast.disk_gb} --ssh --direct --label live2d-qwen-layered
```

Upload the prepared zip, then run:

```bash
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --preflight
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --plan-only
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip
```

If the GHCR package is private, create a GitHub personal access token with
package read permission and configure Docker login on the Vast instance.

The compiler does not store secrets in config files. Keep `OPENROUTER_API_KEY`
and `VAST_API_KEY` in environment variables or the provider's secret store.
""",
        encoding="utf-8",
    )
    return path


def _write_remote_runner(package_dir: Path) -> Path:
    path = package_dir / "run_remote_decomposition.py"
    path.write_text(
        '''from __future__ import annotations

from src.remote.decompose import main


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
    return path

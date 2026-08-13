from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.orchestrator import Pipeline
from src.schemas.config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click AI Live2D asset compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a new Live2D material package")
    build.add_argument("--config", default="config.json", help="Path to config.json")
    build.add_argument("--input", default=None, help="Optional explicit input image")
    build.add_argument(
        "--quality",
        choices=["beginner", "standard", "high"],
        default="standard",
        help="Material planning preset",
    )

    prepare = subparsers.add_parser("prepare-vast", help="Validate input and prepare a Vast.ai job package")
    prepare.add_argument("--config", default="config.json", help="Path to config.json")
    prepare.add_argument("--input", default=None, help="Optional explicit input image")
    prepare.add_argument(
        "--quality",
        choices=["beginner", "standard", "high"],
        default="standard",
        help="Material planning preset",
    )

    resume = subparsers.add_parser("resume", help="Resume an existing run")
    resume.add_argument("run_id")
    resume.add_argument("--config", default="config.json", help="Path to config.json")

    validate = subparsers.add_parser("validate-run", help="Validate returned Vast artifacts for a run")
    validate.add_argument("run_id")
    validate.add_argument("--config", default="config.json", help="Path to config.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.load(Path(args.config))
    pipeline = Pipeline(config)

    try:
        if args.command == "build":
            run = pipeline.build(input_override=Path(args.input) if args.input else None, quality=args.quality)
        elif args.command == "prepare-vast":
            run = pipeline.prepare_vast(input_override=Path(args.input) if args.input else None, quality=args.quality)
        elif args.command == "validate-run":
            run = pipeline.validate_run(args.run_id)
        else:
            run = pipeline.resume(args.run_id)
    except RuntimeError as exc:
        print(f"Build stopped: {exc}")
        return 2

    print(f"Run complete: {run.run_id}")
    print(f"Output: {run.run_dir}")
    report = run.run_dir / "reports" / "report.html"
    vast_package = run.run_dir / "vast"
    if report.exists():
        print(f"Report: {report}")
    elif vast_package.exists():
        print(f"Vast job package: {vast_package}")
        print(f"Status: {run.info.status}")
    return 0

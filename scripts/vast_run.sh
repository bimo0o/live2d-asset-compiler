#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-/workspace/live2d_compiler/run}"
cd "$RUN_DIR"
python3 vast/run_remote_decomposition.py --run-dir .


#!/usr/bin/env bash
set -euo pipefail

REPO_RAW="${LIVE2D_COMPILER_RAW_BASE:-https://raw.githubusercontent.com/bimo0o/live2d-asset-compiler/main}"
PROJECT_DIR="${LIVE2D_COMPILER_DIR:-/workspace/live2d_compiler}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

download() {
  local rel="$1"
  local dest="${TMP_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  curl -fsSL "${REPO_RAW}/${rel}" -o "${dest}"
  if head -n 1 "${dest}" | grep -qiE '<!doctype|<html'; then
    echo "downloaded HTML instead of ${rel}; refusing to patch" >&2
    exit 1
  fi
}

copy_checked() {
  local rel="$1"
  local src="${TMP_DIR}/${rel}"
  local dest="${PROJECT_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}"
  echo "updated ${dest}"
}

cd "${PROJECT_DIR}"

download "src/ai/openrouter.py"
download "src/pipeline/stages.py"
download "src/remote/decompose.py"
download "src/remote/worker.py"
download "src/remote/upscale.py"
download "src/cloud/vast.py"
download "src/schemas/config.py"
download "scripts/patch_torchvision.py"

python3 -m py_compile \
  "${TMP_DIR}/src/ai/openrouter.py" \
  "${TMP_DIR}/src/pipeline/stages.py" \
  "${TMP_DIR}/src/remote/decompose.py" \
  "${TMP_DIR}/src/remote/worker.py" \
  "${TMP_DIR}/src/remote/upscale.py" \
  "${TMP_DIR}/src/cloud/vast.py" \
  "${TMP_DIR}/src/schemas/config.py" \
  "${TMP_DIR}/scripts/patch_torchvision.py"

copy_checked "src/ai/openrouter.py"
copy_checked "src/pipeline/stages.py"
copy_checked "src/remote/decompose.py"
copy_checked "src/remote/worker.py"
copy_checked "src/remote/upscale.py"
copy_checked "src/cloud/vast.py"
copy_checked "src/schemas/config.py"
copy_checked "scripts/patch_torchvision.py"

python3 scripts/patch_torchvision.py || true
python3 -m py_compile src/ai/openrouter.py src/pipeline/stages.py src/remote/decompose.py src/remote/worker.py src/remote/upscale.py src/cloud/vast.py src/schemas/config.py
echo "live2d compiler server code updated"

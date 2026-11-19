#!/usr/bin/env bash
# Serve Qwen-Image 2.1 on one Gaudi2 through vLLM-Omni.
#
#   scripts/serve_qwen_image.sh /path/to/Qwen-Image-2.1 [launcher options]
#   MODEL=... scripts/serve_qwen_image.sh -- --max-num-seqs 1
#
# The launcher leases an idle module, pins its NUMA CPUs, sets the offline caches and the HPU
# environment, then runs `vllm serve --omni --diffusion-attention-backend HPU_SDPA`. Compile the
# DiT blocks with COMPILE=1 (first request per shape pays for it), and keep the cache root on a
# writable filesystem with CACHE_DIR. See docs/features/qwen_image_21.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${1:-}}"
if [ -z "$MODEL" ]; then
  echo "usage: scripts/serve_qwen_image.sh /path/to/Qwen-Image-2.1 [launcher options]" >&2
  exit 2
fi
shift || true

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || { echo "missing $PY: install the environment first (docs/features/qwen_image_21.md)" >&2; exit 1; }

# The Gaudi Python needs the habanalabs library path; the plugin's own launcher forces the rest.
export PYTHONPATH="${PYTHONPATH:-/usr/lib/habanalabs/}"
export PYTHONNOUSERSITE=1
export VLLM_GAUDI_OMNI_HPU_COMPILE="${COMPILE:-0}"
export VLLM_GAUDI_LOCK_DIR="${VLLM_GAUDI_LOCK_DIR:-/tmp/1cat-gaudi-module-locks}"

CACHE="${CACHE_DIR:-${DATA_DIR:-$HOME/.cache/gaudi2-qwen-image}}"
exec "$PY" "$ROOT/vllm-gaudi/tools/qwen_image_2_1/serve_single_hpu.py" "$MODEL" \
  --cache-dir "$CACHE" "$@"

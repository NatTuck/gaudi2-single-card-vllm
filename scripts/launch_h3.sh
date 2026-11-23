#!/usr/bin/env bash
# Run the MiniMax-H3 port on one Gaudi2 with per-block HPU graphs (ping-pong buffers).
#
#   scripts/launch_h3.sh --bucket draft --seed 42
#   scripts/launch_h3.sh --bucket high --seed 42 --out /path/to/high.mp4
#
# Every run_t2va.py argument passes through. The port checkout is read-only in this workspace, so
# the graph strategy and the resolution ladder are applied as a runtime patch from
# vllm-gaudi/tools/minimax_h3/block_graphs.py: set H3_PORT_DIR to the checkout's `hpu_port`
# directory. Block graphs are on by default; VLLM_GAUDI_H3_BLOCK_GRAPHS=0 restores the port's own
# wrapping (whole-DiT graph, which OOMs at capture above ~1280x704).
#
# The lazy-mode environment below is not optional: the graph classes refuse to exist outside lazy
# mode, and the recipe cache is what makes a warm request skip recompilation. Keep RECIPES_DIR on
# a writable filesystem with real space (the r768 recipe set is tens of GB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${H3_PORT_DIR:-}" ]; then
  PORT="$H3_PORT_DIR"
else
  PORT="${1:-}"
  shift || true
fi
if [ -z "$PORT" ]; then
  echo "usage: H3_PORT_DIR=/path/to/hpu_port scripts/launch_h3.sh [run_t2va options]" >&2
  exit 2
fi
if [ ! -f "$PORT/run_t2va.py" ]; then
  echo "H3_PORT_DIR=$PORT has no run_t2va.py; point it at the port's hpu_port directory" >&2
  exit 2
fi

PY="${PYTHON:-$ROOT/.venv032/bin/python}"
[ -x "$PY" ] || { echo "missing $PY: install the vLLM 0.32 environment first (docs/features/minimax_h3.md)" >&2; exit 1; }

# Caches must be writable. $HOME/.cache is not, in a workspace where the home tree is read-only, so
# fall back to a cache next to the checkout rather than dying on a mkdir.
WORKSPACE="$(cd "$ROOT/.." && pwd)"
CACHE_ROOT="${CACHE_DIR:-$HOME/.cache}"
if ! mkdir -p "$CACHE_ROOT" 2>/dev/null || [ ! -w "$CACHE_ROOT" ]; then
  CACHE_ROOT="$WORKSPACE/.cache"
  mkdir -p "$CACHE_ROOT"
fi
RECIPES="${RECIPES_DIR:-${H3_RECIPES_DIR:-$CACHE_ROOT/gaudi2-h3-recipes}}"
mkdir -p "$RECIPES"

# Gaudi environment: lazy mode is required for HPU graphs, autoload pulls in the eager bridge, and
# the recipe cache is keyed on the graph shape so every ladder rung gets its own entry.
export HABANA_PLUGINS_LIB_PATH="${HABANA_PLUGINS_LIB_PATH:-/opt/habanalabs/habana_plugins}"
export HABANA_SCAL_BIN_PATH="${HABANA_SCAL_BIN_PATH:-/opt/habanalabs/engines_fw}"
export HABANA_LOGS="${HABANA_LOGS:-/var/log/habana_logs/}"
export GC_KERNEL_PATH="${GC_KERNEL_PATH:-/usr/lib/habanalabs/libtpc_kernels.so}"
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"
export PT_HPU_AUTOLOAD="${PT_HPU_AUTOLOAD:-1}"
export H3_RECIPES_DIR="$RECIPES"
export PT_HPU_RECIPE_CACHE_CONFIG="${PT_HPU_RECIPE_CACHE_CONFIG:-$RECIPES,false,8192,false}"
# The Gaudi Python needs the habanalabs library path; PYTHONNOUSERSITE keeps a user-site package
# from shadowing the venv's (a stale `peft`/`diffusers` in ~/.local silently wins otherwise).
export PYTHONPATH="${PYTHONPATH:-/usr/lib/habanalabs/}"
export PYTHONNOUSERSITE=1
# The port muxes with ffmpeg; a workspace without a system ffmpeg keeps a static build in
# <workspace>/vendor/bin. An existing PATH entry still wins.
export PATH="${FFMPEG_DIR:-$WORKSPACE/vendor/bin}:$PATH"

export H3_PORT_DIR="$PORT"
cd "$PORT"
# The port's own default repo path (/root/src/h3/MiniMax-H3) is the author's checkout. When the
# port ships next to its weights, point the runner at them instead of failing on a missing
# model_index.json.
REPO_ARGS=()
if [ -n "${H3_REPO:-}" ]; then
  REPO_ARGS=(--repo "$H3_REPO")
elif [ -f "$(dirname "$PORT")/MiniMax-H3/FL2VA/model_index.json" ]; then
  REPO_ARGS=(--repo "$(dirname "$PORT")/MiniMax-H3")
fi
# The port's embed cache defaults to <port>/.ecache, which is read-only in this workspace.
EMBED_CACHE="${EMBED_CACHE_DIR:-$CACHE_ROOT/gaudi2-h3-embed}"
mkdir -p "$EMBED_CACHE"
exec "$PY" "$ROOT/vllm-gaudi/tools/minimax_h3/block_graphs.py" --te-threads "${TE_THREADS:-8}" \
  "${REPO_ARGS[@]}" --embed-cache-dir "$EMBED_CACHE" "$@"

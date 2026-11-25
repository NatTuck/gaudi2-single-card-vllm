#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-block HPU graphs with ping-pong buffers, plus a same-aspect-ratio resolution ladder.

Why this exists
---------------
`wrap_in_hpu_graph` pins every graph's user inputs *and* its user outputs
(`mark_user_inputs` / `mark_user_outputs`), because the runtime cannot know that a caller will
keep a reference to them. Wrapping the whole MiniMax-H3 DiT as one graph is therefore fine (two
tensors pinned), but wrapping each of its 50 transformer blocks separately pins 50 input tensors
plus 50 output tensors. At 1280x704 one hidden-states tensor is ~360 MB, so per-block wrapping
reserved ~32 GiB and the capture died at 94.9/96 GiB. The blocks form a chain - block i's output
is block i+1's input - so the 100 tensors only ever need to be two. Each block's forward copies its
output into one of two shared buffers (even blocks write A, odd blocks write B) and returns it; the
buffers are allocated from block 0's input by a forward pre-hook, i.e. before any wrapped forward
runs, so the allocation stays outside the captured graph. Measured on one Gaudi2 at 1280x704/124
frames/8 steps: whole-DiT graph OOMs at capture, per-block graphs with ping-pong fit in 82-88 GiB
and generate in 293 s cold / 224 s warm.

This assumes nothing retains a block output across calls. Nothing in the released DiT does (the
last block's buffer is consumed by the output norm inside the same forward, and the next denoising
step rebuilds the hidden states from scratch), but a block-caching policy such as cache-dit would
store every block's output across steps and would get two aliased tensors back instead.

What the copying costs: 50 blocks x (360 MB read + 360 MB write) = 36 GB per step, which at the
card's ~2.0 TB/s DMA envelope for tensors this size is ~18 ms, plus 50 replay boundaries at ~0.5 ms
is ~25 ms - together ~0.15% of a 28 s step (224 s warm / 8 steps at 1280x704). The chain is
dataflow-serial, so a per-block graph boundary gives up no cross-engine overlap; the only added
traffic is the copies. Worth checking once on the card that `copy_` lands on the DMA engine rather
than a TPC kernel - eager `copy_` is a TPC kernel, and "compiled, therefore DMA" is a claim about
the recipe, not a given.

One hazard for reuse: this wrapper is generic, but a graph bakes Python scalars as constants. The
diffusers block takes tensors only, so nothing is baked today. The vLLM-Omni block takes
`max_seqlen` / `packed_total` / `num_requests` as Python ints, and a replay with a different value
would silently reuse the old literal - the pre-hook only compares the hidden-states shape, so it
would not catch that. Thread those as 0-d tensors, or assert the captured ints on replay, before
pointing this at the Omni transformer.

One shape per wrapped transformer
--------------------------------
The buffers and the captured graphs are both shape-baked. The pre-hook runs on every call, so it is
the only place that can notice a shape change: it allocates once and raises on any later mismatch
rather than reallocating, because a reallocation cannot redirect a graph that already captured the
old buffers. The port wraps the DiT once at boot, so this is the contract it already has.

The graph-capture guard
-----------------------
The port rejects any request whose packed sequence exceeds `H3_MAX_GRAPH_SEQ` under CP=1 with
graphs on, because a *whole-DiT* capture's host-side IR grows with the sequence (an r768/192f
capture ballooned process RSS to 86.6 GB and was OOM-killed). Per-block capture bounds the IR by
the largest single block, so the guard's premise does not hold in this mode, and `install()` lifts
the cap. Note the guard lives in the port's *server* request path: a single-shot run
(`--bucket ...`) never evaluates it, so the lift is inert there. It matters only for
`H3_SERVER_MODE`.

The resolution ladder
---------------------
Three canvases that share one aspect ratio exactly, so every rung composites into the same output
frame without cropping or letterboxing. Both edges must be multiples of 32 (the DiT patch stride)
and the area must stay under the released 768x1344 cap; at 7:4 every rung satisfies both with no
rounding:

    tier     canvas      pixels    rows/frame   seq @124f (budget 512 / 2048)
    draft    448x256     0.115M    112           5070 / 6606
    medium   896x512     0.459M    448          17502 / 19038
    high     1344x768    1.032M   1008          38222 / 39758

(budget 512 is `buckets.json`'s default; the port's CLI defaults `--text-budget` to 2048.)

This does NOT make a draft a preview of the high render. The initial noise is drawn at the latent
shape, so the same seed gives a different shot at every rung (measured: three different
compositions of the same prompt). What is preserved is the frame geometry, not the content.

Exact 16:9 cannot do this: 32-alignment forces 512k x 288k, and the third rung (1536x864) is 28%
over the area cap. 7:4 is also what the released policy itself returns for a 16:9 request at short
edge 768 (`resolve_minimax_h3_output_canvas` scales to 1365x768, caps the area, then rounds each
edge to 32), so the high rung here is bit-identical to the stock canvas.

Usage
-----
    H3_PORT_DIR=/path/to/hpu_port python block_graphs.py --bucket draft --seed 42
    H3_PORT_DIR=... python block_graphs.py --bucket high --seed 42

Every `run_t2va.py` argument passes through. `scripts/launch_h3.sh` wraps this with the HPU
environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import Any

import torch
from torch import nn

# The released canvas plus the two lower rungs. Order is draft -> high.
LADDER_ASPECT_RATIO = "7:4"
LADDER_ASPECT = 7.0 / 4.0
BLOCK_GRAPHS_ENV = "VLLM_GAUDI_H3_BLOCK_GRAPHS"
MAX_GRAPH_SEQ_ENV = "H3_MAX_GRAPH_SEQ"
# Per-block capture bounds host IR by the largest single block, so the whole-DiT sequence budget
# does not apply. This is a sentinel, not a measurement: it is far above any shape that fits in
# the card's 96 GiB, so the device OOM (or a clean fit) decides, not an arbitrary host-side cap.
BLOCK_GRAPH_SEQ_BUDGET = 65536


@dataclass(frozen=True)
class LadderTier:
    """One rung: a canvas that shares LADDER_ASPECT with the other rungs, plus its step count."""

    name: str
    width: int
    height: int
    steps: int

    @property
    def rows_per_frame(self) -> int:
        return self.width * self.height // 1024


LADDER_TIERS: tuple[LadderTier, ...] = (
    LadderTier("draft", 448, 256, 8),
    LadderTier("medium", 896, 512, 16),
    LadderTier("high", 1344, 768, 30),
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _ladder_resolutions() -> dict[str, dict[str, Any]]:
    """Resolution-table rows for the ladder, keyed the way `buckets.json` keys them."""

    rows: dict[str, dict[str, Any]] = {}
    for tier in LADDER_TIERS:
        rows[f"r{tier.height}"] = {
            "hw": [tier.width, tier.height],
            "rows_per_frame": tier.rows_per_frame,
        }
    return rows


def _ladder_buckets() -> list[dict[str, Any]]:
    """Named bring-up buckets, one per rung, so `--bucket draft` needs no shape arguments."""

    return [
        {
            "name": tier.name,
            "width": tier.width,
            "height": tier.height,
            "seconds": 5,
            "num_inference_steps": tier.steps,
            "notes": (
                f"7:4 ladder rung {index + 1}/{len(LADDER_TIERS)} "
                f"({tier.width * tier.height / 1e6:.3f} Mpx, {tier.rows_per_frame} rows/frame); "
                "shares one aspect ratio with the other rungs"
            ),
        }
        for index, tier in enumerate(LADDER_TIERS)
    ]


_BUFFERS: list[torch.Tensor | None] = [None, None]
# What the buffers (and therefore the captured graphs) were built for.
_ALLOCATED: tuple[tuple[int, ...], torch.dtype, torch.device] | None = None


def _ping_pong_forward(original_forward: Any, slot: int) -> Any:
    """Build a block forward that copies its output into the shared buffer for `slot`.

    The block is wrapped in place rather than nested, so parameter names keep their original paths
    (`transformer_blocks.0.attn...`, not `transformer_blocks.0.block.attn...`). Anything that walks
    `named_parameters()` - a hot-loaded adapter, an offload/restore - would otherwise see a
    different tree than the one the checkpoint was loaded into.
    """

    @wraps(original_forward)
    def forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = original_forward(hidden_states, *args, **kwargs)
        buf = _BUFFERS[slot]
        if buf is None:
            # Capture-time only: the body of this forward is what the graph traces, so nothing
            # here runs again on replay. The live per-call check is the pre-hook.
            raise RuntimeError(
                f"ping-pong buffer {slot} is not allocated; the pre-hook on the first block did "
                "not run, so the allocation would have landed inside the captured graph"
            )
        buf.copy_(out)
        return buf

    return forward


def _prepare_buffers(module: nn.Module, args: tuple[Any, ...]) -> None:
    """Allocate both shared buffers from the first block's input, once per wrapped transformer.

    Runs as a forward pre-hook on block 0, so it executes before the graph capture/replay that lives
    in `block.forward`. That is what keeps the allocation outside the captured graph, and it is
    also the only code that runs on every call - hence the shape check lives here.
    """

    hidden_states = args[0] if args else None
    if not isinstance(hidden_states, torch.Tensor):
        return None
    global _ALLOCATED
    key = (tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device)
    if _ALLOCATED is None:
        for slot in range(2):
            _BUFFERS[slot] = torch.empty_like(hidden_states)
        _ALLOCATED = key
        return None
    if key != _ALLOCATED:
        raise RuntimeError(
            "per-block HPU graphs bake the shared ping-pong buffers and the captured shape, so a "
            f"wrapped DiT serves one shape per process: captured {_ALLOCATED}, this call is {key}. "
            "Reallocating cannot redirect a graph that already captured the old buffers. Run one "
            f"shape per process (the port wraps the DiT once at boot) or set {BLOCK_GRAPHS_ENV}=0 "
            "to fall back to whole-DiT wrapping."
        )
    return None


def reset_buffers() -> None:
    """Drop the shared buffers and the shape they were built for (tests, and a re-wrap)."""

    global _ALLOCATED
    _BUFFERS[0] = None
    _BUFFERS[1] = None
    _ALLOCATED = None


def _resolve_graph_wrapper() -> Any:
    """Return `wrap_in_hpu_graph`, mirroring the port's own preference for its graphs shim."""

    try:
        import importlib

        shim = importlib.import_module("hpu_port.graphs")
        apply = getattr(shim, "wrap_in_hpu_graph", None)
        if apply is not None:
            return apply
    except Exception:
        pass
    try:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    except Exception:
        return None
    return wrap_in_hpu_graph


def wrap_transformer_graphs_pingpong(original: Any) -> Any:
    """Build the `wrap_transformer_graphs(pipe, transformer)` replacement."""

    def wrap_transformer_graphs(pipe: Any, transformer: Any) -> Any:
        if not _env_bool(BLOCK_GRAPHS_ENV, True):
            return original(pipe, transformer)

        apply = _resolve_graph_wrapper()
        if apply is None:
            print(
                "[h3] WARNING: wrap_in_hpu_graph unavailable; staying in eager lazy mode",
                flush=True,
            )
            return transformer

        blocks = list(transformer.transformer_blocks)
        already = [index for index, block in enumerate(blocks) if getattr(block, "_h3_ping_pong", False)]
        if already:
            raise RuntimeError(
                f"transformer blocks {already} are already wrapped with ping-pong buffers; "
                "wrapping again would nest the graphs and register a second pre-hook"
            )

        import habana_frameworks.torch.core as htcore

        htcore.mark_step()
        reset_buffers()
        for index, block in enumerate(blocks):
            # In place: `wrap_in_hpu_graph` replaces `module.forward` and returns the same module,
            # so the block stays exactly where it was in the parameter tree.
            block.forward = _ping_pong_forward(block.forward, index % 2)
            block._h3_ping_pong = True
            applied = apply(block, disable_tensor_cache=True)
            if applied is not block:
                # The Habana plugin mutates `module.forward` in place and returns the same module.
                # This loop relies on that, so a plugin that returns a wrapper instead would leave
                # the blocks running eager with shared buffers while the log below claims graphs.
                raise RuntimeError(
                    f"wrap_in_hpu_graph returned {type(applied).__name__} instead of the block "
                    "itself; this wrapper keeps each block in place, so a returned wrapper would "
                    "silently drop the graph"
                )
        blocks[0].register_forward_pre_hook(_prepare_buffers)
        print(
            f"[h3] DiT wrapped per-block with ping-pong buffers: {len(blocks)} block graphs, "
            "2 shared hidden-state tensors (disable_tensor_cache=True)",
            flush=True,
        )
        return transformer

    return wrap_transformer_graphs


def install(runner: ModuleType) -> dict[str, str]:
    """Patch a loaded `run_t2va` module in place. Returns the state of each patch."""

    state: dict[str, str] = {}

    original_wrap = getattr(runner, "wrap_transformer_graphs", None)
    if original_wrap is None:
        state["block_graphs"] = "skipped: run_t2va has no wrap_transformer_graphs"
    else:
        runner.wrap_transformer_graphs = wrap_transformer_graphs_pingpong(original_wrap)
        state["block_graphs"] = "installed" if _env_bool(BLOCK_GRAPHS_ENV, True) else "disabled"

    original_load = getattr(runner, "load_buckets", None)
    if original_load is None:
        state["resolution_ladder"] = "skipped: run_t2va has no load_buckets"
    else:

        def load_buckets(path: Any = None) -> dict:
            table = original_load(path)
            resolutions = table.setdefault("resolutions", {})
            # setdefault per row: the released table is the authority for any row it already has
            # (r768 is the high rung and already exists), so a future upstream change wins.
            for name, row in _ladder_resolutions().items():
                resolutions.setdefault(name, row)
            buckets = table.setdefault("buckets", [])
            known = {bucket.get("name") for bucket in buckets}
            buckets.extend(bucket for bucket in _ladder_buckets() if bucket["name"] not in known)
            return table

        runner.load_buckets = load_buckets
        state["resolution_ladder"] = "installed"

    if _env_bool(BLOCK_GRAPHS_ENV, True):
        current = os.environ.get(MAX_GRAPH_SEQ_ENV)
        try:
            explicit = int(current) if current is not None else None
        except ValueError:
            raise SystemExit(
                f"{MAX_GRAPH_SEQ_ENV}={current!r} is not an integer; the port would fail on it later"
            ) from None
        if explicit is None or explicit < BLOCK_GRAPH_SEQ_BUDGET:
            os.environ[MAX_GRAPH_SEQ_ENV] = str(BLOCK_GRAPH_SEQ_BUDGET)
            state["graph_seq_guard"] = (
                f"lifted to {BLOCK_GRAPH_SEQ_BUDGET} for the port's server mode (per-block capture "
                "bounds host IR by the largest single block, so the whole-DiT sequence budget does "
                "not apply); single-shot runs never evaluate this guard"
            )
        else:
            state["graph_seq_guard"] = f"left at {explicit} (server mode only)"
    else:
        state["graph_seq_guard"] = "left alone (block graphs disabled)"

    return state


def load_runner(port_dir: str | os.PathLike[str] | None = None) -> ModuleType:
    """Import `run_t2va` from the port checkout, which must be on `sys.path` first."""

    directory = port_dir or os.environ.get("H3_PORT_DIR")
    if directory:
        resolved = os.fspath(directory)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        os.chdir(resolved)
    import run_t2va  # noqa: PLC0415 - resolved from the port checkout above

    return run_t2va


def main(argv: list[str] | None = None) -> int:
    runner = load_runner()
    for name, outcome in install(runner).items():
        print(f"[h3] patch {name}: {outcome}", flush=True)
    return int(runner.main(argv) or 0)


if __name__ == "__main__":
    sys.exit(main())

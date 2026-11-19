#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease one idle Gaudi module and serve Qwen-Image 2.1 through vLLM-Omni.

The diffusion models never share the card with the 27B service (README, "Image
and video generation"): this launcher takes the module lock, checks the card is
idle, pins the NUMA CPUs, and only then starts ``vllm serve --omni``.

    tools/qwen_image_2_1/serve_single_hpu.py /path/to/Qwen-Image-2.1 --module 6
    tools/qwen_image_2_1/serve_single_hpu.py MODEL -- --max-num-seqs 1

Environment:
    VLLM_GAUDI_LOCK_DIR        shared device-lock directory (default /tmp/1cat-gaudi-module-locks)
    VLLM_GAUDI_OMNI_HPU_COMPILE=1   regional compile of the DiT blocks (opt-in, default off)
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import IO, Any

_DEFAULT_LOCK_DIR = Path(os.environ.get("VLLM_GAUDI_LOCK_DIR", "/tmp/1cat-gaudi-module-locks"))
_DEFAULT_WARMUP_PROMPTS = (
    "A photograph of a red ceramic teapot on a wooden table, soft window light.",
    "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement.",
    "An isometric illustration of a small library with floating books, warm palette.",
)


def _parse_cpuset(value: str) -> set[int]:
    result: set[int] = set()
    for field in value.strip().split(","):
        if not field:
            continue
        bounds = [int(item) for item in field.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _accelerators() -> list[dict[str, Any]]:
    devices = []
    for accelerator in sorted(Path("/sys/class/accel").glob("accel[0-9]*")):
        device = accelerator / "device"
        try:
            devices.append({
                "module": int((device / "module_id").read_text()),
                "numa": int((device / "numa_node").read_text()),
                "bus": device.resolve().name,
                "node": Path("/dev/accel") / accelerator.name,
            })
        except (FileNotFoundError, ValueError):
            continue
    return devices


def _device_is_idle(device: Mapping[str, Any]) -> bool:
    owner = subprocess.run(["fuser", str(device["node"])], capture_output=True, text=True)
    if owner.returncode != 1 or owner.stdout.strip() or owner.stderr.strip():
        return False
    status = subprocess.run(
        [
            "hl-smi",
            "-i",
            device["bus"],
            "--query-aip=memory.used,utilization.aip",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory_mib, utilization = (float(field.strip()) for field in status.split(","))
    return memory_mib <= 1024 and utilization == 0


def _try_lease(lock_dir: Path, requested_module: int | None) -> tuple[dict[str, Any], IO[str]] | None:
    for device in _accelerators():
        if requested_module is not None and device["module"] != requested_module:
            continue
        lock_path = lock_dir / f"gaudi-module{device['module']}.lock"
        stream = lock_path.open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _device_is_idle(device):
                return device, stream
        except BlockingIOError:
            pass
        stream.close()
    return None


def _checkpoint(model: Path) -> Path:
    model = model.expanduser().resolve()
    index = model / "model_index.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    config = json.loads(index.read_text(encoding="utf-8"))
    arch = config.get("_class_name")
    if arch != "QwenImage21Pipeline":
        raise ValueError(f"{index} declares {arch!r}; this launcher serves QwenImage21Pipeline")
    return model


def _command(args: argparse.Namespace, model: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--num-gpus",
        "1",
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--diffusion-attention-backend",
        "HPU_SDPA",
        # vLLM-Omni compiles repeated blocks with inductor unless told otherwise; the HPU Omni
        # platform reports no inductor support, and the regional compile below is the Gaudi path.
        "--enforce-eager",
    ]
    if args.vae_use_tiling:
        command.append("--vae-use-tiling")
    return command + args.vllm_args


def _post_request(url: str, payload: dict[str, Any], timeout: float) -> bool:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"[warmup] request failed: {error}", flush=True)
        return False


def _warmup(args: argparse.Namespace) -> None:
    """Compile the DiT/text-encoder/VAE graphs before the first user request."""
    health = f"http://{args.host}:{args.port}/health"
    deadline = time.monotonic() + args.warmup_timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2.0)
    else:
        print("[warmup] server did not become healthy; skipping warm-up", flush=True)
        return

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    for index, prompt in enumerate(_DEFAULT_WARMUP_PROMPTS[:args.warmup_requests]):
        started = time.monotonic()
        ok = _post_request(
            url,
            {
                "model": str(args.model),
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "extra_body": {
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.steps,
                    "seed": 42 + index,
                    "true_cfg_scale": 1.0,
                },
            },
            timeout=args.warmup_timeout,
        )
        print(f"[warmup] {index + 1}/{args.warmup_requests} {'ok' if ok else 'failed'} "
              f"({time.monotonic() - started:.1f} s)", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="local Qwen-Image 2.1 diffusers-layout directory")
    parser.add_argument("--module", type=int, help="physical module ID; otherwise lease the first idle module")
    parser.add_argument("--lock-dir", type=Path, default=_DEFAULT_LOCK_DIR)
    parser.add_argument("--cpu-affinity", help="taskset-style CPU list; defaults to the selected module's NUMA node")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("VLLM_GAUDI_CACHE_DIR", os.path.expanduser("~/.cache"))),
                        help="writable cache root for vLLM-Omni scratch state (default ~/.cache)")
    parser.add_argument("--plugins",
                        default=os.environ.get("VLLM_PLUGINS", "hpu,01.hpu_custom_utils"),
                        help="vLLM plugin entry points to load: the HPU platform plus its utility "
                        "hooks; the LLM groups (ops/models/tool parsers) are excluded by default")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--vae-use-tiling", action=argparse.BooleanOptionalAction, default=False,
                        help="tile the VAE decode (needed above about 1536^2)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="regional HPU compile of the DiT blocks (VLLM_GAUDI_OMNI_HPU_COMPILE)")
    parser.add_argument("--width", type=int, default=1024, help="warm-up width")
    parser.add_argument("--height", type=int, default=1024, help="warm-up height")
    parser.add_argument("--steps", type=int, default=4, help="warm-up denoising steps")
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-timeout", type=float, default=3600.0)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", default=True)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        separator = raw_args.index("--")
        own_args, vllm_args = raw_args[:separator], raw_args[separator + 1:]
    else:
        own_args, vllm_args = raw_args, []
    args = parser.parse_args(own_args)
    args.vllm_args = vllm_args
    return args


def main() -> int:
    args = _parse_args()
    args.model = _checkpoint(args.model)
    args.lock_dir.mkdir(parents=True, exist_ok=True)

    lease = _try_lease(args.lock_dir, args.module)
    while lease is None:
        target = f"module {args.module}" if args.module is not None else "an idle module"
        print(f"Waiting for {target}; existing jobs remain untouched", flush=True)
        time.sleep(args.poll_seconds)
        lease = _try_lease(args.lock_dir, args.module)
    device, lock = lease

    if args.cpu_affinity:
        affinity = _parse_cpuset(args.cpu_affinity)
    else:
        affinity = _parse_cpuset(Path(f"/sys/devices/system/node/node{device['numa']}/cpulist").read_text())
    affinity &= os.sched_getaffinity(0)
    if not affinity:
        lock.close()
        raise RuntimeError("selected CPU affinity is empty")

    environment = dict(os.environ)
    cache_dir = args.cache_dir.expanduser().resolve()
    for sub in ("vllm", "vllm-omni/speakers", "triton", "inductor"):
        (cache_dir / sub).mkdir(parents=True, exist_ok=True)
    environment.update(
        HABANA_VISIBLE_MODULES=str(device["module"]),
        HLS_MODULE_ID=str(device["module"]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PT_HPU_LAZY_MODE="0",
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        # Image/video only: load the HPU platform and its utility hooks, not the LLM plugin
        # groups (ops, models, tool parsers). Those exist for the text models and their imports
        # follow vLLM internals that move between releases.
        VLLM_PLUGINS=args.plugins,
        VLLM_GAUDI_SCOPE="omni",
        # vLLM-Omni writes a few scratch paths under the cache root; keep them off a read-only home.
        XDG_CACHE_HOME=str(cache_dir),
        VLLM_CACHE_ROOT=str(cache_dir / "vllm"),
        SPEAKER_SAMPLES_DIR=str(cache_dir / "vllm-omni" / "speakers"),
        TRITON_CACHE_DIR=str(cache_dir / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache_dir / "inductor"),
    )
    compile_enabled = args.compile if args.compile is not None else os.environ.get(
        "VLLM_GAUDI_OMNI_HPU_COMPILE", "0") == "1"
    environment["VLLM_GAUDI_OMNI_HPU_COMPILE"] = "1" if compile_enabled else "0"

    command = _command(args, args.model)
    print(
        json.dumps(
            {
                "model": str(args.model),
                "module": device["module"],
                "bus": device["bus"],
                "numa": device["numa"],
                "cpu_affinity": sorted(affinity),
                "command": command,
                "regional_compile": compile_enabled,
                "vae_use_tiling": bool(args.vae_use_tiling),
                "warmup": bool(args.warmup),
                "huggingface_network_disabled": True,
            },
            indent=2,
        ),
        flush=True,
    )
    os.sched_setaffinity(0, affinity)

    process: subprocess.Popen[bytes] | None = None

    def forward(signum, _frame):
        if process is not None:
            os.killpg(process.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    try:
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        if args.warmup:
            threading.Thread(target=_warmup, args=(args, ), daemon=True).start()
        return process.wait()
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

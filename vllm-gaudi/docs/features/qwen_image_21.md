# Qwen-Image 2.1 on Gaudi

Qwen-Image 2.1 text-to-image serving on one Gaudi2 through vLLM-Omni and this plugin
(`vllm_gaudi.omni`). The launcher lives in
[`tools/qwen_image_2_1/serve_single_hpu.py`](../../tools/qwen_image_2_1/serve_single_hpu.py).

## Qualified dependency set

| Dependency | Version used here |
| --- | --- |
| Gaudi driver / firmware | 1.24.2 |
| PyTorch | 2.13.0a0 (Intel SDK build) |
| habana-torch-plugin | 1.24.2.481 |
| vLLM | this checkout (`0.28.1rc1.dev100`) |
| vLLM-Omni | `3dc35694b3d4` (first revision with native Qwen-Image 2.1) |
| Transformers | 5.16.1 |
| Diffusers | 0.40.0 |
| Checkpoint | `Qwen/Qwen-Image-2.1`, BF16, snapshot `b3179ad` |

The pinned vLLM-Omni revision in `setup.py`'s `omni` extra (`767cc79`) predates Qwen-Image 2.1:
its diffusion registry has no `QwenImage21Pipeline`, so the checkpoint cannot load on it. The
`omni-qwen-image21` extra installs the revision that has the model:

```bash
python -m pip install -e '.[omni-qwen-image21]'
```

Do not install both extras into one environment; they pin different vLLM-Omni revisions. The
MiniMax H3 path stays on `.[omni]`.

## Environment (what was actually built here)

`scripts/install.sh --venv .venv` cannot create the venv on a host without `ensurepip`, so the
sequence below was used with `uv`. Two traps: the `omni` install pulls a CUDA `torch` and CUDA
libraries into the venv (they shadow the Gaudi build and break `habana_frameworks`), and a
user-site `peft` shadows the venv's `diffusers`.

```bash
cd gaudi2-single-card-vllm
export UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/usr/lib/habanalabs/ PYTHONNOUSERSITE=1
uv venv --python 3.12 --system-site-packages .venv          # inherits the Gaudi torch 2.13 + plugin
uv pip install --python .venv/bin/python --no-deps -r requirements/lock-sdk-1.24.1.txt
(cd vllm-core && VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.28.1rc1.dev100+gfe755c889 \
    uv pip install --python ../.venv/bin/python --no-build-isolation --no-deps \
    --config-settings editable_mode=compat -e .)
(cd vllm-gaudi && VLLM_GAUDI_BUILD_FLASHINFER=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    uv pip install --python ../.venv/bin/python --no-build-isolation --no-deps \
    --config-settings editable_mode=compat -e .)

# vLLM-Omni with the 2.1 model, without letting it resolve torch:
uv pip install --python .venv/bin/python --no-deps \
    "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@3dc35694b3d4"
uv pip install --python .venv/bin/python \
    "transformers==5.16.1" "tokenizers>=0.23.1,<0.24" "diffusers==0.40.0" "accelerate==1.12.0" \
    imageio imageio-ffmpeg "modelscope==1.40.0" pydantic-settings msgpack pyzmq omegaconf \
    "cache-dit==1.5.0" "kernels==0.16.1" x-transformers prettytable "aenum==3.1.16" janus gguf scipy

# If the resolution above still installed a CUDA torch, remove it so the Gaudi build shows through:
rm -rf .venv/lib/python3.12/site-packages/{torch,torchgen,triton,nvidia,cuda} \
       .venv/lib/python3.12/site-packages/{torch,triton}-*.dist-info \
       .venv/lib/python3.12/site-packages/nvidia_*.dist-info
.venv/bin/python -c "import torch; print(torch.__version__, torch.hpu.is_available())"
```

`uv` also upgraded numpy to 2.5.5 while resolving the diffusion stack; the Gaudi torch build is
happy with it, so that is what is installed.

## Serving

```bash
tools/qwen_image_2_1/serve_single_hpu.py /path/to/Qwen-Image-2.1 \
    --module 6 --port 8091 --compile \
    --cache-dir ~/.cache/gaudi2-qwen-image
```

The launcher leases one idle module (`fuser` on `/dev/accel/accelN` plus an `hl-smi` idle check
under a `flock`), pins the module's NUMA CPUs, forces the Hugging Face caches offline, sets
`PT_HPU_LAZY_MODE=0`, and starts `vllm serve --omni --diffusion-attention-backend HPU_SDPA
--enforce-eager`. Warm-up requests compile the DiT graphs before the first user request; disable
them with `--no-warmup`.

Requests use the OpenAI chat endpoint, with the diffusion parameters in `extra_body`:

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen-image-2.1",
        "messages": [{"role": "user", "content": "A red ceramic teapot on a wooden table."}],
        "extra_body": {"height": 1024, "width": 1024, "num_inference_steps": 20, "seed": 42,
                       "true_cfg_scale": 1.0}
      }'
```

The image comes back as a base64 data URL in `choices[0].message.content[0].image_url.url`.

## Measured (1024², BF16, one Gaudi2)

| Configuration | 4 steps | 20 steps |
| --- | ---: | ---: |
| eager (`--no-compile`) | 3.87 s | 12.28 s |
| regional HPU compile (`--compile`) | **3.12–3.39 s** | **8.91 s** |

First request in a session is 4.8 s (eager) or 6.5 s (compile, which builds the block graphs).
The same seed is byte-identical within a configuration; eager and compiled differ by 42.4 dB
PSNR at the same seed (compilation changes the fusion order, not the result). Boot is about 20 s
with the weights in the page cache: 3.1 s to load 750 tensors, then the startup dummy run.

For reference, the snapshot's README reports 4.4 s per 1024² HTTP request on the deployment that
shipped the Qwen-Image worker separately; the numbers above come from this checkout's launcher.

## What the plugin adds

1. **Real-valued rotary embedding** — [`vllm_gaudi/omni/qwen_image_21.py`](../../vllm_gaudi/omni/qwen_image_21.py).
   The upstream rope builds complex frequency tables (`torch.polar`) and rotates with a complex
   multiply; Gaudi has no complex dtype on the device, so the first denoising step fails with
   "Complex datatype is not supported on HPU device". The patch keeps the tables real (cosine half
   followed by sine half, the same `[S, D]` contract the transformer passes around) and applies
   the rotation in FP32 before casting back. CPU parity against the stock complex path:
   `max_abs = 4.8e-7` on the rope and `1.1e-5` for a full 32-block forward in float32; the tests in
   `tests/unit_tests/omni/test_qwen_image_21.py` pin it.
2. **Import bridges** — [`vllm_gaudi/omni/compat.py`](../../vllm_gaudi/omni/compat.py).
   The 2.1 revision of vLLM-Omni imports a few vLLM names this checkout's vLLM does not have yet:
   the `configure_logging` hooks, ten `vllm.distributed` re-exports, `MultiModalProcessingResult`,
   `MultiModalDummyOptions`, `ModelOptLinearMethod`, the `vllm.v1.kv_hints` module and
   `torchaudio`. Each bridge is a no-op when the name already exists, and only import-time
   compatibility is provided: none of them is on the image path.
3. **H3 patch isolation** — `register_omni()` now installs the MiniMax H3 patches inside a
   `try/except`. They rebind internals of the pinned vLLM-Omni revision; on the 2.1 revision they
   fail (`install_minimax_h3_patches` cannot find H3's CUDA-Triton modulation bindings) and used to
   take the whole Omni plugin down with them.
4. **Regional compile and the SDPA backend** are the repository's existing ones: the 2.1
   transformer declares `_repeated_blocks = ["QwenImage21TransformerBlock"]` and
   `_dit_modules = ["transformer"]`, so `VLLM_GAUDI_OMNI_HPU_COMPILE=1` compiles its blocks and the
   attention runs through `HPUSDPAImpl` (its `additive_mask` fast path is what the block-causal
   mask uses).

## Deviations from the snapshot's pins

These are environmental, not optional:

- **transformers 5.16.1**, not the 5.14.1 of the `omni` extra: diffusers 0.40.0 imports
  `HybridCache` from transformers, which 5.14.1 does not export. 5.16.1 also satisfies vLLM
  0.28's `transformers<5.18` requirement.
- **tokenizers 0.23.x** (transformers 5.16.1 requires `>=0.23.1,<0.24`).
- **numpy 2.5.5** (resolved by the install; the SDK-1.24.1 lock pins 1.26.4, which this torch
  build does not need).
- **No torchaudio**: no wheel matches this torch build, so `compat.py` installs a stand-in. Audio
  models are out of scope here; the image path never touches it.
- **`PYTHONNOUSERSITE=1`**: a user-site `peft` (built for a newer transformers) shadows the venv
  and breaks `diffusers`. The launcher sets it.

## Limitations

- **Only text-to-image is exercised.** Image editing, multi-reference input, RGBA output and the
  2K/2048² buckets are untested here. Above about 1536² the VAE needs `--vae-use-tiling`.
- **No online FP8.** The snapshot's optimization log measured 17–26 dB PSNR against BF16 for
  Qwen-Image 2.1 with online FP8; it stays off.
- **No text-length bucketing yet.** The snapshot's worker padded text tokens to 64-token buckets
  so compiled graphs are reused across prompts. The upstream pipeline here recompiles per text
  length: with `--compile`, a prompt of a new length costs about 0.5–1.5 s extra on its first
  request (measured: 3.13 s for a length already compiled, 4.62 s then 4.00 s for a longer new
  prompt).
- **MiniMax H3 is not qualified on this vLLM-Omni revision.** Four H3 unit tests fail on it
  (the modulation-ops rebinding and the pinned schedule test); use `.[omni]` for H3.
- **One model at a time.** As in the snapshot's README, the diffusion models never share the card
  with the 27B service.

## Reproducing the numbers

```bash
# 1024^2, 4 steps, seed 42
time curl -s -o /dev/null http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","messages":[{"role":"user","content":"A red ceramic teapot on a wooden table."}],
       "extra_body":{"height":1024,"width":1024,"num_inference_steps":4,"seed":42,"true_cfg_scale":1.0}}'
```

`tools/qwen_image_2_1/serve_single_hpu.py --compile` writes the block-graph compile to the log
(`HPU regional compile: 32 blocks of ['QwenImage21TransformerBlock']`); the first request after
startup pays for it.

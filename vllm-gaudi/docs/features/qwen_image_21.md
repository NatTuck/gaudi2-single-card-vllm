# Qwen-Image 2.1 on Gaudi

Qwen-Image 2.1 text-to-image serving on one Gaudi2 through vLLM-Omni and this plugin
(`vllm_gaudi.omni`). The launcher lives in
[`tools/qwen_image_2_1/serve_single_hpu.py`](../../tools/qwen_image_2_1/serve_single_hpu.py).

This configuration serves **image and video generation only**. The LLM plugin groups (ops, models,
tool parsers) are not loaded, and the LLM runtime patches are tolerant of a vLLM that moved under
them - see [Scope](#scope-image-and-video-only).

## Qualified dependency set

| Dependency | Version |
| --- | --- |
| Gaudi driver / firmware | 1.24.2 |
| PyTorch | 2.13.0a0 (Intel SDK build) |
| habana-torch-plugin | 1.24.2.481 |
| vLLM | **v0.32.0** (vanilla upstream; the LLM patch set of the snapshot is not used here) |
| vLLM-Omni | **v0.32.0** |
| Transformers | 5.16.1 (vLLM 0.32's range is `>=5.16.1,<5.20`) |
| Tokenizers | 0.23.x |
| Diffusers | 0.40.0 |
| Checkpoint | `Qwen/Qwen-Image-2.1`, BF16, snapshot `b3179ad` |

Both projects are releases, so there is one revision of each. The snapshot's pinned vLLM-Omni
(`767cc79`) predates Qwen-Image 2.1 - its diffusion registry has no `QwenImage21Pipeline` - and the
0.31 line does not carry it either; `v0.32.0` is the first release that does.

```bash
python -m pip install -e '.[omni]'
```

## Scope: image and video only

The diffusion server loads only the HPU platform and its utility hooks:

```
VLLM_PLUGINS=hpu,01.hpu_custom_utils   # launcher default
VLLM_GAUDI_SCOPE=omni                  # launcher default
```

`VLLM_PLUGINS` is vLLM's own filter over plugin entry points, so the ops/models/tool-parser groups
are never imported. `VLLM_GAUDI_SCOPE=omni` makes `vllm_gaudi.patches.apply()` report a runtime
patch that no longer applies instead of aborting the server; without it the behaviour is unchanged
(a failing patch raises).

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
| regional HPU compile (`--compile`) | **2.80-3.07 s** | **8.13 s** |

Boot is about 25 s with the weights in the page cache. The same seed is byte-identical across runs
and across the two stack revisions (the 4-step and 20-step PNGs hash the same as the earlier
0.28/0.31 mixed install), so the move to vLLM 0.32 changed timing, not numerics.

For reference, the snapshot's README reports 4.4 s per 1024² HTTP request on the deployment that
shipped the Qwen-Image worker separately.

## What the plugin adds

1. **Real-valued rotary embedding** - [`vllm_gaudi/omni/qwen_image_21.py`](../../vllm_gaudi/omni/qwen_image_21.py).
   The upstream rope builds complex frequency tables (`torch.polar`) and rotates with a complex
   multiply; Gaudi has no complex dtype on the device, so the first denoising step fails with
   "Complex datatype is not supported on HPU device". The patch keeps the tables real (cosine half
   followed by sine half, the same `[S, D]` contract the transformer passes around) and applies the
   rotation in FP32 before casting back. CPU parity against the stock complex path:
   `max_abs = 4.8e-7` on the rope and `1.1e-5` for a full 32-block forward in float32; the tests in
   `tests/unit_tests/omni/test_qwen_image_21.py` pin it.
2. **Import bridges** - [`vllm_gaudi/omni/compat.py`](../../vllm_gaudi/omni/compat.py). vLLM-Omni
   imports ten `vllm.distributed` names that vLLM 0.32 only defines in
   `parallel_state`/`communication_op`, the `vllm.v1.kv_hints` module (absent), and `torchaudio`
   (no build matches the Gaudi torch). Each bridge is a no-op when the name already exists, and only
   import-time compatibility is provided.
3. **Regional compile and the SDPA backend** are the repository's existing ones: the 2.1
   transformer declares `_repeated_blocks = ["QwenImage21TransformerBlock"]` and
   `_dit_modules = ["transformer"]`, so `VLLM_GAUDI_OMNI_HPU_COMPILE=1` compiles its 32 blocks and
   the attention runs through `HPUSDPAImpl` (its `additive_mask` fast path is what the block-causal
   mask uses).

## Environment (what was actually built)

`scripts/install.sh --venv .venv` cannot create the venv on a host without `ensurepip`, so the
sequence below uses `uv`. Two traps: the diffusion dependency set pulls a CUDA `torch` into the venv
(it shadows the Gaudi build and breaks `habana_frameworks`), and a user-site `peft` shadows the
venv's `diffusers`.

```bash
cd gaudi2-single-card-vllm
export UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/usr/lib/habanalabs/ PYTHONNOUSERSITE=1
uv venv --python 3.12 --system-site-packages .venv032
PY=.venv032/bin/python

# vLLM v0.32.0 from the upstream tag, against the Gaudi torch
uv pip install --python $PY -r /path/to/vllm-0.32.0/requirements/common.txt
uv pip install --python $PY "setuptools>=77,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" \
    wheel jinja2 "packaging>=24.2" ninja
(cd /path/to/vllm-0.32.0 && VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0 \
    uv pip install --python $OLDPWD/$PY --no-build-isolation --no-deps \
    --config-settings editable_mode=compat -e .)
(cd vllm-gaudi && VLLM_GAUDI_BUILD_FLASHINFER=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    uv pip install --python ../$PY --no-build-isolation --no-deps \
    --config-settings editable_mode=compat -e .)

uv pip install --python $PY --no-deps \
    "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.32.0"
uv pip install --python $PY "transformers==5.16.1" "tokenizers>=0.23.2,<0.24" \
    "diffusers==0.40.0" "accelerate==1.12.0" imageio imageio-ffmpeg "modelscope==1.40.0" \
    pydantic-settings msgpack pyzmq omegaconf "cache-dit==1.5.0" "kernels==0.16.1" \
    x-transformers prettytable "aenum==3.1.16" janus gguf scipy soundfile soxr av \
    s3tokenizer openai-whisper cosmos-guardrail

# Drop any CUDA torch the resolution installed so the Gaudi build shows through:
rm -rf .venv032/lib/python3.12/site-packages/{torch,torchgen,torchvision,torchaudio,triton,nvidia,cuda} \
       .venv032/lib/python3.12/site-packages/{torch,torchvision,torchaudio,triton}-*.dist-info \
       .venv032/lib/python3.12/site-packages/{nvidia,cuda}_*.dist-info
.venv032/bin/python -c "import torch, habana_frameworks; print(torch.__version__, torch.hpu.is_available())"
```

## Limitations

- **Only text-to-image is exercised.** Image editing, multi-reference input, RGBA output and the
  2K/2048² buckets are untested here. Above about 1536² the VAE needs `--vae-use-tiling`.
- **No online FP8.** The snapshot's optimization log measured 17-26 dB PSNR against BF16 for
  Qwen-Image 2.1 with online FP8; it stays off.
- **No text-length bucketing yet.** The snapshot's worker padded text tokens to 64-token buckets so
  compiled graphs are reused across prompts. The upstream pipeline recompiles per text length: with
  `--compile`, a prompt of a new length costs about 0.5-1.5 s extra on its first request.
- **MiniMax H3 is ported but not exercised end to end here.** All ten H3 hooks install on vLLM-Omni
  v0.32.0 and the H3 unit tests pass, but the 0.32 adapter loads the *native* checkpoint layout
  (`auto_map.AutoModel`), while the snapshot on this host is the diffusers variant. A 5 s T2VA run
  needs the native ModelScope artifacts; see the H3 guide's
  [port status](../features/minimax_h3.md#vllm-omni-032-port-status).
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

`--compile` writes the block-graph compile to the log
(`HPU regional compile: 32 blocks of ['QwenImage21TransformerBlock']`); the first request after
startup pays for it.

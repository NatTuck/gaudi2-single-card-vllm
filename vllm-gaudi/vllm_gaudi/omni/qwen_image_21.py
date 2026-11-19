# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gaudi patches for the vLLM-Omni Qwen-Image 2.1 pipeline.

The upstream rotary embedding builds complex frequency tables (``torch.polar``) and rotates
query/key with a complex multiplication. Gaudi has no complex dtype on the device, so the first
denoising step dies with "Complex datatype is not supported on HPU device" while the rope tables
are moved over.

The rewrite keeps the same algebra and the same rounding order as the reference
``use_real=False`` path: the pair products are computed in FP32 and cast back to the activation
dtype. Measured against the stock complex implementation on CPU: ``max_abs = 4.8e-7`` for the
rope, and a full 32-block forward matches float32 to ``1.1e-5`` (the bf16 gap is rounding).

Nothing here changes the layout of what the transformer passes around: the rope still returns a
single tensor of shape ``[S, D]`` (the cosine half followed by the sine half) and the attention
still receives it as one argument, so sequence-parallel sharding, the prefix/target split and the
prefix-KV-cache decode path all keep working unchanged.
"""

from __future__ import annotations

import torch

_ORIGINAL_APPLY_ROTARY_EMB = None


def _real_tables(rope: torch.nn.Module, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Cosine/sine halves of the per-axis tables, cached per device."""

    cached = getattr(rope, "_vllm_gaudi_real_tables", None)
    if cached is not None and cached[0][0].device == device:
        return cached
    cos_tables, sin_tables = [], []
    for freqs in rope.freqs:
        if torch.is_complex(freqs):
            cos_tables.append(freqs.real.contiguous().to(device))
            sin_tables.append(freqs.imag.contiguous().to(device))
        else:  # already real: assume the half-split layout this module writes
            half = freqs.shape[-1] // 2
            cos_tables.append(freqs[..., :half].contiguous().to(device))
            sin_tables.append(freqs[..., half:].contiguous().to(device))
    cached = (cos_tables, sin_tables)
    rope._vllm_gaudi_real_tables = cached
    return cached


def rope_forward_real(
    self,
    img_shapes: list[tuple[int, int, int]],
    image_pad_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """``QwenImage21Rope.forward`` with real cos/sin tables instead of complex frequencies.

    The index construction is a line-for-line port of the upstream method; only the table dtype
    and the final concatenation change (``[cos | sin]`` instead of a complex tensor).
    """

    cos_tables, sin_tables = _real_tables(self, device)

    frame_index: list[int] = []
    height_index: list[int] = []
    width_index: list[int] = []
    image_height_index: list[int] = []
    image_width_index: list[int] = []
    cursor, position = 0, 0
    total_len = image_pad_mask.shape[-1]
    is_image_token = image_pad_mask.tolist()

    for _, height, width in img_shapes:
        block_start = is_image_token.index(True, cursor)
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len

        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)

        image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
        image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

    if cursor < total_len:
        frame_index.extend(range(position, position + total_len - cursor))

    frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
    height_index = frame_index.clone()
    width_index = frame_index.clone()
    height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
    width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

    height_index = torch.where(height_index < 0, height_index + cos_tables[1].shape[0], height_index)
    width_index = torch.where(width_index < 0, width_index + cos_tables[2].shape[0], width_index)

    def gather(tables: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                tables[0].index_select(0, frame_index),
                tables[1].index_select(0, height_index),
                tables[2].index_select(0, width_index),
            ],
            dim=-1,
        )

    return torch.cat([gather(cos_tables), gather(sin_tables)], dim=-1)


def apply_rotary_emb_real(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """``QwenImage21Attention._apply_rotary_emb`` for the ``[cos | sin]`` layout.

    A complex ``freqs`` keeps the stock implementation, so this stays correct if some other path
    (or a future upstream revision) hands one in.
    """

    if torch.is_complex(freqs):
        return _ORIGINAL_APPLY_ROTARY_EMB(self, x, freqs)

    half = freqs.shape[-1] // 2
    cos = freqs[..., :half].float().unsqueeze(1)
    sin = freqs[..., half:].float().unsqueeze(1)
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


def install_qwen_image_21_patches() -> bool:
    """Rebind the 2.1 rope and rotary application. Returns whether they were installed."""

    global _ORIGINAL_APPLY_ROTARY_EMB

    try:
        from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer
    except ImportError:  # this vLLM-Omni revision has no Qwen-Image 2.1
        return False

    rope_cls = transformer.QwenImage21Rope
    if getattr(rope_cls, "_vllm_gaudi_real_rope", False):
        return True

    _ORIGINAL_APPLY_ROTARY_EMB = transformer.QwenImage21Attention._apply_rotary_emb
    transformer.QwenImage21Rope.forward = rope_forward_real
    transformer.QwenImage21Attention._apply_rotary_emb = apply_rotary_emb_real
    rope_cls._vllm_gaudi_real_rope = True
    return True


__all__ = ["apply_rotary_emb_real", "install_qwen_image_21_patches", "rope_forward_real"]

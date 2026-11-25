# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity for the Qwen-Image 2.1 real-valued rotary embedding.

The patch replaces the upstream complex rope (``torch.polar`` + complex multiply) because Gaudi
has no complex dtype on the device. These tests pin the replacement to the reference algebra on
CPU, so a future upstream revision cannot silently change the layout under us.

They need no Gaudi device and no vLLM-Omni checkpoint.
"""

import pytest
import torch

from vllm_gaudi.omni.qwen_image_21 import apply_rotary_emb_real, rope_forward_real

AXES_DIM = [16, 56, 56]


class _ReferenceRope(torch.nn.Module):
    """The upstream ``QwenImage21Rope``: complex tables, complex rotation."""

    def __init__(self, theta: int = 10000, axes_dim: list[int] | None = None):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim or AXES_DIM)
        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [
            torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
            for dim in self.axes_dim
        ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, img_shapes, image_pad_mask, device):
        self.freqs = [freq.to(device) for freq in self.freqs]
        frame_index, height_index, width_index = [], [], []
        image_height_index, image_width_index = [], []
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

        height_index = torch.where(height_index < 0, height_index + self.freqs[1].shape[0], height_index)
        width_index = torch.where(width_index < 0, width_index + self.freqs[2].shape[0], width_index)
        return torch.cat(
            [
                self.freqs[0].index_select(0, frame_index),
                self.freqs[1].index_select(0, height_index),
                self.freqs[2].index_select(0, width_index),
            ],
            dim=-1,
        )


def _reference_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    paired = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(paired * freqs.unsqueeze(1)).flatten(3).to(x.dtype)


def _text_and_image_sequence(text_tokens: int, height: int, width: int) -> tuple[list, torch.Tensor]:
    """One text run followed by one image block, the shape a text-to-image step produces."""

    img_shapes = [(1, height, width)]
    mask = torch.zeros(text_tokens + height * width, dtype=torch.bool)
    mask[text_tokens:] = True
    return img_shapes, mask


def test_rope_matches_the_complex_reference():
    rope = _ReferenceRope()
    img_shapes, mask = _text_and_image_sequence(text_tokens=7, height=8, width=8)

    expected = rope(img_shapes, mask, torch.device("cpu"))
    actual = rope_forward_real(rope, img_shapes, mask, torch.device("cpu"))

    assert not torch.is_complex(actual)
    half = actual.shape[-1] // 2
    assert torch.allclose(actual[..., :half], expected.real, atol=1e-6)
    assert torch.allclose(actual[..., half:], expected.imag, atol=1e-6)


def test_rope_is_stable_across_calls():
    rope = _ReferenceRope()
    img_shapes, mask = _text_and_image_sequence(text_tokens=3, height=4, width=6)
    first = rope_forward_real(rope, img_shapes, mask, torch.device("cpu"))
    second = rope_forward_real(rope, img_shapes, mask, torch.device("cpu"))
    assert torch.equal(first, second)


def test_rotary_application_matches_the_complex_multiply():
    rope = _ReferenceRope()
    img_shapes, mask = _text_and_image_sequence(text_tokens=5, height=8, width=8)
    freqs = rope(img_shapes, mask, torch.device("cpu"))
    real = rope_forward_real(rope, img_shapes, mask, torch.device("cpu"))

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, mask.shape[0], 4, 128, generator=generator, dtype=torch.float32)

    expected = _reference_apply(x, freqs)
    actual = apply_rotary_emb_real(None, x, real)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_rotary_application_keeps_the_activation_dtype():
    rope = _ReferenceRope()
    img_shapes, mask = _text_and_image_sequence(text_tokens=2, height=4, width=4)
    real = rope_forward_real(rope, img_shapes, mask, torch.device("cpu"))
    x = torch.randn(1, mask.shape[0], 2, 128, dtype=torch.bfloat16)
    assert apply_rotary_emb_real(None, x, real).dtype is torch.bfloat16


def test_complex_input_still_uses_the_upstream_path():
    """A complex ``freqs`` must fall through to whatever implementation was captured at patch time."""

    import vllm_gaudi.omni.qwen_image_21 as patch

    captured = patch._ORIGINAL_APPLY_ROTARY_EMB
    calls = []

    def fake(self, x, freqs):
        calls.append(torch.is_complex(freqs))
        return x

    patch._ORIGINAL_APPLY_ROTARY_EMB = fake
    try:
        freqs = torch.zeros(4, 64, dtype=torch.complex64)
        apply_rotary_emb_real(None, torch.zeros(1, 4, 2, 128), freqs)
    finally:
        patch._ORIGINAL_APPLY_ROTARY_EMB = captured
    assert calls == [True]


def test_patch_is_idempotent_and_skips_missing_models():
    """Installing twice is a no-op, and a vLLM-Omni without 2.1 reports that it did nothing."""

    from vllm_gaudi.omni.qwen_image_21 import install_qwen_image_21_patches

    first = install_qwen_image_21_patches()
    second = install_qwen_image_21_patches()
    assert first == second
    if first:
        pytest.importorskip("vllm_omni")

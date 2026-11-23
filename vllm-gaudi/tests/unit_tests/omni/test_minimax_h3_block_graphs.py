# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
from pathlib import Path
from types import ModuleType
import sys

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[3]


def _load_tool(name: str):
    path = _REPO_ROOT / "tools" / "minimax_h3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


block_graphs = _load_tool("block_graphs")


@pytest.fixture(autouse=True)
def _clean_buffers(monkeypatch):
    block_graphs.reset_buffers()
    for name in (block_graphs.BLOCK_GRAPHS_ENV, block_graphs.MAX_GRAPH_SEQ_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    block_graphs.reset_buffers()


def _fake_runner(buckets: dict | None = None) -> ModuleType:
    runner = ModuleType("fake_run_t2va")
    runner.wrap_transformer_graphs = lambda pipe, transformer: ("original", transformer)
    table = buckets or {
        "resolutions": {
            "r384": {"hw": [384, 224], "rows_per_frame": 84},
            "r768": {"hw": [1344, 768], "rows_per_frame": 1008},
        },
        "buckets": [{"name": "preview", "width": 384, "height": 224, "seconds": 5}],
    }
    runner.load_buckets = lambda path=None: table
    runner.main = lambda argv=None: 0
    return runner


class _AddOne(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + 1


def _block_transformer(count: int) -> torch.nn.Module:
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([_AddOne() for _ in range(count)])
    return transformer


def test_ladder_rungs_share_one_aspect_ratio_exactly():
    tiers = block_graphs.LADDER_TIERS
    assert [tier.name for tier in tiers] == ["draft", "medium", "high"]
    ratios = {tier.width / tier.height for tier in tiers}
    assert ratios == {block_graphs.LADDER_ASPECT}
    for tier in tiers:
        assert tier.width % 32 == 0 and tier.height % 32 == 0
        assert tier.width * tier.height <= 768 * 1344


def test_ladder_rungs_are_one_two_three_in_linear_scale():
    widths = [tier.width for tier in block_graphs.LADDER_TIERS]
    heights = [tier.height for tier in block_graphs.LADDER_TIERS]
    assert widths == [448, 896, 1344]
    assert heights == [256, 512, 768]


def test_ladder_rows_per_frame_matches_the_released_table_formula():
    for tier in block_graphs.LADDER_TIERS:
        assert tier.rows_per_frame == tier.width * tier.height // 1024


def test_ladder_resolution_rows_keep_the_released_high_rung_identical():
    rows = block_graphs._ladder_resolutions()
    assert rows["r256"] == {"hw": [448, 256], "rows_per_frame": 112}
    assert rows["r512"] == {"hw": [896, 512], "rows_per_frame": 448}
    # The high rung is the released 768p canvas: same row, no drift.
    assert rows["r768"] == {"hw": [1344, 768], "rows_per_frame": 1008}


def test_install_injects_the_ladder_without_duplicating_named_buckets():
    runner = _fake_runner()
    state = block_graphs.install(runner)
    assert state["resolution_ladder"] == "installed"

    table = runner.load_buckets()
    assert table["resolutions"]["r256"]["hw"] == [448, 256]
    assert table["resolutions"]["r512"]["hw"] == [896, 512]
    names = [bucket["name"] for bucket in table["buckets"]]
    assert names == ["preview", "draft", "medium", "high"]

    # A second load must not append the rungs again: the server reloads the table per request.
    table = runner.load_buckets()
    assert [bucket["name"] for bucket in table["buckets"]] == ["preview", "draft", "medium", "high"]


def test_install_lifts_the_whole_dit_sequence_budget_for_block_graphs(monkeypatch):
    runner = _fake_runner()
    state = block_graphs.install(runner)
    assert "lifted" in state["graph_seq_guard"]
    assert int(os.environ[block_graphs.MAX_GRAPH_SEQ_ENV]) == block_graphs.BLOCK_GRAPH_SEQ_BUDGET


def test_install_leaves_the_sequence_budget_alone_when_block_graphs_are_off(monkeypatch):
    monkeypatch.setenv(block_graphs.BLOCK_GRAPHS_ENV, "0")
    runner = _fake_runner()
    state = block_graphs.install(runner)
    assert state["block_graphs"] == "disabled"
    assert state["graph_seq_guard"].startswith("left alone")
    assert block_graphs.MAX_GRAPH_SEQ_ENV not in os.environ


def test_install_respects_an_explicit_higher_sequence_budget(monkeypatch):
    monkeypatch.setenv(block_graphs.MAX_GRAPH_SEQ_ENV, "999999")
    runner = _fake_runner()
    state = block_graphs.install(runner)
    assert state["graph_seq_guard"] == "left at 999999"


def test_block_graphs_are_on_by_default(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)

    apply = lambda module, **kwargs: module  # noqa: E731 - stand-in for wrap_in_hpu_graph
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: apply)
    transformer = _block_transformer(4)
    pipe = object()

    assert runner.wrap_transformer_graphs(pipe, transformer) is transformer
    assert all(isinstance(block, block_graphs._PingPong) for block in transformer.transformer_blocks)
    # The chain alternates slots so block i never aliases its own pinned input.
    assert [block.slot for block in transformer.transformer_blocks] == [0, 1, 0, 1]


def test_disabling_block_graphs_restores_the_ports_own_wrapping(monkeypatch):
    monkeypatch.setenv(block_graphs.BLOCK_GRAPHS_ENV, "0")
    runner = _fake_runner()
    block_graphs.install(runner)
    transformer = _block_transformer(2)

    assert runner.wrap_transformer_graphs("pipe", transformer) == ("original", transformer)


def test_ping_pong_copies_into_the_shared_buffer_instead_of_returning_a_new_tensor():
    buffers = [torch.zeros(2, 3), torch.zeros(2, 3)]
    block_graphs._BUFFERS[0] = buffers[0]
    block_graphs._BUFFERS[1] = buffers[1]
    hidden = torch.ones(2, 3)

    out = block_graphs._PingPong(_AddOne(), 0)(hidden)
    assert out is buffers[0]
    assert torch.equal(buffers[0], torch.full((2, 3), 2.0))
    assert torch.equal(buffers[1], torch.zeros(2, 3))

    out = block_graphs._PingPong(_AddOne(), 1)(out)
    assert out is buffers[1]
    assert torch.equal(buffers[1], torch.full((2, 3), 3.0))


def test_ping_pong_refuses_to_run_without_allocated_buffers():
    with pytest.raises(RuntimeError, match="not allocated"):
        block_graphs._PingPong(_AddOne(), 0)(torch.ones(2, 3))


def test_ping_pong_refuses_a_buffer_that_no_longer_matches_the_block_output():
    block_graphs._BUFFERS[0] = torch.zeros(4, 5)
    block_graphs._BUFFERS[1] = torch.zeros(2, 3)
    with pytest.raises(RuntimeError, match="does not match the block output"):
        block_graphs._PingPong(_AddOne(), 0)(torch.ones(2, 3))


def test_prepare_buffers_allocates_two_distinct_tensors_of_the_input_shape():
    hidden = torch.ones(3, 7, dtype=torch.bfloat16)
    block_graphs._prepare_buffers(None, (hidden, ))

    first, second = block_graphs._BUFFERS
    assert first is not None and second is not None
    assert first.data_ptr() != second.data_ptr()
    for buf in (first, second):
        assert buf.shape == hidden.shape
        assert buf.dtype == hidden.dtype


def test_prepare_buffers_ignores_a_call_without_a_tensor_input():
    block_graphs._prepare_buffers(None, ())
    assert block_graphs._BUFFERS == [None, None]


def test_prepare_buffers_reuses_the_buffers_across_steps_of_one_shape():
    hidden = torch.ones(3, 7)
    block_graphs._prepare_buffers(None, (hidden, ))
    first, second = block_graphs._BUFFERS

    block_graphs._prepare_buffers(None, (torch.ones(3, 7), ))
    assert block_graphs._BUFFERS[0] is first
    assert block_graphs._BUFFERS[1] is second


def test_the_pre_hook_keeps_the_allocation_outside_the_captured_forward(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)
    apply = lambda module, **kwargs: module  # noqa: E731
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: apply)

    seen: list[tuple[int, ...] | None] = []

    class _Recording(block_graphs._PingPong):
        def forward(self, hidden_states, *args, **kwargs):
            seen.append(None if block_graphs._BUFFERS[0] is None else tuple(
                block_graphs._BUFFERS[0].shape))
            return super().forward(hidden_states, *args, **kwargs)

    transformer = _block_transformer(2)
    # Wrap by hand the way the patched function does, so the pre-hook can be observed.
    blocks = [_Recording(block, index % 2) for index, block in enumerate(transformer.transformer_blocks)]
    transformer.transformer_blocks = torch.nn.ModuleList(blocks)
    blocks[0].register_forward_pre_hook(block_graphs._prepare_buffers)

    out = transformer.transformer_blocks[0](torch.ones(5, 5))
    assert seen == [(5, 5)], "the buffers must already exist when the first block runs"
    out = transformer.transformer_blocks[1](out)
    assert seen == [(5, 5), (5, 5)]
    assert out.shape == (5, 5)

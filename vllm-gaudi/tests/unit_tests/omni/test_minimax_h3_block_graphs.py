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
    """Minimal stand-in for a transformer block: same in/out shape, extra positional args."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + 1


class _Projecting(torch.nn.Module):
    """Block with a parameter, so the parameter tree can be inspected after wrapping."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, *args, **kwargs):
        return self.proj(hidden_states)


def _block_transformer(count: int, block_type=type(_AddOne())) -> torch.nn.Module:
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([block_type() for _ in range(count)])
    return transformer


def _in_place_apply(calls: list) -> object:
    """Stand-in for `wrap_in_hpu_graph`: replaces `forward` and returns the same module."""

    def apply(module, **kwargs):
        calls.append((module, kwargs))
        original = module.forward
        module.forward = lambda *args, **kw: original(*args, **kw)
        return module

    return apply


# --------------------------------------------------------------------------- ladder definition


def test_ladder_rungs_share_one_aspect_ratio_exactly():
    tiers = block_graphs.LADDER_TIERS
    assert [tier.name for tier in tiers] == ["draft", "medium", "high"]
    assert {tier.width / tier.height for tier in tiers} == {block_graphs.LADDER_ASPECT}
    for tier in tiers:
        assert tier.width % 32 == 0 and tier.height % 32 == 0
        assert tier.width * tier.height <= 768 * 1344


def test_ladder_rungs_are_one_two_three_in_linear_scale():
    assert [tier.width for tier in block_graphs.LADDER_TIERS] == [448, 896, 1344]
    assert [tier.height for tier in block_graphs.LADDER_TIERS] == [256, 512, 768]


def test_ladder_rows_per_frame_matches_the_released_table_formula():
    for tier in block_graphs.LADDER_TIERS:
        assert tier.rows_per_frame == tier.width * tier.height // 1024


def test_ladder_resolution_rows_keep_the_released_high_rung_identical():
    rows = block_graphs._ladder_resolutions()
    assert rows["r256"] == {"hw": [448, 256], "rows_per_frame": 112}
    assert rows["r512"] == {"hw": [896, 512], "rows_per_frame": 448}
    assert rows["r768"] == {"hw": [1344, 768], "rows_per_frame": 1008}


# --------------------------------------------------------------------------------- install()


def test_install_injects_the_ladder_without_duplicating_named_buckets():
    runner = _fake_runner()
    state = block_graphs.install(runner)
    assert state["resolution_ladder"] == "installed"

    table = runner.load_buckets()
    assert table["resolutions"]["r256"]["hw"] == [448, 256]
    assert table["resolutions"]["r512"]["hw"] == [896, 512]
    assert [bucket["name"] for bucket in table["buckets"]] == ["preview", "draft", "medium", "high"]

    # A second load must not append the rungs again: the server reloads the table per request.
    table = runner.load_buckets()
    assert [bucket["name"] for bucket in table["buckets"]] == ["preview", "draft", "medium", "high"]


def test_install_defers_to_the_released_table_for_rows_it_already_has():
    """r768 is both the released 768p canvas and the high rung: the released row must win."""

    released = {"hw": [1344, 768], "rows_per_frame": 1008, "source": "buckets.json"}
    runner = _fake_runner({"resolutions": {"r768": dict(released)}, "buckets": []})
    block_graphs.install(runner)

    assert runner.load_buckets()["resolutions"]["r768"] == released


def test_install_lifts_the_sequence_budget_and_says_it_is_server_only(monkeypatch):
    runner = _fake_runner()
    state = block_graphs.install(runner)
    assert "lifted" in state["graph_seq_guard"]
    assert "server mode" in state["graph_seq_guard"]
    assert int(os.environ[block_graphs.MAX_GRAPH_SEQ_ENV]) == block_graphs.BLOCK_GRAPH_SEQ_BUDGET


def test_install_leaves_the_sequence_budget_alone_when_block_graphs_are_off():
    os.environ[block_graphs.BLOCK_GRAPHS_ENV] = "0"
    try:
        state = block_graphs.install(_fake_runner())
        assert state["block_graphs"] == "disabled"
        assert state["graph_seq_guard"].startswith("left alone")
        assert block_graphs.MAX_GRAPH_SEQ_ENV not in os.environ
    finally:
        del os.environ[block_graphs.BLOCK_GRAPHS_ENV]


def test_install_respects_an_explicit_higher_sequence_budget(monkeypatch):
    monkeypatch.setenv(block_graphs.MAX_GRAPH_SEQ_ENV, "999999")
    state = block_graphs.install(_fake_runner())
    assert state["graph_seq_guard"].startswith("left at 999999")


def test_install_refuses_a_non_numeric_sequence_budget(monkeypatch):
    monkeypatch.setenv(block_graphs.MAX_GRAPH_SEQ_ENV, "lots")
    with pytest.raises(SystemExit, match="not an integer"):
        block_graphs.install(_fake_runner())


# ------------------------------------------------------------------------------ block wrapping


def test_block_graphs_are_on_by_default_and_wrap_in_place(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)
    calls: list = []
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: _in_place_apply(calls))

    transformer = _block_transformer(4)
    original_blocks = list(transformer.transformer_blocks)

    assert runner.wrap_transformer_graphs(object(), transformer) is transformer
    # Identity preserved: no wrapper module is inserted, so the parameter tree is untouched.
    assert list(transformer.transformer_blocks) == original_blocks
    assert all(getattr(block, "_h3_ping_pong", False) for block in transformer.transformer_blocks)
    assert len(calls) == 4


def test_wrapping_does_not_add_a_level_to_parameter_names(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: _in_place_apply([]))

    transformer = _block_transformer(2, _Projecting)
    before = sorted(name for name, _ in transformer.named_parameters())
    runner.wrap_transformer_graphs(object(), transformer)
    after = sorted(name for name, _ in transformer.named_parameters())

    assert after == before
    assert "transformer_blocks.0.proj.weight" in after


def test_disabling_block_graphs_restores_the_ports_own_wrapping(monkeypatch):
    monkeypatch.setenv(block_graphs.BLOCK_GRAPHS_ENV, "0")
    runner = _fake_runner()
    block_graphs.install(runner)
    transformer = _block_transformer(2)

    assert runner.wrap_transformer_graphs("pipe", transformer) == ("original", transformer)


def test_wrapping_twice_is_refused(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: _in_place_apply([]))
    transformer = _block_transformer(2)

    runner.wrap_transformer_graphs(object(), transformer)
    with pytest.raises(RuntimeError, match="already wrapped"):
        runner.wrap_transformer_graphs(object(), transformer)


def test_the_allocation_happens_before_the_first_wrapped_forward(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: _in_place_apply([]))

    seen: list = []

    def record_buffers(*args, **kwargs):
        # Runs *after* the pre-hook and before the block body, which is exactly the window the
        # graph would replay in: the buffers must already exist by here.
        seen.append(None if block_graphs._BUFFERS[0] is None else tuple(block_graphs._BUFFERS[0].shape))
        return original_forward(*args, **kwargs)

    transformer = _block_transformer(2)
    runner.wrap_transformer_graphs(object(), transformer)
    original_forward = transformer.transformer_blocks[0].forward
    transformer.transformer_blocks[0].forward = record_buffers

    out = transformer.transformer_blocks[0](torch.ones(5, 5))
    assert seen == [(5, 5)], "the buffers must already exist when the first block runs"

    # The chain still flows through both buffers.
    out = transformer.transformer_blocks[1](out)
    assert out.shape == (5, 5)
    assert out is block_graphs._BUFFERS[1]


# ---------------------------------------------------------------------------------- buffers


def test_ping_pong_copies_into_the_shared_buffer_instead_of_returning_a_new_tensor():
    buffers = [torch.zeros(2, 3), torch.zeros(2, 3)]
    block_graphs._BUFFERS[0] = buffers[0]
    block_graphs._BUFFERS[1] = buffers[1]
    hidden = torch.ones(2, 3)

    out = block_graphs._ping_pong_forward(_AddOne().forward, 0)(hidden)
    assert out is buffers[0]
    assert torch.equal(buffers[0], torch.full((2, 3), 2.0))
    assert torch.equal(buffers[1], torch.zeros(2, 3))

    out = block_graphs._ping_pong_forward(_AddOne().forward, 1)(out)
    assert out is buffers[1]
    assert torch.equal(buffers[1], torch.full((2, 3), 3.0))


def test_ping_pong_forward_passes_extra_positional_arguments_through():
    calls: list = []

    def block(hidden_states, temb, indices, rotary_emb):
        calls.append((temb, indices, rotary_emb))
        return hidden_states * 2

    block_graphs._BUFFERS[0] = torch.zeros(1, 2)
    out = block_graphs._ping_pong_forward(block, 0)(torch.ones(1, 2), "t", "i", "r")

    assert calls == [("t", "i", "r")]
    assert torch.equal(out, torch.full((1, 2), 2.0))


def test_ping_pong_refuses_to_run_without_allocated_buffers():
    with pytest.raises(RuntimeError, match="not allocated"):
        block_graphs._ping_pong_forward(_AddOne().forward, 0)(torch.ones(2, 3))


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
    block_graphs._prepare_buffers(None, (torch.ones(3, 7), ))
    first, second = block_graphs._BUFFERS

    block_graphs._prepare_buffers(None, (torch.ones(3, 7), ))
    assert block_graphs._BUFFERS[0] is first
    assert block_graphs._BUFFERS[1] is second


def test_prepare_buffers_refuses_a_second_shape_instead_of_reallocating():
    """A captured graph holds the buffers it saw; reallocating cannot redirect it."""

    block_graphs._prepare_buffers(None, (torch.ones(3, 7), ))
    first = block_graphs._BUFFERS[0]

    with pytest.raises(RuntimeError, match="one shape per process"):
        block_graphs._prepare_buffers(None, (torch.ones(4, 7), ))

    assert block_graphs._BUFFERS[0] is first, "a refused call must not swap the buffers out"


def test_prepare_buffers_refuses_a_second_dtype():
    block_graphs._prepare_buffers(None, (torch.ones(3, 7, dtype=torch.bfloat16), ))
    with pytest.raises(RuntimeError, match="one shape per process"):
        block_graphs._prepare_buffers(None, (torch.ones(3, 7, dtype=torch.float32), ))


def test_reset_buffers_forgets_the_shape_too():
    block_graphs._prepare_buffers(None, (torch.ones(3, 7), ))
    block_graphs.reset_buffers()

    assert block_graphs._BUFFERS == [None, None]
    assert block_graphs._ALLOCATED is None
    # A different shape is fine again once the buffers are gone.
    block_graphs._prepare_buffers(None, (torch.ones(9, 2), ))
    assert block_graphs._BUFFERS[0].shape == (9, 2)


# ------------------------------------------------------- same shape, different data (the real gate)
#
# Allocation-once and shape-refusal tests both pass on a wrapper that quietly returns the wrong
# buffer, because they only ever look at one forward. The case that catches a slot-alternation
# off-by-one, a skipped pre-hook or a block invoked twice is two consecutive forwards at the SAME
# shape with DIFFERENT data: the second output has to follow the second input.


def _chain_forward(transformer, hidden):
    for block in transformer.transformer_blocks:
        hidden = block(hidden)
    return hidden


def test_two_steps_at_one_shape_each_follow_their_own_input(monkeypatch):
    runner = _fake_runner()
    block_graphs.install(runner)
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: _in_place_apply([]))

    transformer = _block_transformer(3)
    runner.wrap_transformer_graphs(object(), transformer)

    first = torch.zeros(2, 3)
    second = torch.full((2, 3), 100.0)

    out_first = _chain_forward(transformer, first)
    assert torch.equal(out_first, torch.full((2, 3), 3.0))

    out_second = _chain_forward(transformer, second)
    assert torch.equal(out_second, torch.full((2, 3), 103.0)), (
        "the second step must track its own input; a stale or aliased buffer would return the "
        "first step's values"
    )


def test_the_same_shape_gate_fails_when_a_block_hands_back_a_stale_buffer(monkeypatch):
    """Negative control: stop the copies and the gate above must go red.

    This is what a per-graph tensor cache or a skipped pre-hook produces - the block runs, its
    output is discarded, and the buffer still holds the previous step's values. A test that only
    ever runs one forward cannot see it, which is why the gate runs two at one shape.
    """

    runner = _fake_runner()
    block_graphs.install(runner)
    monkeypatch.setattr(block_graphs, "_resolve_graph_wrapper", lambda: _in_place_apply([]))

    transformer = _block_transformer(3)
    # The raw block bodies, captured before wrapping replaces `forward`.
    bodies = [block.forward for block in transformer.transformer_blocks]
    runner.wrap_transformer_graphs(object(), transformer)

    # Step 1 runs correctly: the buffers end up holding 3.0.
    out_first = _chain_forward(transformer, torch.zeros(2, 3))
    assert torch.equal(out_first, torch.full((2, 3), 3.0))

    # Now break the copy: compute, discard, hand back whatever the buffer already held.
    for index, block in enumerate(transformer.transformer_blocks):
        slot = index % 2

        def stale_forward(hidden_states, *args, _body=bodies[index], _slot=slot, **kwargs):
            _body(hidden_states, *args, **kwargs)
            return block_graphs._BUFFERS[_slot]

        block.forward = stale_forward

    out_second = _chain_forward(transformer, torch.full((2, 3), 100.0))

    # The gate above expects 103.0 here; a stale buffer hands back the first step's 3.0.
    assert torch.equal(out_second, torch.full((2, 3), 3.0)), (
        "the negative control did not go stale, so the same-shape gate would have no teeth"
    )

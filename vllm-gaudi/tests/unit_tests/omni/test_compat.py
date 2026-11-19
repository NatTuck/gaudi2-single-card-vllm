# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.omni.compat import install_vllm_omni_compat
from vllm_gaudi.omni import register_omni_platform
from vllm_gaudi.platform import HpuPlatform


def test_compat_install_is_idempotent():
    install_vllm_omni_compat()
    install_vllm_omni_compat()

    # Every bridge is installed by name only when the name is absent, so the second call must
    # leave the modules untouched.
    import vllm.distributed as distributed

    assert callable(distributed.get_tp_group)


def test_distributed_reexports_match_their_sources():
    """vLLM-Omni imports these from the package; vLLM only defines them in the submodules."""

    install_vllm_omni_compat()

    import vllm.distributed as distributed
    from vllm.distributed import communication_op, parallel_state

    for name in (
            "get_tp_group",
            "get_pp_group",
            "get_ep_group",
            "get_dp_group",
            "get_tensor_model_parallel_rank",
            "get_tensor_model_parallel_world_size",
    ):
        assert getattr(distributed, name) is getattr(parallel_state, name), name

    for name in ("tensor_model_parallel_all_gather", "tensor_model_parallel_all_reduce"):
        assert getattr(distributed, name) is getattr(communication_op, name), name


def test_compat_does_not_shadow_modules_vllm_provides():
    """The stand-ins must never replace a real vLLM module, whatever revision is installed."""

    install_vllm_omni_compat()

    import importlib
    import importlib.util

    if importlib.util.find_spec("vllm.v1.kv_hints") is not None:
        module = importlib.import_module("vllm.v1.kv_hints")
        assert not getattr(module, "__file__", "").endswith("<stub>")
        assert module.__file__ is not None


def test_torchaudio_stub_raises_on_use():
    install_vllm_omni_compat()

    import importlib.util

    if importlib.util.find_spec("torchaudio") is not None:
        pytest.skip("a real torchaudio is installed")

    import torchaudio

    with pytest.raises(RuntimeError, match="torchaudio is not available"):
        torchaudio.functional.resample(None, 1, 2)


def test_hpu_omni_platform_exposes_cache_eviction(monkeypatch):
    # Platform class paths are normally resolved after vllm_omni finishes its
    # package-level compatibility patching. Reproduce that entrypoint order in
    # this isolated test module as well.
    import vllm_omni  # noqa: F401

    from vllm_gaudi.omni.platform import HPUOmniPlatform

    calls = []
    monkeypatch.setattr(HpuPlatform, "empty_cache", classmethod(lambda cls: calls.append(cls)))

    HPUOmniPlatform.empty_cache()

    assert calls == [HpuPlatform]


def test_omni_platform_entrypoint_is_inactive_without_hpu(monkeypatch):
    monkeypatch.setattr(torch.hpu, "is_available", lambda: False)

    assert register_omni_platform() is None


def test_omni_platform_entrypoint_defers_compat_until_class_resolution(monkeypatch):
    monkeypatch.setattr(torch.hpu, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(
        "vllm_gaudi.omni.compat.install_vllm_omni_compat",
        lambda: calls.append("install"),
    )

    assert register_omni_platform() == "vllm_gaudi.omni.platform.HPUOmniPlatform"
    assert calls == []


def test_hpu_omni_profiler_registration():
    from vllm_gaudi.omni.platform import HPUOmniPlatform
    from vllm_gaudi.omni.profiler import HPUOmniTorchProfilerWrapper

    assert HPUOmniPlatform.get_profiler_cls() == "vllm_gaudi.omni.profiler.HPUOmniTorchProfilerWrapper"
    assert HPUOmniTorchProfilerWrapper._get_default_activities(object()) == ["CPU", "HPU"]


def test_hpu_omni_profiler_creates_cpu_hpu_trace(monkeypatch):
    from vllm_gaudi.omni.profiler import HPUOmniTorchProfilerWrapper

    captured = {}
    sentinel = object()

    def fake_profile(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(torch.profiler, "profile", fake_profile)
    config = SimpleNamespace(
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
        torch_profiler_with_flops=False,
    )
    wrapper = object.__new__(HPUOmniTorchProfilerWrapper)

    assert wrapper._create_profiler(config, ["CPU", "HPU"]) is sentinel
    assert captured["activities"] == [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.HPU,
    ]
    assert captured["record_shapes"] is True

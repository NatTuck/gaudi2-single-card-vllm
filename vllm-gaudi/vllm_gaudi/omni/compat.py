# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import bridges for the vLLM / vLLM-Omni 0.32 pair on Gaudi.

Three things vLLM-Omni 0.32 imports are not importable from a vanilla vLLM 0.32:

* names vLLM only defines in ``vllm.distributed.parallel_state`` /
  ``vllm.distributed.communication_op``, which the diffusion models import from the package;
* the ``vllm.v1.kv_hints`` module (AR-stage KV-cache hints, added after 0.32);
* ``torchaudio``, which has no build matching the Gaudi torch.

Each bridge is a no-op when the name or module already exists, so a later vLLM that ships them
keeps working unchanged. Only import-time compatibility is provided.
"""

from __future__ import annotations

import sys
import types


# (package module, name, module that already defines it in vLLM 0.32)
_REEXPORTS: tuple[tuple[str, str, str], ...] = (
    ("vllm.distributed", "get_tp_group", "vllm.distributed.parallel_state"),
    ("vllm.distributed", "get_pp_group", "vllm.distributed.parallel_state"),
    ("vllm.distributed", "get_ep_group", "vllm.distributed.parallel_state"),
    ("vllm.distributed", "get_dp_group", "vllm.distributed.parallel_state"),
    ("vllm.distributed", "get_tensor_model_parallel_rank", "vllm.distributed.parallel_state"),
    ("vllm.distributed", "get_tensor_model_parallel_world_size", "vllm.distributed.parallel_state"),
    ("vllm.distributed", "tensor_model_parallel_all_gather", "vllm.distributed.communication_op"),
    ("vllm.distributed", "tensor_model_parallel_all_reduce", "vllm.distributed.communication_op"),
    ("vllm.distributed", "split_tensor_along_last_dim", "vllm.distributed.utils"),
    ("vllm.distributed", "divide", "vllm.distributed.utils"),
)


def _install_reexports() -> None:
    """Re-publish names that vLLM-Omni imports from a package that does not export them."""

    import importlib

    for package_name, name, source_name in _REEXPORTS:
        package = importlib.import_module(package_name)
        if hasattr(package, name):
            continue
        source = importlib.import_module(source_name)
        value = getattr(source, name, None)
        if value is not None:
            setattr(package, name, value)


def _install_module_stub(module_name: str, **attributes: object) -> types.ModuleType | None:
    """Register a stand-in for a vLLM module that this vLLM does not have.

    Returns ``None`` (and touches nothing) when the module is importable, so the stand-in can
    never shadow a real vLLM module.
    """

    import importlib
    import importlib.util

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    try:
        if importlib.util.find_spec(module_name) is not None:
            return importlib.import_module(module_name)
    except (ImportError, ValueError):
        pass
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules[module_name] = module
    parent_name, _, child = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, module)
    return module


def _install_module_stubs() -> None:
    """Stand-ins for vLLM modules this revision does not have.

    ``vllm.v1.kv_hints`` carries the AR-stage KV-cache hint envelope. The diffusion path rejects
    it explicitly ("kv_hints require an AR/LLM stage with a KV cache"), so the type only has to
    exist for the annotations of the async engine to import.
    """

    import vllm.v1  # the parent package must exist before the attribute is set

    class KvHintsEnvelope:
        """AR-stage KV-cache hints (not present in vLLM 0.32; unused by diffusion stages)."""

        def __init__(self, **fields: object) -> None:
            self.__dict__.update(fields)

    _install_module_stub("vllm.v1.kv_hints", KvHintsEnvelope=KvHintsEnvelope)


class _StubModule(types.ModuleType):
    """A module whose every attribute exists and raises when called."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def _unavailable(*_args: object, **_kwargs: object):
            raise RuntimeError(f"torchaudio is not available in this environment (used {self.__name__}.{name})")

        setattr(self, name, _unavailable)
        return _unavailable


def _install_torchaudio_stub() -> None:
    """Stand in for ``torchaudio`` when the environment has no build matching this torch.

    vLLM-Omni imports ``torchaudio`` at module import time from its audio helpers (and from
    several audio pipelines that its registry loads eagerly). The Gaudi PyTorch build here has no
    matching torchaudio wheel, and the CUDA wheel fails on the extension ABI, so the audio paths
    get a stand-in: every attribute exists, calling it raises. Image and video serving never
    reaches them.
    """

    import importlib.machinery
    import importlib.util

    if "torchaudio" in sys.modules:
        return
    try:
        if importlib.util.find_spec("torchaudio") is not None:
            return
    except (ImportError, ValueError):
        pass

    package = _StubModule("torchaudio")
    package.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    package.__path__ = []  # a package, so ``torchaudio.functional`` resolves through it
    package.__version__ = "0.0.0+stub"
    sys.modules["torchaudio"] = package

    for child in ("functional", "transforms", "librosa_compat"):
        module = _StubModule(f"torchaudio.{child}")
        module.__spec__ = importlib.machinery.ModuleSpec(f"torchaudio.{child}", None)
        sys.modules[f"torchaudio.{child}"] = module
        setattr(package, child, module)


def install_vllm_omni_compat() -> None:
    """Install idempotent import bridges required by vLLM-Omni 0.32."""

    _install_reexports()
    _install_module_stubs()
    _install_torchaudio_stub()


__all__ = ["install_vllm_omni_compat"]

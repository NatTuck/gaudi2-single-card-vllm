# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility bridges for the vLLM and vLLM-Omni revisions.

The pinned vLLM is older than the vLLM-Omni revision that carries Qwen-Image 2.1, and that
revision imports a handful of vLLM names at module import time. Three kinds of bridge live here:

* module paths that moved (``vllm.entrypoints.*.protocol``),
* names vLLM added to a package's ``__init__`` after the pin (``vllm.distributed`` re-exports),
* names that did not exist yet (logging hooks, one dataclass, one quant method base).

Only import-time compatibility is provided. The bridges are idempotent and are no-ops when the
installed vLLM already has the name, so the pinned vLLM-Omni keeps running exactly as before.
"""

from __future__ import annotations

import logging
import sys
import types


def _install_protocol_modules() -> None:
    """Provide protocol module paths introduced after pinned vLLM."""

    from vllm.entrypoints.openai.engine import protocol

    for module_name in (
            "vllm.entrypoints.generate.base.protocol",
            "vllm.entrypoints.serve.engine.protocol",
    ):
        sys.modules.setdefault(module_name, protocol)


def _install_logger_compat() -> None:
    """Provide the runtime logging hooks newer vLLM-Omni calls into.

    vLLM 0.28 configures its root logger when ``vllm.logger`` is imported and exposes no
    runtime switch; later revisions added ``configure_logging`` / ``configure_logging_from_args``
    for the CLI and the engine-core process. Both are pure logging configuration, so a bridge
    that only honours an explicit level is equivalent for our purposes.
    """

    from vllm import logger as vllm_logger

    def _set_level(level: object) -> None:
        if level is None:
            return
        if isinstance(level, int):
            logging.getLogger("vllm").setLevel(level)
            return
        if isinstance(level, str):
            resolved = logging.getLevelName(level.upper())
            if isinstance(resolved, int):
                logging.getLogger("vllm").setLevel(resolved)

    if not hasattr(vllm_logger, "configure_logging_from_args"):

        def configure_logging_from_args(args: object) -> None:
            _set_level(getattr(args, "log_level", None))

        vllm_logger.configure_logging_from_args = configure_logging_from_args

    if not hasattr(vllm_logger, "configure_logging"):

        def configure_logging(logging_config: object) -> None:
            level = getattr(logging_config, "log_level", None)
            if level is None and isinstance(logging_config, str):
                level = logging_config
            _set_level(level)

        vllm_logger.configure_logging = configure_logging


# (package module, name, module that already defines it in the pinned vLLM)
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
    """Re-publish names that later vLLM re-exports from a package ``__init__``."""

    import importlib

    for package_name, name, source_name in _REEXPORTS:
        package = importlib.import_module(package_name)
        if hasattr(package, name):
            continue
        source = importlib.import_module(source_name)
        value = getattr(source, name, None)
        if value is not None:
            setattr(package, name, value)


def _install_symbol_stubs() -> None:
    """Provide names that do not exist in the pinned vLLM at all.

    None of these is on the diffusion path we serve; they are imported by modules that
    vLLM-Omni loads eagerly (its model zoo, its benchmarks). A stand-in keeps the import
    working, and the ``isinstance`` check below stays correct because nothing in the pinned
    vLLM is an instance of the stand-in.
    """

    from vllm.config import multimodal as vllm_multimodal_config
    from vllm.model_executor.layers.quantization import modelopt as vllm_modelopt
    from vllm.multimodal.processing import processor as vllm_processor

    if not hasattr(vllm_multimodal_config, "MultiModalDummyOptions"):

        class MultiModalDummyOptions(dict):
            """Per-modality dummy-data options container (added after pinned vLLM)."""

            def __init__(self, options: object = None, **fields: object) -> None:
                super().__init__(options or {})
                self.__dict__.update(fields)

        vllm_multimodal_config.MultiModalDummyOptions = MultiModalDummyOptions

    if not hasattr(vllm_processor, "MultiModalProcessingResult"):

        class MultiModalProcessingResult:
            """Processed multimodal payload (added after pinned vLLM)."""

            def __init__(self, **fields: object) -> None:
                self.__dict__.update(fields)

        vllm_processor.MultiModalProcessingResult = MultiModalProcessingResult

    if not hasattr(vllm_modelopt, "ModelOptLinearMethod"):

        class ModelOptLinearMethod:
            """Common ModelOpt linear-method base (added after pinned vLLM)."""

        vllm_modelopt.ModelOptLinearMethod = ModelOptLinearMethod


def _install_module_stub(module_name: str, **attributes: object) -> types.ModuleType:
    """Register a stand-in for a vLLM module that a newer vLLM-Omni imports but this vLLM lacks."""

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
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
    """Stand-ins for vLLM modules that do not exist in the pinned revision.

    ``vllm.v1.kv_hints`` carries the AR-stage KV-cache hint envelope. The diffusion path
    rejects it explicitly ("kv_hints require an AR/LLM stage with a KV cache"), so the type
    only has to exist for the annotations of the async engine to import.
    """

    import vllm.v1  # the parent package must exist before the attribute is set

    class KvHintsEnvelope:
        """AR-stage KV-cache hints (added after pinned vLLM; unused by diffusion stages)."""

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
    several audio pipelines that its registry loads eagerly). The Gaudi PyTorch build here has
    no matching torchaudio wheel, and a mismatched one fails on the extension ABI, so the audio
    paths get a stand-in: every attribute exists, calling it raises. Image and video serving
    never reaches them.
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
    """Install idempotent import bridges required by pinned vLLM-Omni."""

    _install_protocol_modules()
    _install_logger_compat()
    _install_reexports()
    _install_symbol_stubs()
    _install_module_stubs()
    _install_torchaudio_stub()


__all__ = ["install_vllm_omni_compat"]

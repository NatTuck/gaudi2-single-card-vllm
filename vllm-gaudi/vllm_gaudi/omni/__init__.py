# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-Omni integration for Intel Gaudi."""

import logging

logger = logging.getLogger(__name__)


def _install_compat_early() -> None:
    """Install the vLLM import bridges before vLLM-Omni's own modules are imported.

    vLLM-Omni imports ``vllm.logger.configure_logging_from_args`` and the
    ``vllm.entrypoints.serve.engine.protocol`` module path at module import time, and it
    imports them while the platform plugin is being resolved - before ``register_omni()``
    runs. Installing the bridges at package import keeps that path working without touching
    detection: an import failure here is logged and ignored, so probing still falls through
    to the plugin's own return value.
    """

    try:
        from vllm_gaudi.omni.compat import install_vllm_omni_compat
    except ImportError:  # vLLM itself is missing: nothing to bridge
        return
    try:
        install_vllm_omni_compat()
    except Exception:  # never break platform detection over a compatibility shim
        logger.debug("vLLM-Omni compatibility bridges not installed", exc_info=True)


_install_compat_early()


def register_omni_platform() -> str | None:
    """Activate the Omni HPU platform only when a Gaudi device is usable."""

    try:
        import torch
    except ImportError:
        return None
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        return None

    # Keep detection side-effect free. Omni suppresses errors raised while it
    # probes platform plugins; compatibility setup therefore happens when it
    # resolves the selected class, where an import failure is fatal instead of
    # silently selecting the unspecified platform.
    return "vllm_gaudi.omni.platform.HPUOmniPlatform"


def register_omni() -> None:
    """Install Gaudi-specific vLLM-Omni compatibility hooks."""

    from vllm_gaudi.omni.compat import install_vllm_omni_compat
    from vllm_gaudi.omni.minimax_h3 import install_minimax_h3_patches
    from vllm_gaudi.omni.qwen_image_21 import install_qwen_image_21_patches
    from vllm_gaudi.omni.regional_compile import install_hpu_regional_compile

    install_vllm_omni_compat()
    # The H3 patches rebind internals of the pinned vLLM-Omni revision. They are specific to
    # that model, so a revision that moved those internals must not take the other diffusion
    # models (Qwen-Image) down with it.
    try:
        install_minimax_h3_patches()
    except Exception:
        logger.warning("MiniMax H3 patches not installed on this vLLM-Omni revision", exc_info=True)
    install_qwen_image_21_patches()
    install_hpu_regional_compile()


__all__ = ["register_omni", "register_omni_platform"]

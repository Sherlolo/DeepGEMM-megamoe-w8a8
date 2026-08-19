"""V3 backend names and test-layer backend selection helpers."""

from __future__ import annotations

import os

V3_BACKEND_LL = "ll"
V3_BACKEND_NORMAL = "normal"
V3_BACKEND_AUTO = "auto"

V3_QUANT_FP8 = "fp8"
V3_QUANT_INT8 = "int8"

BACKEND_ENV = "MEGAMOE_DCU_BACKEND"
NORMAL_LL_TOKEN_THRESHOLD_ENV = "MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD"
DEFAULT_NORMAL_LL_TOKEN_THRESHOLD = 512

_VALID_V3_BACKENDS = {V3_BACKEND_LL, V3_BACKEND_NORMAL}
_VALID_BACKEND_MODES = {V3_BACKEND_AUTO, V3_BACKEND_LL, V3_BACKEND_NORMAL}
_VALID_V3_QUANTS = {V3_QUANT_FP8, V3_QUANT_INT8}

SUPPORTED_STAGED_EP_RANKS = (8, 16, 32)
STAGED_FP8_MAX_LOCAL_EXPERTS = 64
STAGED_FP8_HIDDEN_ALIGNMENT = 256
STAGED_FP8_INTERMEDIATE_ALIGNMENT = 128
STAGED_FP8_MAX_INTERMEDIATE = 4096

DEEPSEEK_V4_FLASH_SHAPE = (256, 6, 4096, 2048)
DEEPSEEK_V4_PRO_SHAPE = (384, 6, 7168, 3072)
YGZP_INT8_SHAPE = (288, 8, 4096, 2048)
STAGED_PACK5_MODEL_SHAPES = {
    "DeepSeek-V4-Flash": DEEPSEEK_V4_FLASH_SHAPE,
    "DeepSeek-V4-Pro": DEEPSEEK_V4_PRO_SHAPE,
}
STAGED_INT8_NORMAL_MODEL_SHAPES = {
    "DeepSeek-V4-Flash-INT8": DEEPSEEK_V4_FLASH_SHAPE,
    "YGZP-INT8": YGZP_INT8_SHAPE,
}
STAGED_PACK5_LOCAL_EXPERTS = tuple(range(1, STAGED_FP8_MAX_LOCAL_EXPERTS + 1))
STAGED_PACK5_SHAPE_CONTRACT = (
    "DCU MegaMoE staged normal FP8 pack5 path supports EP8/EP16/EP32, "
    "positive experts divisible by EP ranks with at most 64 local experts, "
    "topk in [1, experts], hidden divisible by 256, "
    "intermediate divisible by 128 and in [128, 4096]; the LL path remains "
    "specialized for DeepSeek-V4-Flash/Pro"
)

# This registry retains the exact LL FP8 and normal INT8 capabilities. Normal
# FP8 uses the aligned runtime-dimension gate in staged_v3_capability_supported.
STAGED_V3_MODEL_CAPABILITIES = {
    "DeepSeek-V4-Flash": {
        "shape": DEEPSEEK_V4_FLASH_SHAPE,
        "quant": V3_QUANT_FP8,
        "backends": (V3_BACKEND_LL, V3_BACKEND_NORMAL),
        "ep_ranks": SUPPORTED_STAGED_EP_RANKS,
    },
    "DeepSeek-V4-Pro": {
        "shape": DEEPSEEK_V4_PRO_SHAPE,
        "quant": V3_QUANT_FP8,
        "backends": (V3_BACKEND_LL, V3_BACKEND_NORMAL),
        "ep_ranks": SUPPORTED_STAGED_EP_RANKS,
    },
    **{
        name: {
            "shape": shape,
            "quant": V3_QUANT_INT8,
            "backends": (V3_BACKEND_NORMAL,),
            "ep_ranks": SUPPORTED_STAGED_EP_RANKS,
        }
        for name, shape in STAGED_INT8_NORMAL_MODEL_SHAPES.items()
    },
}
STAGED_V3_CAPABILITY_CONTRACT = (
    "DCU MegaMoE staged V3 supports dynamic FP8 experts/topk and aligned "
    "hidden/intermediate on the normal backend with at most 64 local experts, "
    "FP8 DeepSeek-V4-Flash/Pro on LL, and INT8 DeepSeek-V4-Flash/YGZP "
    "normal on EP8/EP16/EP32"
)


def normalize_v3_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in _VALID_V3_BACKENDS:
        raise ValueError(
            f"V3 backend must be one of {sorted(_VALID_V3_BACKENDS)}, got {value!r}"
        )
    return backend


def normalize_v3_quant(value: str) -> str:
    quant = str(value).strip().lower()
    if quant not in _VALID_V3_QUANTS:
        raise ValueError(
            f"V3 quant must be one of {sorted(_VALID_V3_QUANTS)}, got {value!r}"
        )
    return quant


def normalize_backend_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in _VALID_BACKEND_MODES:
        raise ValueError(
            f"{BACKEND_ENV} must be one of {sorted(_VALID_BACKEND_MODES)}, got {value!r}"
        )
    return mode


def normal_ll_token_threshold(value: str | int | None = None) -> int:
    raw = os.getenv(
        NORMAL_LL_TOKEN_THRESHOLD_ENV,
        str(DEFAULT_NORMAL_LL_TOKEN_THRESHOLD),
    )
    if value is not None:
        raw = value
    threshold = int(raw)
    if threshold < 0:
        raise ValueError(f"{NORMAL_LL_TOKEN_THRESHOLD_ENV} must be non-negative")
    return threshold


def staged_pack5_dims_supported(*, hidden: int, intermediate_hidden: int) -> bool:
    hidden = int(hidden)
    intermediate = int(intermediate_hidden)
    return (
        hidden >= STAGED_FP8_HIDDEN_ALIGNMENT
        and hidden % STAGED_FP8_HIDDEN_ALIGNMENT == 0
        and STAGED_FP8_INTERMEDIATE_ALIGNMENT
        <= intermediate
        <= STAGED_FP8_MAX_INTERMEDIATE
        and intermediate % STAGED_FP8_INTERMEDIATE_ALIGNMENT == 0
        and hidden * (2 * intermediate) <= 0xFFFFFFFF
    )


def staged_pack5_legacy_model_shape_supported(
    *,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    shape = (
        int(num_experts),
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
    )
    return shape in STAGED_PACK5_MODEL_SHAPES.values()


def staged_pack5_model_shape_supported(
    *,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    experts = int(num_experts)
    topk = int(num_topk)
    return (
        experts > 0
        and 0 < topk <= experts
        and staged_pack5_dims_supported(
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    )


def staged_pack5_shape_supported(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    ranks = int(num_ranks)
    experts = int(num_experts)
    return (
        ranks in SUPPORTED_STAGED_EP_RANKS
        and experts > 0
        and experts % ranks == 0
        and 0 < experts // ranks <= STAGED_FP8_MAX_LOCAL_EXPERTS
        and staged_pack5_model_shape_supported(
            num_experts=experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    )


def staged_v3_capability_supported(
    *,
    quant: str,
    backend: str,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    """Return whether a quant/backend/EP/model-shape combination is executable."""

    quant_mode = str(quant).strip().lower()
    backend_mode = str(backend).strip().lower()
    ranks = int(num_ranks)
    experts = int(num_experts)
    if (
        quant_mode not in _VALID_V3_QUANTS
        or backend_mode not in _VALID_V3_BACKENDS
        or ranks <= 0
        or experts % ranks != 0
    ):
        return False

    shape = (
        experts,
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
    )
    if quant_mode == V3_QUANT_FP8 and backend_mode == V3_BACKEND_NORMAL:
        return staged_pack5_shape_supported(
            num_ranks=ranks,
            num_experts=experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    return any(
        shape == capability["shape"]
        and quant_mode == capability["quant"]
        and backend_mode in capability["backends"]
        and ranks in capability["ep_ranks"]
        for capability in STAGED_V3_MODEL_CAPABILITIES.values()
    )


def staged_v3_capability_local_experts(
    *,
    quant: str,
    backend: str,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> int:
    """Return local experts only after the execution capability gate succeeds."""

    if not staged_v3_capability_supported(
        quant=quant,
        backend=backend,
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    ):
        raise ValueError(STAGED_V3_CAPABILITY_CONTRACT)
    return int(num_experts) // int(num_ranks)


def staged_pack5_local_experts(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> int:
    if not staged_pack5_shape_supported(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    ):
        raise ValueError(STAGED_PACK5_SHAPE_CONTRACT)
    return int(num_experts) // int(num_ranks)


def staged_pack5_local_experts_supported(local_experts: int) -> bool:
    return 0 < int(local_experts) <= STAGED_FP8_MAX_LOCAL_EXPERTS


def staged_pack5_k1_shape_supported(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    l1_rows: int,
) -> bool:
    intermediate_hidden = int(l1_rows) // 2
    return (
        int(l1_rows) % 2 == 0
        and staged_pack5_shape_supported(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    )


def staged_pack5_k3_dims_supported(*, hidden: int, intermediate_hidden: int) -> bool:
    return staged_pack5_dims_supported(
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )


def v3_backend_mode(value: str | None = None) -> str:
    raw = os.getenv(BACKEND_ENV, V3_BACKEND_AUTO) if value is None else value
    return normalize_backend_mode(raw)


def select_v3_backend(selector_tokens: int, backend_mode: str | None = None) -> str:
    """Return the test-layer V3 backend for a uniform EP selector token bucket."""

    tokens = int(selector_tokens)
    if tokens < 0:
        raise ValueError("selector_tokens must be non-negative")
    mode = v3_backend_mode(backend_mode)
    if mode != V3_BACKEND_AUTO:
        return normalize_v3_backend(mode)
    threshold = normal_ll_token_threshold()
    return V3_BACKEND_LL if tokens <= threshold else V3_BACKEND_NORMAL

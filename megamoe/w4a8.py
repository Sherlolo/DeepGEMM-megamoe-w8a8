"""Packed INT4 / INT8 HIPC MegaMoE for EP8 prefill."""

import logging
import os

import torch

from .dcu_megamoe_opt import w4a8_ext
from .dcu_megamoe_opt.K3_fused.k3_fused import rank_barrier

logger = logging.getLogger(__name__)


def _stage_w4a8_inputs(x, ids, weights, buffer):
    """Stage inputs with fused reference quantization on supported layouts."""
    tokens, hidden = x.shape
    if tokens == 0:
        return
    if tokens > buffer.x.shape[0] or hidden != buffer.x.shape[1]:
        raise ValueError("W4A8 input exceeds the staging buffer shape")
    if (ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != tokens
            or ids.shape[1] != buffer.topk_idx.shape[1]
            or ids.shape[1] != buffer.topk_weights.shape[1]
            or buffer.topk_idx.shape[0] < tokens or buffer.topk_weights.shape[0] < tokens
            or buffer.x_sf.numel() < tokens):
        raise ValueError("W4A8 staging route shape mismatch")
    if any(t.device != buffer.x.device for t in (x, ids, weights, buffer.x_sf,
                                                buffer.topk_idx, buffer.topk_weights)):
        raise ValueError("W4A8 staging tensors must share a device")
    if (x.dtype == torch.bfloat16 and x.shape[1] == 4096
            and ids.shape[1] == 6 and x.is_contiguous()
            and ids.is_contiguous() and weights.is_contiguous()):
        from ._w4a8_staging import stage_inputs
        stage_inputs(x, ids, weights, buffer)
    else:
        from lightop.quant import per_token_quant_int8
        qx, scale = per_token_quant_int8(x.contiguous())
        buffer.x[:tokens].copy_(qx)
        buffer.x_sf[:tokens].copy_(scale.reshape(-1))
        buffer.topk_idx[:tokens].copy_(ids)
        buffer.topk_weights[:tokens].copy_(weights)


def grouped_gemm(x, weight, out, indices, *, block_m=64, n_loop=4):
    """Expert-contiguous GEMM; segment starts must be aligned to ``block_m``."""
    narrow_block = os.environ.get("MEGAMOE_W4A8_NARROW_BLOCK", "0") == "1"
    if block_m not in (32, 64, 128, 256) or n_loop not in (1, 2, 4) or (block_m == 256 and n_loop == 4 and not narrow_block):
        raise ValueError("unsupported W4A8 tile")
    if x[0].shape[1] % 128 or out.shape[1] % 512:
        raise ValueError("W4A8 HIPC requires K divisible by 128 and N by 512")
    if out.shape[0] == 0:
        return out
    mode = 10000 + block_m * 10 + n_loop
    if (block_m, n_loop) in ((64, 1), (64, 2), (64, 4), (128, 2)) and os.environ.get("MEGAMOE_W4A8_VECTOR_A", "0") == "1":
        mode += 40000
    elif block_m == 128 and n_loop in (1, 2) and os.environ.get("MEGAMOE_W4A8_WIDE_WAVE", "0") == "1":
        mode += 30000
    elif narrow_block and block_m in (64, 128, 256) and n_loop == 4:
        mode += 20000
    elif mode == 10644 and os.environ.get("MEGAMOE_W4A8_LDS_SWIZZLE", "0") == "1":
        mode = 20644
    return w4a8_ext.gemm(x[0], weight[0], out, x[1], weight[1], indices, mode)


def _check_weights(weights, experts, n, k, device):
    w, scale = weights
    if (w.dtype != torch.int8 or w.shape != (experts, n, k // 2)
            or not w.is_contiguous() or w.device != device):
        raise ValueError("W4A8 weights must be contiguous device INT4 HIPC bytes [E,N,K/2]")
    if (scale.dtype != torch.float32 or scale.shape not in ((experts, n), (experts, n, 1))
            or not scale.is_contiguous() or scale.device != device):
        raise ValueError("W4A8 weight scales must be contiguous device FP32 [E,N] or [E,N,1]")


def w4a8_mega_moe(y, l1_weights, l2_weights, sym_buffer,
                  cumulative_local_expert_recv_stats=None, *, activation_clamp=None,
                  combine_group_map=None, gemm_config=None):
    """Consume staged INT8 inputs and physical expert IDs in ``sym_buffer``.

    Weights use DeepGEMM's HIPC N32 packing and true per-channel scales.
    ``y.shape[0]`` is this rank's token count; ranks may have different counts.
    All ranks must call, including ranks with no local input tokens.
    ``combine_group_map`` maps physical experts to their canonical EP rank
    for the two-level BF16 reduction, preserving baseline rounding under EPLB.
    """
    if sym_buffer.quant_mode != "w4a8":
        raise ValueError("W4A8 requires SymmBuffer(quant_mode='w4a8')")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("W4A8 HIPC currently supports eager prefill only")
    p = sym_buffer
    rank, ranks = p.group.rank(), p.group.size()
    tokens, hidden = y.shape
    if (y.dtype != torch.bfloat16 or y.device != p.buffer.device or not y.is_contiguous()
            or hidden != p.hidden or tokens > p.num_max_tokens_per_rank):
        raise ValueError("invalid W4A8 output shape, dtype, device or layout")
    local_experts = p.num_experts // ranks
    intermediate = p.intermediate_hidden
    if gemm_config is None:
        block_m = int(os.environ.get("MEGAMOE_W4A8_BLOCK_M", "64"))
        n_loop = int(os.environ.get("MEGAMOE_W4A8_N_LOOP", "4"))
        gemm_config = ((block_m, n_loop), (block_m, n_loop))
    (block_m1, n_loop1), (block_m2, n_loop2) = gemm_config
    alignment = max(64, block_m1, block_m2)
    _check_weights(l1_weights, local_experts, 2 * intermediate, hidden, y.device)
    _check_weights(l2_weights, local_experts, hidden, intermediate, y.device)
    if cumulative_local_expert_recv_stats is not None:
        raise NotImplementedError("Use SGLang's logical expert distribution recorder for offline EPLB")
    args = (ranks, p.num_experts, p.num_max_tokens_per_rank, p.num_topk, hidden, rank)
    rank_partials = (p.num_topk == 6 and combine_group_map is None and
                     os.environ.get("MEGAMOE_W4A8_RANK_PARTIALS", "1") == "1")
    inverse = None
    route_weights = None
    mapped_gather = (p.num_topk == 6 and activation_clamp is not None
                     and os.environ.get("MEGAMOE_W4A8_DEVICE_ROWS", "1") == "1")
    if rank_partials or mapped_gather:
        if not hasattr(p, "_w4a8_inverse"):
            p._w4a8_inverse = torch.empty(ranks * p.num_max_tokens_per_rank * p.num_topk,
                                          device=y.device, dtype=torch.int32)
            p._w4a8_route_weights = torch.empty_like(p._w4a8_inverse, dtype=torch.float32)
        inverse = p._w4a8_inverse
        route_weights = p._w4a8_route_weights
    reuse_combine = os.environ.get("MEGAMOE_W4A8_REUSE_COMBINE", "1") == "1"
    if not reuse_combine:
        p.combine.zero_()
    rank_barrier(p, rank_idx=rank, num_ranks=ranks, graph_max_tokens=tokens)
    offsets = w4a8_ext.count_routes(p.buffer, *args, alignment)
    device_rows = activation_clamp is not None and os.environ.get("MEGAMOE_W4A8_DEVICE_ROWS", "1") == "1"
    if device_rows:
        active_rows = int(offsets[-1].item())
        p._w4a8_active_rows = active_rows
        cached = getattr(p, "_w4a8_workspace", None)
        if (cached is not None and cached[0][1:] == (hidden, intermediate)
                and active_rows <= cached[0][0] <= max(alignment, 2 * active_rows)
                and cached[0][0] % alignment == 0):
            rows = cached[0][0]
        else:
            quantum = max(alignment, ((active_rows + 8 * alignment - 1)
                                       // (8 * alignment)) * alignment)
            rows = ((active_rows + quantum - 1) // quantum) * quantum
        key = (rows, hidden, intermediate)
        if cached is None or cached[0] != key:
            # Release stale oversized buffers before allocating their replacement.
            p._w4a8_workspace = None
            cached = None
            routes = torch.empty(rows, dtype=torch.int32, device=y.device)
            indices = torch.empty_like(routes)
            x = torch.empty((rows, hidden), dtype=torch.int8, device=y.device)
            scale = torch.empty(rows, dtype=torch.float32, device=y.device)
            gate_up = torch.empty((rows, 2 * intermediate), dtype=torch.bfloat16, device=y.device)
            act = torch.empty((rows, intermediate), dtype=torch.int8, device=y.device)
            act_scale = torch.empty((rows, 1), dtype=torch.float32, device=y.device)
            down = torch.empty((rows, hidden), dtype=torch.bfloat16, device=y.device)
            cached = (key, routes, indices, x, scale, gate_up, act, act_scale, down)
            p._w4a8_workspace = cached
            p._w4a8_workspace_allocations = getattr(p, "_w4a8_workspace_allocations", 0) + 1
            logger.info("W4A8 workspace rank=%d rows=%d active_rows=%s bytes=%d",
                        rank, rows, active_rows, rows * (3 * hidden + 5 * intermediate + 16))
        _, routes, indices, x, scale, gate_up, act, act_scale, down = cached
    else:
        rows = int(offsets[-1].item())
        routes = torch.empty(rows, dtype=torch.int32, device=y.device)
        indices = torch.empty_like(routes)
        x = torch.empty((rows, hidden), dtype=torch.int8, device=y.device)
        scale = torch.empty(rows, dtype=torch.float32, device=y.device)
    w4a8_ext.gather_routes(p.buffer, offsets, routes, indices, x, scale, *args,
                           device_rows, inverse, route_weights)
    if rows:
        if not device_rows:
            gate_up = torch.zeros((rows, 2 * intermediate), dtype=torch.bfloat16, device=y.device)
        grouped_gemm((x, scale), l1_weights, gate_up, indices, block_m=block_m1, n_loop=n_loop1)
        # Match the framework reference's BF16 activation and INT8 rounding.
        from lightop import fuse_silu_mul_clamp_quant
        from lightop.activation import fuse_silu_mul_quant

        if device_rows:
            from lightop import fuse_silu_mul_clamp_quant_ep
            fuse_silu_mul_clamp_quant_ep(
                gate_up.unsqueeze(0), float(activation_clamp), offsets[-1:],
                expect_m=min(rows, 4096), output=act.unsqueeze(0), scales=act_scale.unsqueeze(0))
        elif activation_clamp is None:
            act, act_scale = fuse_silu_mul_quant(gate_up)
        else:
            act, act_scale = fuse_silu_mul_clamp_quant(gate_up, float(activation_clamp))
        if not device_rows:
            down = torch.empty((rows, hidden), dtype=torch.bfloat16, device=y.device)
        grouped_gemm((act, act_scale), l2_weights, down, indices, block_m=block_m2, n_loop=n_loop2)
        if rank_partials:
            w4a8_ext.scatter_rank_partials(p.buffer, inverse, route_weights, down, *args)
        else:
            w4a8_ext.scatter_routes(p.buffer, routes, down, *args,
                                   os.environ.get("MEGAMOE_W4A8_VECTOR_SCATTER", "1") == "1")
    rank_barrier(p, rank_idx=rank, num_ranks=ranks, barrier_signal_slot_base=20)
    if tokens:
        w4a8_ext.combine_routes(p.buffer, combine_group_map, y, *args,
                                os.environ.get("MEGAMOE_W4A8_COMPACT_COMBINE", "1") == "1", rank_partials)
    # Combine reads only local slots. The next start barrier gates peer reuse,
    # and all earlier peer input reads have finished at the post-scatter barrier.
    if not reuse_combine:
        rank_barrier(p, rank_idx=rank, num_ranks=ranks, barrier_signal_slot_base=18)

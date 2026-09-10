"""W4A8 correctness first, followed by synchronized 10/100 timing.

Run from outside the source root after ``python setup.py install``:
  python /path/to/repo/tests/test_w4a8_mega_moe.py --gemm
  torchrun --standalone --nproc-per-node=8 /path/to/repo/tests/test_w4a8_mega_moe.py
"""

import argparse
import faulthandler
import json
import os
import time

import torch
import torch.distributed as dist

from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc as reference_gemm
from deepgemm import pack_w4a8_moe_hipc_weight
from deepgemm.group_gemm_weight_pack import pack_w4a8_weight_k_contiguous
from lightop import fuse_silu_mul_clamp_quant
from lightop.activation import fuse_silu_mul_quant
from lightop.quant import per_token_quant_int8
from lightop.moe import ep_gather
from megamoe import SymmBuffer, get_symm_buffer_for_mega_moe
from megamoe.w4a8 import grouped_gemm, w4a8_mega_moe


def packed_weight(e, n, k):
    integer = torch.randint(-8, 8, (e, n, k), device="cuda", dtype=torch.int8)
    packed = pack_w4a8_weight_k_contiguous(integer & 15)
    weight = pack_w4a8_moe_hipc_weight(packed)
    scale = torch.rand((e, n, 1), device="cuda", dtype=torch.float32) * 0.01 + 0.001
    return (weight, scale), integer


def timing(fn, byte_count, warmup=10, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    values = torch.tensor(samples, dtype=torch.float64)
    mean = values.mean().item()
    return dict(mean_us=mean, p50_us=values.quantile(0.5).item(),
                p90_us=values.quantile(0.9).item(), cv=values.std().item() / mean,
                effective_GB_s=byte_count / mean / 1000)


def test_gemm(args):
    e = args.experts // 8
    for n, k in ((2 * args.intermediate, args.hidden), (args.hidden, args.intermediate)):
        weight, integer = packed_weight(e, n, k)
        for valid_per_expert in (1, 63, 64, 65, args.tokens):
            padded = ((valid_per_expert + 63) // 64) * 64
            m = e * padded
            x = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
            scale = torch.rand((m, 1), device="cuda") * .01 + .001
            indices = torch.arange(e, device="cuda", dtype=torch.int32).repeat_interleave(padded)
            valid = torch.arange(m, device="cuda") % padded < valid_per_expert
            indices[~valid] = -1
            ref = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)
            out = torch.zeros_like(ref)
            reference_gemm((x, scale), weight, ref, indices)
            grouped_gemm((x, scale), weight, out, indices)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            oracle = (x[:valid_per_expert].float() @ integer[0].float().T)
            oracle = (oracle * scale[:valid_per_expert] * weight[1][0].T).bfloat16()
            torch.testing.assert_close(out[:valid_per_expert], oracle, atol=.01, rtol=.01)
            row = dict(test="gemm", E=e, M=m, valid_per_expert=valid_per_expert,
                       N=n, K=k, correctness="exact DeepGEMM parity", dtype="w4a8/bf16",
                       selected_card=os.environ.get("HIP_VISIBLE_DEVICES"))
            if valid_per_expert == args.tokens and not args.correctness_only:
                byte_count = x.numel() + weight[0].numel() + out.numel() * 2 + scale.numel() * 4 + weight[1].numel() * 4
                row["baseline"] = timing(lambda: reference_gemm((x, scale), weight, ref, indices), byte_count)
                row["candidate"] = timing(lambda: grouped_gemm((x, scale), weight, out, indices), byte_count)
            print(json.dumps(row), flush=True)
        bad_indices = torch.full((64,), e, device="cuda", dtype=torch.int32)
        untouched = torch.full((64, n), .5, device="cuda", dtype=torch.bfloat16)
        grouped_gemm((x[:64], scale[:64]), weight, untouched, bad_indices)
        torch.testing.assert_close(untouched, torch.full_like(untouched, .5), atol=0, rtol=0)
        print(json.dumps(dict(test="gemm_invalid_expert", N=n, K=k, correctness="guarded")), flush=True)


def reference_moe(all_x, all_ids, all_weights, l1, l2, rank, local_e, clamp, group_map=None):
    ranks, cap, hidden = all_x.shape
    topk = all_ids.shape[-1]
    result = torch.zeros((ranks * cap * topk, hidden), device="cuda", dtype=torch.bfloat16)
    qx, sx = per_token_quant_int8(all_x.reshape(-1, hidden))
    ids = all_ids.reshape(-1)
    weights = all_weights.reshape(-1)
    for expert in range(local_e):
        route = torch.where(ids == rank * local_e + expert)[0]
        count = route.numel()
        if not count:
            continue
        rows = ((count + 63) // 64) * 64
        tokens = route // topk
        x = torch.zeros((rows, hidden), dtype=torch.int8, device="cuda")
        scale = torch.ones((rows, 1), device="cuda")
        x[:count] = qx[tokens]
        scale[:count] = sx.reshape(-1, 1)[tokens]
        indices = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
        indices[:count] = expert
        gu = torch.zeros((rows, l1[1].shape[1]), dtype=torch.bfloat16, device="cuda")
        reference_gemm((x, scale), l1, gu, indices)
        act, act_scale = (fuse_silu_mul_quant(gu) if clamp is None
                          else fuse_silu_mul_clamp_quant(gu, clamp))
        down = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
        reference_gemm((act, act_scale), l2, down, indices)
        result[route] = down[:count]
    dist.all_reduce(result)
    local_ids = all_ids[rank]
    groups = local_ids // local_e if group_map is None else group_map[local_ids.clamp_min(0)]
    groups = torch.where(local_ids >= 0, groups, -1)
    inv_perm = (torch.arange(cap * topk, device="cuda", dtype=torch.int32)
                + rank * cap * topk).reshape(cap, topk)
    partial = torch.empty((cap, hidden), device="cuda", dtype=torch.bfloat16)
    # Use the framework's actual gather kernel: FP32 FMA within each rank,
    # BF16 rank partials, then the independently probed DeepEP rank reduction.
    combined = torch.zeros((cap, hidden), device="cuda", dtype=torch.float32)
    for owner in range(ranks):
        local_topk = torch.where(groups == owner, local_ids, -1)
        ep_gather(result, local_topk, all_weights[rank], inv_perm, None, partial)
        combined.add_(partial.float())
    return combined.bfloat16()


def test_distributed(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.manual_seed(100 + rank)
    e = args.experts // 8
    l1, _ = packed_weight(e, 2 * args.intermediate, args.hidden)
    l2, _ = packed_weight(e, args.hidden, args.intermediate)
    capacity = ((args.tokens + 63) // 64) * 64
    buf = get_symm_buffer_for_mega_moe(
        dist.group.WORLD, args.experts, capacity, args.topk,
        args.hidden, args.intermediate, quant_mode="w4a8")
    assert buf.num_max_tokens_per_rank == capacity
    benchmark_inputs = None
    try:
        for case in ("uniform", "skew", "ragged", "empty", "static_map"):
            tokens = args.tokens if case != "empty" else 0
            if case == "ragged":
                tokens = args.tokens * rank // 7
            group_map = None
            if case == "static_map":
                logical_ids = torch.arange(args.experts, device="cuda").reshape(-1, 8).T.contiguous().flatten()
                group_map = (logical_ids // e).to(torch.int32)
            x = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16) * .1
            ids = torch.rand((args.tokens, args.experts), device="cuda").topk(args.topk, dim=1).indices
            if case == "skew":
                ids[:] = torch.arange(args.topk, device="cuda")
            ids[tokens:] = -1
            weights = torch.rand((args.tokens, args.topk), device="cuda")
            weights /= weights.sum(1, keepdim=True)
            gathered = []
            for t in (x, ids, weights):
                parts = [torch.empty_like(t) for _ in range(8)]
                dist.all_gather(parts, t)
                gathered.append(torch.stack(parts))
            qx, sx = per_token_quant_int8(x)
            buf.x[:args.tokens].copy_(qx)
            buf.x_sf[:args.tokens].copy_(sx.reshape(-1))
            buf.topk_idx[:args.tokens].copy_(ids)
            buf.topk_weights[:args.tokens].copy_(weights)
            y = torch.empty((tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
            if case == args.benchmark_case:
                benchmark_inputs = (qx, sx, ids, weights, y)
            for clamp in (None, 10.0):
                ref = reference_moe(*gathered, l1, l2, rank, e, clamp, group_map)[:tokens]
                fn = lambda: w4a8_mega_moe(y, l1, l2, buf, activation_clamp=clamp,
                                          combine_group_map=group_map)
                fn()
                torch.cuda.synchronize()
                mismatch = (y != ref).sum()
                if mismatch.item():
                    delta = (y.float() - ref.float()).abs()
                    print(f"rank {rank}: {case} clamp={clamp} mismatches={mismatch.item()} "
                          f"max_abs={delta.max().item()}", flush=True)
                dist.all_reduce(mismatch)
                if mismatch.item():
                    raise AssertionError(f"distributed MoE has {mismatch.item()} mismatched elements")
                # Keep slow host-side first-use validation out of peer barrier
                # timeout windows when faster ranks begin the next measurement.
                dist.barrier()
                torch.cuda.synchronize()
                if rank == 0:
                    print(json.dumps(dict(test="moe", case=case, clamp=clamp,
                                          correctness="exact DeepGEMM composition parity")), flush=True)
        if not args.correctness_only:
            qx, sx, ids, weights, y = benchmark_inputs
            buf.x[:args.tokens].copy_(qx)
            buf.x_sf[:args.tokens].copy_(sx.reshape(-1))
            buf.topk_idx[:args.tokens].copy_(ids)
            buf.topk_weights[:args.tokens].copy_(weights)
            fn = lambda: w4a8_mega_moe(y, l1, l2, buf, activation_clamp=10.0)
            if os.environ.get("W4A8_DEBUG_STACKS") == "1":
                print(f"rank {rank}: entering timing", flush=True)
                faulthandler.dump_traceback_later(15, repeat=True)
            bytes_ = l1[0].numel() + l2[0].numel() + args.tokens * args.hidden * 3
            env_key = ("MEGAMOE_W4A8_RANK_PARTIALS" if args.paired_rank_partials else
                       "MEGAMOE_W4A8_COMPACT_COMBINE" if args.paired_compact_combine else
                       "MEGAMOE_W4A8_REUSE_COMBINE" if args.paired_reuse_combine else
                       "MEGAMOE_W4A8_DEVICE_ROWS" if args.paired_device_rows else "MEGAMOE_W4A8_VECTOR_SCATTER")
            paired = (args.paired_scatter or args.paired_device_rows or args.paired_reuse_combine or
                      args.paired_compact_combine or args.paired_rank_partials)
            modes = ("0", "1") if paired else (os.environ.get(env_key, "1"),)
            for repeat in range(3 if paired else 1):
                for mode in modes:
                    os.environ[env_key] = mode
                    stats = timing(fn, bytes_)
                    if rank == 0:
                        print(json.dumps(dict(test="moe_timing", routing=args.benchmark_case, config={env_key: mode},
                                              repeat=repeat, clamp=10.0, **stats)), flush=True)
            faulthandler.cancel_dump_traceback_later()
    finally:
        buf.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--paired-scatter", action="store_true")
    parser.add_argument("--paired-device-rows", action="store_true")
    parser.add_argument("--paired-reuse-combine", action="store_true")
    parser.add_argument("--paired-compact-combine", action="store_true")
    parser.add_argument("--paired-rank-partials", action="store_true")
    parser.add_argument("--benchmark-case", choices=("uniform", "skew"), default="uniform")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=65)
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.gemm:
        test_gemm(args)
    else:
        test_distributed(args)

"""Train HIPC tile choices from captured per-expert token counts."""

import argparse
import json
import time
from pathlib import Path

import torch

from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc as reference
from deepgemm import pack_w4a8_moe_hipc_weight
from deepgemm.group_gemm_weight_pack import pack_w4a8_weight_k_contiguous
from megamoe.w4a8 import grouped_gemm

TILES = [(m, n) for m in (32, 64, 128, 256) for n in (1, 2, 4) if (m, n) != (256, 4)]


def bench(fn, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    t = torch.tensor(times, dtype=torch.float64)
    return dict(mean_us=t.mean().item(), p50_us=t.quantile(.5).item(),
                p90_us=t.quantile(.9).item(), cv=t.std().item() / t.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--case-ids", type=int, nargs="*")
    parser.add_argument("--tiles", nargs="*")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    for index, case in enumerate(cases):
        case["case_id"] = index
    if args.case_ids is not None:
        cases = [x for x in cases if x["case_id"] in args.case_ids]
    elif not args.all:
        selected = set(range(min(args.limit, len(cases))))
        for nk in sorted({(x["n"], x["k"]) for x in cases}):
            items = sorted((x for x in cases if (x["n"], x["k"]) == nk), key=lambda x: x["m"])
            for q in (0, .25, .5, .75, .9, .99, 1):
                selected.add(items[round((len(items) - 1) * q)]["case_id"])
        cases = [x for x in cases if x["case_id"] in selected]
    cases.sort(key=lambda x: (x["n"], x["k"], x["m"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260910)
    current_nk = None
    tiles = [tuple(map(int, x.split("x"))) for x in args.tiles] if args.tiles else TILES
    with args.output.open("w") as stream:
        for case in cases:
            n, k = case["n"], case["k"]
            counts = case["counts"]
            e = len(counts)
            if (n, k) != current_nk:
                raw = pack_w4a8_weight_k_contiguous(torch.randint(0, 16, (e, n, k), device="cuda", dtype=torch.int8))
                weight = pack_w4a8_moe_hipc_weight(raw)
                sw = torch.rand((e, n, 1), device="cuda") * .01 + .001
                current_nk = (n, k)
            valid_m = sum(counts)
            logical_x = torch.randint(-127, 128, (valid_m, k), dtype=torch.int8, device="cuda")
            logical_sx = torch.rand((valid_m, 1), device="cuda") * .01 + .001
            buffers = {}
            for alignment in (64, 128, 256):
                padded = [((c + alignment - 1) // alignment) * alignment for c in counts]
                m = sum(padded)
                x = torch.zeros((m, k), dtype=torch.int8, device="cuda")
                sx = torch.ones((m, 1), device="cuda")
                ids = torch.full((m,), -1, dtype=torch.int32, device="cuda")
                out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                src = dst = 0
                for expert, (count, pad) in enumerate(zip(counts, padded)):
                    x[dst:dst + count].copy_(logical_x[src:src + count])
                    sx[dst:dst + count].copy_(logical_sx[src:src + count])
                    ids[dst:dst + count] = expert
                    src += count
                    dst += pad
                buffers[alignment] = (x, sx, ids, out, ids >= 0)
            x, sx, ids, out, valid = buffers[64]
            reference((x, sx), (weight, sw), out, ids)
            expected = out[valid].clone()
            base = bench(lambda: reference((x, sx), (weight, sw), out, ids), args.repeats) if not args.profile_only else {}
            for block_m, n_loop in tiles:
                x, sx, ids, out, valid = buffers[max(64, block_m)]
                fn = lambda: grouped_gemm((x, sx), (weight, sw), out, ids, block_m=block_m, n_loop=n_loop)
                fn()
                torch.cuda.synchronize()
                actual = out[valid]
                mismatches = int((actual != expected).sum())
                row = dict(case_id=case["case_id"], source_m=case["m"], padded_m=x.shape[0],
                           valid_m=valid_m, n=n, k=k, block_m=block_m, n_loop=n_loop,
                           frequency=case["frequency"], mismatches=mismatches, reference=base)
                if mismatches:
                    row["max_abs"] = (actual.float() - expected.float()).abs().max().item()
                    row["correctness"] = "failed"
                else:
                    row["correctness"] = "exact"
                    if not args.profile_only:
                        row.update(bench(fn, args.repeats))
                        row["speedup_vs_reference"] = base["mean_us"] / row["mean_us"]
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

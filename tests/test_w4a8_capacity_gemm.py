"""Compare real active rows against the device-row workspace capacity."""

import argparse
import json
import os
import time

import torch

from test_w4a8_mega_moe import packed_weight, timing
from megamoe.w4a8 import grouped_gemm
from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc as reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--rows-per-expert', type=int, default=768)
    parser.add_argument('--k', type=int, choices=(2048, 4096))
    parser.add_argument('--tile', default='64x4')
    args = parser.parse_args()
    torch.manual_seed(20260912)
    e, n = 32, 4096
    rows = e * ((args.rows_per_expert + 255) // 256) * 256
    capacity = (8 * 4096 * 6 + e * 63 + 63) // 64 * 64
    tiles = [(m, nl) for m in (32, 64, 128, 256) for nl in (1, 2, 4)
             if (m, nl) != (256, 4)] if args.sweep else [tuple(map(int, args.tile.split('x')))]
    for k in ((args.k,) if args.k else (4096, 2048)):
        pack_start = time.perf_counter()
        weight, _ = packed_weight(e, n, k)
        torch.cuda.synchronize()
        print(json.dumps(dict(stage='weight_fixture_and_packing', k=k,
                              elapsed_s=time.perf_counter() - pack_start)), flush=True)
        x = torch.randint(-127, 128, (capacity, k), device='cuda', dtype=torch.int8)
        sx = torch.rand((capacity, 1), device='cuda') * .01 + .001
        ids = torch.full((capacity,), -1, device='cuda', dtype=torch.int32)
        ids[:rows] = torch.arange(e, device='cuda', dtype=torch.int32).repeat_interleave(rows // e)
        expected = torch.empty((rows, n), device='cuda', dtype=torch.bfloat16)
        reference((x[:rows], sx[:rows]), weight, expected, ids[:rows])
        out = torch.empty((capacity, n), device='cuda', dtype=torch.bfloat16)
        for bm, nl in tiles:
            for allocated in (rows, capacity):
                def run():
                    grouped_gemm((x[:allocated], sx[:allocated]), weight,
                                 out[:allocated], ids[:allocated], block_m=bm, n_loop=nl)
                run()
                torch.cuda.synchronize()
                torch.testing.assert_close(out[:rows], expected, atol=0, rtol=0)
                if args.profile:
                    for _ in range(2):
                        run()
                    torch.cuda.synchronize()
                    stats = {}
                else:
                    bytes_ = rows * (k + 2 * n + 4) + weight[0].numel() + weight[1].numel() * 4
                    stats = timing(run, bytes_)
                print(json.dumps(dict(active_m=rows, allocated_m=allocated, n=n, k=k,
                                      block_m=bm, n_loop=nl, correctness='exact',
                                      card=os.environ.get('HIP_VISIBLE_DEVICES'), **stats)), flush=True)


if __name__ == '__main__':
    main()

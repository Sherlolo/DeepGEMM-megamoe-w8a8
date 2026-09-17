"""Measure clamped activation layouts using bounded preallocated outputs."""

import json
import os

import torch
from lightop import fuse_silu_mul_clamp_quant, fuse_silu_mul_clamp_quant_ep
from test_w4a8_mega_moe import timing


torch.manual_seed(20260912)
for rows in (4096, 25600):
    hidden = 2048
    x = torch.randn((rows, hidden * 2), device='cuda', dtype=torch.bfloat16) * 4
    expected, expected_scale = fuse_silu_mul_clamp_quant(x, 10.0)
    out = torch.empty_like(expected)
    scale = torch.empty_like(expected_scale)
    active = torch.tensor([rows], device='cuda', dtype=torch.int32)
    for grid in (None, 512, 1024, 4096, 8192, rows):
        if grid is None:
            def run():
                fuse_silu_mul_clamp_quant(x, 10.0, output=out, scales=scale)
        else:
            def run():
                fuse_silu_mul_clamp_quant_ep(x.unsqueeze(0), 10.0, active,
                                            expect_m=grid, output=out.unsqueeze(0),
                                            scales=scale.unsqueeze(0))
        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
        print(json.dumps(dict(rows=rows, hidden=hidden, grid=grid, correctness='exact',
                              card=os.environ.get('HIP_VISIBLE_DEVICES'),
                              **timing(run, rows * (hidden * 5 + 4)))), flush=True)

"""Exact staging parity, boundary sentinels, then synchronized paired timing."""

import json
from types import SimpleNamespace

import torch
from lightop.quant import per_token_quant_int8
from megamoe.w4a8 import _stage_w4a8_inputs
from test_w4a8_mega_moe import timing

torch.manual_seed(20260912)
for tokens in (0, 1, 63, 65, 4096):
    capacity, hidden, topk = tokens + 8, 4096, 6
    buffer = SimpleNamespace(x=torch.full((capacity, hidden), -99, device='cuda', dtype=torch.int8),
                             x_sf=torch.full((capacity,), -99., device='cuda'),
                             topk_idx=torch.full((capacity, topk), -99, device='cuda', dtype=torch.int64),
                             topk_weights=torch.full((capacity, topk), -99., device='cuda'))
    ids = torch.randint(-1, 256, (tokens, topk), device='cuda',
                        dtype=torch.int64 if tokens % 2 == 0 else torch.int32)
    weights = torch.rand((tokens, topk), device='cuda')
    for magnitude in (0., 1e-12, 1e-6, .1, 10.):
        x = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16) * magnitude
        _stage_w4a8_inputs(x, ids, weights, buffer)
        if tokens:
            expected, scales = per_token_quant_int8(x)
            torch.testing.assert_close(buffer.x[:tokens], expected, atol=0, rtol=0)
            torch.testing.assert_close(buffer.x_sf[:tokens], scales.flatten(), atol=0, rtol=0)
            torch.testing.assert_close(buffer.topk_idx[:tokens], ids.to(torch.int64), atol=0, rtol=0)
            torch.testing.assert_close(buffer.topk_weights[:tokens], weights, atol=0, rtol=0)
        for tensor in (buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights):
            assert (tensor[tokens:] == -99).all()
        print(json.dumps(dict(tokens=tokens, magnitude=magnitude, correctness='exact')), flush=True)
    if tokens == 4096:
        def composition():
            qx, sx = per_token_quant_int8(x)
            buffer.x[:tokens].copy_(qx)
            buffer.x_sf[:tokens].copy_(sx.flatten())
            buffer.topk_idx[:tokens].copy_(ids)
            buffer.topk_weights[:tokens].copy_(weights)

        for repeat in range(3):
            for mode in ('0', '1'):
                fn = composition if mode == '0' else lambda: _stage_w4a8_inputs(x, ids, weights, buffer)
                print(json.dumps(dict(tokens=tokens, repeat=repeat, fused=mode,
                                      **timing(fn, tokens * (3 * hidden + topk * 24 + 4)))), flush=True)

small = SimpleNamespace(x=torch.empty((1, 4096), device='cuda', dtype=torch.int8),
                        x_sf=torch.empty(1, device='cuda'),
                        topk_idx=torch.empty((1, 6), device='cuda', dtype=torch.int64),
                        topk_weights=torch.empty((1, 6), device='cuda'))
try:
    _stage_w4a8_inputs(torch.empty((2, 4096), device='cuda', dtype=torch.bfloat16),
                       torch.empty((2, 6), device='cuda', dtype=torch.int64),
                       torch.empty((2, 6), device='cuda'), small)
except ValueError:
    print(json.dumps(dict(test='overflow_guard', correctness='rejected before launch')), flush=True)
else:
    raise AssertionError('oversized staging must be rejected')

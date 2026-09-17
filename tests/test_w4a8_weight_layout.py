"""Isolate one-time HIPC packing cost from weight fixture generation."""
import json
import os
import torch
from deepgemm import pack_w4a8_moe_hipc_weight
from deepgemm.group_gemm_weight_pack import pack_w4a8_weight_k_contiguous
from megamoe.w4a8 import grouped_gemm
from test_w4a8_mega_moe import timing

torch.manual_seed(20260912)
for k in (4096, 2048):
    e, n, m = 32, 4096, 64
    integer = torch.randint(-8, 8, (e, n, k), device='cuda', dtype=torch.int8)
    raw = pack_w4a8_weight_k_contiguous(integer & 15)
    packed = pack_w4a8_moe_hipc_weight(raw)
    x = torch.randint(-127, 128, (m, k), device='cuda', dtype=torch.int8)
    sx = torch.full((m, 1), .01, device='cuda')
    sw = torch.full((e, n, 1), .005, device='cuda')
    ids = torch.zeros(m, device='cuda', dtype=torch.int32)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    grouped_gemm((x, sx), (packed, sw), out, ids)
    oracle = ((x.float() @ integer[0].float().T) * sx * sw[0].T).bfloat16()
    torch.testing.assert_close(out, oracle, atol=.01, rtol=.01)
    print(json.dumps(dict(E=e, N=n, K=k, dtype='packed_int4',
                          correctness='PyTorch dot-product oracle',
                          selected_card=os.environ.get('HIP_VISIBLE_DEVICES'),
                          **timing(lambda: pack_w4a8_moe_hipc_weight(raw), 2 * raw.numel()))), flush=True)

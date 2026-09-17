"""Private fused W4A8 input quantization and communication-buffer staging.

Quantization matches LightOp's _lmslim_native/layers/gemm/int8_utils.py:
FP32 absmax clamped to 1e-10, 127/absmax, and nearbyint before INT8 conversion.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _stage_kernel(x, ids, weights, out_x, out_scale, out_ids, out_weights,
                  hidden: tl.constexpr, topk: tl.constexpr, block: tl.constexpr,
                  route_block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    values = tl.load(x + row * hidden + cols, cols < hidden, other=0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values)), 1e-10)
    scale = absmax / 127
    quantized = libdevice.nearbyint(values * (127 / absmax)).to(tl.int8)
    tl.store(out_x + row * hidden + cols, quantized, cols < hidden)
    tl.store(out_scale + row, scale)
    route_cols = tl.arange(0, route_block)
    expert = tl.load(ids + row * topk + route_cols, route_cols < topk, other=-1)
    weight = tl.load(weights + row * topk + route_cols, route_cols < topk, other=0)
    tl.store(out_ids + row * topk + route_cols, expert, route_cols < topk)
    tl.store(out_weights + row * topk + route_cols, weight, route_cols < topk)


def stage_inputs(x, ids, weights, buffer):
    tokens, hidden = x.shape
    if tokens == 0:
        return
    block = triton.next_power_of_2(hidden)
    _stage_kernel[(tokens,)](
        x, ids, weights, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
        hidden, ids.shape[1], block, triton.next_power_of_2(ids.shape[1]),
        num_warps=min(max(block // 256, 1), 8), num_stages=1)

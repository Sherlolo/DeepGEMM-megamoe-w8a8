# MegaMoE W8A8 INT8 — 4096 Token 优化报告

## 结论

在 EP8、288 experts、topk=8、hidden=4096、intermediate=2048、
BF16 I/O、W8A8 INT8、每 rank 4096 tokens/capacity 的固定负载上，最终
三组 10 warmup / 100 repeat 中位数为：

| 版本 | 三组中位数 (ms) | 均值 (ms) | 相对冻结基线 |
|---|---:|---:|---:|
| 冻结基线 `019b969` | 7.592597 / 7.605336 / 7.598797 | 7.598910 | — |
| 最终 `v4096_023` | 7.237177 / 7.254896 / 7.240377 | 7.244150 | **提升 4.6686%** |

目标 5% 对应 7.218964 ms。最终版本距离目标尚差 0.025186 ms
（0.3314 个百分点）。按本轮终止要求，没有继续引入新候选。
EP8 聚合吞吐由约 4.312M tokens/s 提高至约 4.523M tokens/s。

## 代码结构与编译流程

- `tests/test_mega_moe_int8_baseline.py`：EP8 多进程正确性与 TileLang event
  基准入口，对比 true INT8 normal-contiguous baseline。
- `csrc/kernels/mega_moe_baseline_hip.cu`：pre-dispatch、量化和通用 HIP
  小算子。
- `K1_fused/k1_fused_ext.cu` 与 K1 INT8 `.s`：compact route
  init/count/prefix/emit、输入 staging、第一层 DeepGEMM。
- `K2_fused/k2_fused_ext.cu`：SwiGLU、route weight、channelwise quant。
- `K3_fused/k3_fused_ext.cu` 与 K3 INT8 `.s`：第二层 DeepGEMM、
  combine store、rank barrier 与 local combine reduction。
- `scripts/build_dcu_megamoe.sh`：hipify C++/HIP 源，使用 gfx936 编译扩展，
  汇编 K1/K3 code object，经 `setup.py bdist_wheel` 打包并做 import、
  shared-object、code-object 完整性检查。

最终构建命令：

```bash
MEGAMOE_DCU_ARCH=gfx936 \
bash megamoe/dcu_megamoe_opt/scripts/build_dcu_megamoe.sh
```

## 最终保留的优化

1. K3 在汇编入口读取 device `active_tiles`，在 grouped-GEMM
   解析和 remap 之前结束 inactive capacity workgroups。
2. K1 通过 packed kernarg 直接传递 device `active_tiles`，去除依赖的
   `GpuProb` 指针追踪。
3. hidden=2048 的 INT8 K2 使用 128 threads / 4 vector groups，
   VGPR 从 68 降至 48，保持零 scratch。
4. hidden=4096 INT8 pre-dispatch 使用单次读取的 vec16 专用路径；
   FP8 和其他 shape 保留原路径。
5. K1 staging producer 从 8 调到 10，稳定整流程收益 0.5521%。
6. INT8 compact padding 的 combine pointer 使用 0；K3 每个 compute wave
   读取其 32-row slice 首个完整 64-bit pointer，仅对全 padding wave 跳过
   MMAC。B load、LDS、wait、循环计数、`SMQUANT`、store 与所有 barrier
   均保持原样。

## 正确性

- random route：3/3，通过；`max_abs=0.00048828125`，
  `mean_abs=1.031944793794537e-05`，routing stats exact。
- single-local-rank skew：3/3，通过；max/mean error 均为 0，
  routing stats exact。
- 所有失败或性能回退候选均已回退；不支持的 gfx936 packed-convert
  指令只在编译阶段失败，从未执行。

## hipprof 整流程证据

最终 full-flow trace 的主要 kernel 平均时间：

| Kernel/阶段 | 平均时间 |
|---|---:|
| K3 INT8 DeepGEMM + combine | 3.255516 ms |
| K1 INT8 DeepGEMM + dispatch | 3.069328 ms |
| rank barrier | 0.487471 ms |
| K2 SwiGLU + quant | 0.358881 ms |
| local combine reduce | 0.293047 ms |
| INT8 pre-dispatch vec16 | 0.063877 ms |
| k1 emit / count / init rows / build / init | 0.039170 / 0.025125 / 0.007193 / 0.005395 / 0.005229 ms |

K3 相对 `v4096_016` profile 的 3.344475 ms 下降 2.6599%。
完整 HIP trace 中未发现 DeviceToHost/D2H copy；active-tile 决策保持在
device 上，没有恢复 host scalar read 或全 stream 同步。

## 融合分析

- pre-dispatch 与 DeepEP dispatch 之间存在通信缓冲区所有权边界；
  将它直接塞入 K1 会改变跨 rank staging 协议，因此本轮只采用
  单次读 vec16 专用 kernel。
- compact route 的 count 与 emit 之间必须完成 expert tile prefix；
  `build_compact_tiles` 是全局依赖点。普通 kernel 内直接融合会需要
  cooperative global synchronization，并有驻留/死锁风险。
- rank barrier 的时间主要是在等待最慢 rank 的 K3，而不是自身指令。
  把 reduce 合进 barrier 会让等待 workgroups 占用 CU；已测试的 K3
  tail-reduce 路线出现明显回退，因此未保留。
- K2 已将 SwiGLU、route weight 和 channelwise quant 融合；继续扩大
  vector 宽度或 wave reduction 的候选没有稳定收益。

## 失败候选摘要

已测试并回退的方向包括：reduce vec16/线程数、K1 producer=12、
K2 更大 grid、K2 wave reduction、K3 persistent/remap、K3 store-wave
重排、K3 tail-reduce、全互联 rank barrier、pre-dispatch vec32、
静态展开 reduce，以及 gfx936 不支持的 packed int16 convert。
失败原因覆盖精度、nonfinite、汇编指令不支持或稳定性能回退。

## 交付物

- wheel：`build/whl/megamoe-0.1-cp310-cp310-linux_x86_64.whl`
- SHA256：
  `52187ea03f3ca4d9f11919951b5e426909acd7c20329fc2b31d07cf3f1e95267`
- profile：
  `.humanize/lightop-agent/profile-artifacts/v4096_023_k3_padding_wave_skip/`

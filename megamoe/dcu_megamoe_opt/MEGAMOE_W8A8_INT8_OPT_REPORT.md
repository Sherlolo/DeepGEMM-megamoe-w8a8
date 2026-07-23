# MegaMoE W8A8 INT8 DCU 优化报告

## 结论

在 `e08r3n03` 的 `sglang_glm_0721` 容器、8 张 `gfx936` DCU 上，EP8
`288 experts / topk=8 / H=4096 / I=2048 / 512 tokens per rank` 的
`megamoe_w8a8_int8` 端到端均值从 **2.054666 ms** 降到
**1.938959 ms**，提升 **5.6314%**。本轮继续优化相对已接受版本的
同环境父版本 1.945146 ms 再降 **0.3181%**；最终三轮极差/均值为
**0.2837%**，小于增益。相对现有 DeepEP + DeepGEMM INT8 reference
的可靠均值 2.931572 ms，最终路径约 **1.512x**。

精度保持不变：随机路由 3/3 次均为
`max_abs=0.00048828125`、`mean_abs=1.02747695e-05`、
`stats_exact=true`。首次优化完成时，单本地 rank 极端偏斜路由 3/3
为零误差，`capacity=1024, tokens=512` 邻近容量场景 3/3 通过；本轮
源码契约测试 33/33 通过。本轮收尾时节点上另有一个 8 卡 SGLang
服务占用约 56–57 GiB/卡，容量与偏斜复跑均在 reference 输出张量
分配阶段 OOM、尚未进入候选结果比较，因此不把这两次环境失败计为
新的精度结论。

## 仓库与编译流程

- Python 公共入口在 `megamoe/__init__.py` 和 `megamoe/opt.py`。
- pre-dispatch 和基线 HIP kernel 在
  `dcu_megamoe_opt/csrc/kernels/mega_moe_baseline_hip.cu`。
- K1/K2/K3 扩展分别位于 `K1_fused/`、`K2_fused/`、`K3_fused/`。
- K1/K3 主 GEMM 使用预编译 gfx936 code object，HIP 扩展负责构建
  route metadata、启动汇编 kernel 和后处理。
- 权威构建命令：

```bash
MEGAMOE_DCU_ARCH=gfx936 \
  bash megamoe/dcu_megamoe_opt/scripts/build_dcu_megamoe.sh
```

该命令完成 hipify、PyTorch HIP 扩展编译、gfx936 code object 检查、
wheel 打包、仓库内 `.so` 同步和 import 验证。

## 接受的优化

### 1. 去除 K1/K3 热路径 D2H

原 eager K1 在 compact route 后把 `active_tiles` 标量从 device 拷回
host，并执行 `hipStreamSynchronize`；K3 在没有 host hint 时再次执行
相同模式。gfx936 K1/K3 汇编已经读取 device active-tile 元数据并让
无效 workgroup 提前退出，因此改为按 capacity 启动、由 device gate
截断。热路径不再包含 active-tile D2H 和对应全 stream 同步。

该修改单独把三轮均值降到 1.978305 ms，提升 3.7165%。

### 2. K1 build 拆分为 prefix 与多 CU row init

原 `k1_build_compact_tiles_kernel` 用单个 1024-thread workgroup 在一个
CU 上初始化约 4.17 万行。PMC 显示 27,016 次 VMEM 写、约 0.8% L2
命中且无 LDS bank conflict，瓶颈是单 CU 并行度。

优化后，小型 prefix 留在一个 wave；独立行初始化由最多
128 x 256 threads grid-stride kernel 分布到多个 CU。完整剖析中，
原 42.6 us build 变为 16.6 us prefix + 5.1 us row init。

### 3. 36-expert prefix 使用 wave64 scan

每个 expert 由一个 lane 计算 tile 数，用 HIP
`__shfl_up(..., width=64)` 做 inclusive scan，再生成容量裁剪后的
exclusive base、tile-to-expert 映射和 active-tile 总数。prefix 从
16.6 us 降至约 5.0 us；prefix + row init 最终约 9.9 us，相比原
40.7 us build 明显下降。

曾尝试自写带分支的 `ds_bpermute` helper，但严格精度出现
`max_abs=0.097900390625`，在性能测试前即淘汰并记录。

### 4. K1 无效 capacity workgroup 在汇编入口退出

K1 原先虽然由 device `active_tiles` 截断，但判断位置在 grouped-GEMM
参数解析、workgroup remap 和较大汇编序言之后。最终版本把
`active_tiles` device pointer 通过 `GpuProb` ABI 传给 K1；由于当前
INT8 K1 固定输出宽度 4096、每个 compact row tile 对应 16 个 N 方向
workgroup，入口直接用 flat workgroup id 计算 row tile，并在无效时
立即结束。

该修改没有重新引入 D2H 或 stream synchronize，也没有改变 K1/K3
公共 Python API。K1 的 workgroup、LDS、VGPR、SGPR 和 scratch 资源
保持不变。

## 稳定性能

| 版本 | 三轮 fused median（ms） | 均值（ms） | 相对原始 |
|---|---:|---:|---:|
| v000 原始 | 2.055359 / 2.055419 / 2.053219 | 2.054666 | 基线 |
| v001 无 D2H | 1.975239 / 1.982239 / 1.977439 | 1.978305 | -3.7165% |
| v002 多 CU init | 1.957179 / 1.961319 / 1.959659 | 1.959386 | -4.6373% |
| v003b wave prefix | 1.948299 / 1.948219 / 1.946899 | 1.947806 | **-5.2008%** |
| v008 K1 入口 gate | 1.942199 / 1.937979 / 1.936699 | 1.938959 | **-5.6314%** |

所有稳定测量均使用相同 seed/shape、10 次 warmup、100 次 repeat，并
在每轮前记录设备状态。

## hipprof 与 PMC

完整流程使用 `hipprof --hip-trace --stats --follow-fork --show-pid`，
并对所有正确候选采集 `--pmc --pmc-type 3`、`--pmc-read`、
`--pmc-write`。最终 full trace 的八 rank 平均每次调用：

| kernel/阶段 | v000（us） | final（us） | 说明 |
|---|---:|---:|---|
| K1 ASM GEMM | 1013.5 | 1015.9 | profile 扰动范围内 |
| K3 ASM GEMM | 690.3 | 703.4 | profile 扰动范围内 |
| K2 SwiGLU + INT8 quant | 70.1 | 69.4 | 基本不变 |
| pre-dispatch | 19.6 | 19.7 | 基本不变 |
| K1 emit | 16.2 | 16.6 | 基本不变 |
| K1 count | 8.8 | 8.9 | 基本不变 |
| K1 build/prefix | 40.7 | 5.0 | wave64 scan |
| K1 row init | 合并在 build | 4.9 | 多 CU |
| K1 counter init | 4.9 | 5.0 | 后续可融合 |

`rank_barrier_kernel` 的 profile 时间在 rank 间高度偏斜，主要反映
等待最慢 rank，而非有效计算，因此不把其绝对值用于 kernel
吞吐比较。

本轮 K1 定向 PMC 相对继续优化前父版本：

| K1 指标 | 变化 |
|---|---:|
| active VALU instructions | -6.1984% |
| VALU instructions | -9.2649% |
| VMEM reads | -2.2107% |
| LDS wait instructions | -9.3294% |
| VMEM stores / LDS bank conflict | 不变 |

最终 K1 code object 仍为 768 threads/workgroup、64 KiB LDS、256 VGPR、
112 SGPR、0 scratch。最终 full trace 未出现 MegaMoE 热路径 D2H。
当前 hipprof 的 SQTT 路径不可用，standalone code-object analyzer 也
拒绝 raw `.co`，因此本轮以 PMC 与汇编 metadata 作为指令和资源证据。

## 融合可行性

- `count -> prefix/build -> init -> emit` 存在全 grid 数据依赖。普通
  multi-block HIP kernel 没有隐式 grid barrier，直接合并会产生竞态。
- pre-dispatch 与 K1 之间的 rank barrier 负责让 symmetric-buffer
  写入对 peer 可见，不能跨通信边界直接融合。
- 本轮实测过把 `k1_init_compact_routes_kernel` 并入 start
  `rank_barrier_kernel`：精度 3/3 通过并消除了一个 launch，但端到端
  从 1.945146 ms 变为 1.945905 ms（+0.0391%），处于噪声内且方向
  不利，因此已完整回退，不进入最终源码。
- 仓库及本测试 trace 中不存在用户提到的
  `moe_fused_gate_kernel_gourp1`；测试接收已生成的 top-k tensor。
  gate 与 pre-dispatch 融合需要上游路由组件共同修改，不能在
  DeepGEMM 仓库内单独安全完成。

## 验证命令

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python megamoe/dcu_megamoe_opt/tests/test_mega_moe_int8_baseline.py \
  --num-processes 8 --num-max-tokens-per-rank 512 --num-tokens 512 \
  --correctness-iters 3 --warmup 10 --repeat 100
```

偏斜和邻近容量分别追加：

```bash
--route-pattern single-local-rank --route-target-rank 0 --skip-bench
--num-max-tokens-per-rank 1024 --num-tokens 512 --skip-bench
```

源码契约：

```bash
python -m pytest -q \
  megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py
```

## 修改文件

- `K1_fused/k1_fused_ext.cu`
- `K1_fused/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_`
  `MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s`
- `K3_fused/k3_fused_ext.cu`
- `tests/test_dcu_megamoe_v3.py`
- 本报告

所有原始/候选 benchmark、hipprof、PMC、设备状态和失败候选日志保留
在优化工作树 `.humanize/lightop-agent/` 下。

# MegaMoE W8A8 INT8 优化汇总

## 1. 优化范围

本轮针对 DCU gfx936 上的 MegaMoE W8A8 INT8 全流程进行优化，覆盖：

- gate、pre-dispatch 与路由准备；
- K1 compact route、输入 staging 和第一层 DeepGEMM；
- K2 SwiGLU、route weight 与 channelwise quant；
- K3 第二层 DeepGEMM、combine store、rank barrier 和 local reduce；
- Device/Host 数据流、构建、正确性、整流程性能分析与 whl 打包。

主要测试负载为 EP8、288 experts、topk=8、hidden=4096、
intermediate=2048、每 rank 4096 tokens/capacity。

## 2. 最终性能

| 负载 | 优化前 | 最优版本 | 提升 |
|---|---:|---:|---:|
| 512 tokens/rank | 2.054666 ms | 1.938959 ms | 5.6314% |
| 4096 tokens/rank | 7.598910 ms | 7.244150 ms | 4.6686% |

4096-token 目标线为 7.218964 ms，最终距离 5% 目标约 0.025186 ms。
EP8 聚合吞吐约由 4.312M 提升至 4.523M tokens/s。

## 3. 最终保留的优化点

### 数据流与调度

1. 去除 K1/K3 active-tile 的中间 D2H 拷贝和 stream synchronization，
   active-tile 判断全部保留在 device 侧。
2. K1/K3 在汇编入口提前过滤 inactive capacity workgroups，减少无效
   grouped-GEMM 解析、remap 和计算。
3. K1 通过 packed kernarg 直接传递 device `active_tiles` 指针，减少依赖
   指针读取。

### 路由与小算子

4. 将 K1 tile-prefix 构建与 capacity-row 初始化拆开，并行完成初始化。
5. expert tile prefix 改为 wave64 collective prefix scan。
6. hidden=4096 的 INT8 pre-dispatch 使用单次读取的 vec16 专用 kernel。
7. K1 staging producer 数量由 8 调整为 10。
8. 对 `k1_init/count/build/emit_compact_routes`、gate、rank barrier 和
   local reduce 进行了整流程 profiling；保留有稳定收益的实现。

### K2 与 K3 核心算子

9. hidden=2048 的 INT8 K2 改为 128 threads / 4 vector groups，VGPR
   从 68 降至 48，并保持 zero scratch。
10. INT8 compact padding 的 combine pointer 设为 0。
11. K3 对完整 padding 的 32-row compute wave 跳过 MMAC，同时保留
    load、LDS、barrier、量化和 store 的原有同步关系。

## 4. 融合与 D2H 结论

- 最终 hipprof trace 中未发现 DeviceToHost/D2H 拷贝。
- K2 已融合 SwiGLU、route weight 和 channelwise quant。
- pre-dispatch 与 K1 之间存在通信缓冲区所有权边界，未做强行融合。
- compact count、prefix、emit 之间存在全局依赖，直接合并有全局同步和
  驻留风险。
- rank barrier 主要反映各 rank 的计算等待；将 reduce 合入 barrier 或
  K3 tail 的方案实测回退，因此未保留。

## 5. 已测试但未采用的方向

- K1 producer=12；
- local reduce vec16、64/256 threads 和静态展开；
- K2 更大 grid、wave reduction 和其他 vector geometry；
- K3 persistent/remap、store-wave 重排和 tail-reduce；
- pre-dispatch vec32；
- 全互联 rank barrier；
- gfx936 不支持的 packed int16 convert。

这些候选因性能不稳定、出现回退、精度失败或指令不受支持而全部回退，
最终源码中没有保留失败候选。

## 6. 正确性与交付

- random route：3/3 通过，`max_abs=0.00048828125`，routing stats exact。
- single-local-rank skew：3/3 通过，误差为 0，routing stats exact。
- 最终提交：`cfb416f266a96b4d2edea394b3528ec2ad306cc7`
- 最终分支：`codex/megamoe-w8a8-int8-opt`
- whl：`build/whl/megamoe-0.1-cp310-cp310-linux_x86_64.whl`
- SHA256：
  `52187ea03f3ca4d9f11919951b5e426909acd7c20329fc2b31d07cf3f1e95267`

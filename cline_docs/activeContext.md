
# 当前工作（2025-09-14）

SpikeNet-X 在 DBLP 上出现 **训练很慢 & 指标仅 ~0.3** 的异常，根因已排查并形成补丁：

- **子图边集构造错误**：之前只保留了“种子 → 邻居”的边，导致多层传播被掐断；已修复为**保留子图内所有边（src/dst ∈ 子图节点）**。
- **读出策略**：SpikeNet-X 默认 `readout="last"`，对抖动敏感；已统一为 **`readout="mean"`**。
- **注意力实现选择**：稀疏 STA 当前实现不支持 Top-k，且在小子图上速度欠佳；训练侧改为 **`attn_impl=dense`**，以获得矩阵并行和可用的 Top-k。
- **邻居采样**：将 `num_neighbors_to_sample` 提升至 **25**（或 `-1` 全邻居）以稳定覆盖。

## 最近的变更

1. **`sample_subgraph` 修复**：保留子图内部所有边，并使用纯 Torch 做“全局 → 局部”映射，移除 Python 字典热点。
2. **模型构造**：`SpikeNetX(..., readout="mean", attn_impl=...)`；命令行默认推荐 `--attn_impl dense`。
3. **训练脚本**：将 `num_neighbors_to_sample` 从 10 → 25；建议 `--W 32 --topk 8` 起步。

## 当前状态

- SpikeNet（SAGE+SNN）F1≈0.75（基线稳）。
- SpikeNet-X 预期：修复后应恢复到“与 SpikeNet 同量级”的区间；具体以本周全量训练为准。

## 下一步计划

- 以 DBLP 为基线，跑满 100 epoch，记录 Macro/Micro-F1、训练时长与显存曲线。
- 做 `W∈{16,32,64}` × `topk∈{8,16}` × `heads∈{2,4}` 的小网格。
- 若需大图扩展，再切回 `attn_impl=sparse` 并开发 CUDA/高阶算子版本的 Top-k。

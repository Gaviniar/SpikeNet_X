
# Spike-TDANet: 脉冲时序延迟注意力网络

## 目录

1. [项目概述](#项目概述)
2. [模型架构与创新点](#模型架构与创新点)
   * [核心架构哲学：先空间，后时间](#核心架构哲学先空间后时间)
   * [三大创新组件](#三大创新组件)
3. [数据处理流程 (单层)](#数据处理流程-单层)
4. [项目文件结构](#项目文件结构)
5. [模块详解](#模块详解)
   * [主程序 (`main.py`)](#主程序-mainpy)
   * [顶层模型 (`spikenet_x/model.py`)](#顶层模型-spikenet_xmodelpy)
   * [核心编排层 (`spikenet_x/spiketdanet_layer.py`)](#核心编排层-spikenet_xspiketdanet_layerpy)
   * [空间GNN预处理器 (`spikenet_x/new_modules/spatial_gnn_wrapper.py`)](#空间gnn预处理器-spikenet_xnew_modulesspatial_gnn_wrapperpy)
   * [可学习延迟通路 (`spikenet_x/new_modules/delay_line.py`)](#可学习延迟通路-spikenet_xnew_modulesdelay_linepy)
   * [脉冲时序注意力聚合器 (`spikenet_x/new_modules/sta_gnn_agg.py`)](#脉冲时序注意力聚合器-spikenet_xnew_modulessta_gnn_aggpy)
   * [LIF神经元 (`spikenet_x/lif_cell.py`)](#lif神经元-spikenet_xlif_cellpy)
   * [辅助模块](#辅助模块)
6. [如何运行](#如何运行)

---

## 1. 项目概述

**Spike-TDANet (Spiking Temporal Delay Attention Network)** 是一个为处理时序图数据而设计的下一代脉冲图神经网络（SNN-GNN）。该模型旨在通过一个统一、强大且稳定的新架构，解决动态图中复杂的时空依赖性建模问题。

本项目的核心目标是将一个基础的脉冲网络 (`SpikeNet-X`) 彻底重构，集成三大先进理念：

1. **空间GNN预处理 (Spatial GNN Preprocessing)**：在时序处理前增强节点特征的空间上下文。
2. **可学习延迟通路 (Learnable DelayLine)**：显式地、高效地为信号传播建模多种时间延迟。
3. **脉冲时序注意力聚合 (Spiking Temporal Attention Aggregator)**：执行一种事件驱动的、因果的、稀疏化的时空信息聚合。

最终形成一个逻辑清晰、性能卓越且易于训练的端到端模型。

**Tensor 形状约定**: 在本文档中，时序节点特征张量的形状统一表示为 `[T, N, d]`，其中 `T` 是时间步数，`N` 是节点数，`d` 是特征维度。

## 2. 模型架构与创新点

### 核心架构哲学：先空间，后时间

为了最大化模型的稳定性和表达能力，我们将复杂的时空信息处理解耦为两个清晰的阶段：

1. **阶段一：空间特征增强 (由标准GNN负责)**

   * **任务**: 在每个时间点独立地应用图卷积，平滑节点特征并聚合其邻域的静态上下文信息。
   * **作用**: 为后续的时序模块提供一个高质量、低噪声、信息丰富的输入。这极大地稳定了训练过程，避免了在原始、嘈杂的特征上直接进行复杂的时序建模。
2. **阶段二：时序动态建模 (由SNN核心模块负责)**

   * **任务**: 在经过空间增强的特征之上，显式地建模时间延迟，执行跨时间的注意力聚合，并以事件驱动的方式产生脉冲。
   * **作用**: 专注于捕捉图中复杂的时间依赖关系和因果传播模式。

### 三大创新组件

1. **空间GNN预处理 (`SpatialGNNWrapper`)**

   * **创新**: 将GNN作为时序SNN的前置“降噪器”和“特征增强器”。通过一个高效的Wrapper，将 `T`个时间点的图卷积操作并行化为一次大的批处理运算，实现了在不牺牲性能的前提下，为每个时间点注入必要的空间信息。
2. **可学习延迟通路 (`DelayLine`)**

   * **创新**: 使用**因果深度可分离1D卷积**来替代传统的固定延迟或隐式建模。该模块能以极低的计算成本，为每个特征通道学习一组不同的延迟响应权重，从而动态地捕捉不同类型信息在时间维度上的传播速度差异。
3. **脉冲时序注意力聚合器 (`STAGNNAggregator`)**

   * **创新**: 这是模型的核心计算单元，实现了高度稀疏化和事件驱动的注意力机制。
     * **时空注意力**: Attention机制同时作用于**空间邻居**和**时间窗口**内的历史信息。
     * **因果与稀疏**: 严格遵守因果性（只关注过去），并且注意力计算被限制在图的边结构上（`O(E)`复杂度），适用于大规模图。
     * **脉冲门控 (Spike-Gated)**: 注意力分数受到源节点历史脉冲活动的调制。这意味着只有“活跃”（即近期发放过脉冲）的邻居节点才能传递更重要的信息，完美契合SNN的事件驱动特性。
     * **数值稳定**: 采用两遍(two-pass)的 `segment-softmax`机制，确保在稀疏图上进行softmax时数值的稳定性和准确性。

## 3. 数据处理流程 (单层)

`SpikeTDANetLayer` 是模型的基本构建块。其单层数据处理流程如下：

1. **输入**:

   * `x`: 节点特征 `[T, N, d]`
   * `spikes`: 上一层或初始脉冲 `[T, N]`
   * `edge_index`: 图结构 `[2, E]`
   * `time_idx`: 时间索引 `[T]`
2. **[空间GNN预处理] `SpatialGNNWrapper`**:

   * 对每个时间步的 `x` 应用一次 `SAGEConv`，输出空间聚合后的特征 `x_spatial`。
   * 应用残差连接和层归一化: `x_norm1 = LayerNorm(x + x_spatial)`。
3. **[时间延迟建模] `DelayLine`**:

   * 将 `x_norm1` 输入因果深度可分离1D卷积，学习多种延迟响应，输出 `x_delayed`。
   * 应用残差连接和层归一化: `x_norm2 = LayerNorm(x_norm1 + x_delayed)`。
4. **[时空信息聚合] `STAGNNAggregator`**:

   * 以 `x_norm2` 作为Query, Key, Value的基础，`spikes` 作为门控信号。
   * 执行因果、稀疏化的时空多头注意力，聚合来自时空邻域的信息。
   * 输出聚合后的消息 `aggregated_message` `[T, N, d]`。
5. **[脉冲发放] `LIFCell`**:

   * 将 `aggregated_message` 通过一个线性层 `msg_proj` 投影为标量输入电流 `I` `[T, N]`。
   * 将电流 `I` 注入LIF神经元，更新膜电位并生成新脉冲。
   * 输出 `new_spikes` `[T, N]` (用于下一层) 和 `new_v` `[T, N]` (膜电位)。
6. **[最终输出]**:

   * 将聚合消息 `aggregated_message` 通过一个前馈网络 (FFN) 进行变换。
   * 应用一个宏观的残差连接，将FFN的输出与该层的最开始输入 `x` 相加，确保梯度流的通畅。
   * `output_features = LayerNorm(x + ffn(aggregated_message))`。
7. **输出**:

   * `output_features`: `[T, N, d]`，作为下一层的特征输入。
   * `new_spikes`: `[T, N]`，作为下一层的脉冲门控输入。

## 4. 项目文件结构

```
spikenet_x/
├── __init__.py
├── model.py                 # (新) 顶层模型 SpikeTDANet
├── spiketdanet_layer.py     # (新) 核心编排层 SpikeTDANetLayer
├── lif_cell.py              # (新) 纯粹的LIF神经元
├── rel_time.py              # (新) 相对时间编码
├── masked_ops.py            # (新) 掩码与Top-k工具
└── new_modules/             # <-- 新建文件夹
    ├── __init__.py
    ├── spatial_gnn_wrapper.py # <-- 新建: 空间GNN预处理器
    ├── delay_line.py          # <-- 新建: 可学习多延迟通路
    └── sta_gnn_agg.py         # <-- 新建: 脉冲时序注意力GNN聚合器
```

## 5. 模块详解

### 主程序 (`main.py`)

* **文件目的**: 整个项目的入口，负责数据加载、模型训练、验证和测试的完整流程。
* **核心逻辑**:
  1. **参数解析**: 定义并解析命令行参数，如模型类型 (`--model spiketdanet`)、数据集、学习率、批大小等。
  2. **数据加载**: 使用 `spikenet.dataset` 加载 DBLP, Tmall, Patent 等时序图数据集。
  3. **子图采样 (`sample_subgraph`)**:
     * **作用**: 为了处理大规模图，训练时采用mini-batch方式。此函数负责为一个批次的中心节点（`nodes`）采样一个1-hop邻域子图。
     * **输入**: `nodes` (中心节点ID `[B]`), `edge_index_full` (全图的边 `[2, E_full]`), `num_neighbors` (邻居采样数)。
     * **输出**: `subgraph_nodes` (子图包含的所有节点ID), `subgraph_edge_index` (在子图节点上的局部边索引), `nodes_local_index` (中心节点在子图中的索引)。
  4. **模型初始化**: 根据 `--model` 参数选择实例化 `SpikeTDANet`。
  5. **训练循环 (`train_model`)**:
     * 遍历 `train_loader`，对每个batch进行子图采样。
     * 将子图的特征和边索引送入模型，得到 logits。
     * 计算损失函数 (CrossEntropyLoss)，并加入一个脉冲发放率的正则项来鼓励稀疏激发。
     * 反向传播并更新模型参数。
  6. **评估逻辑 (`test_model`)**:
     * 在验证集或测试集上进行评估，同样使用子图采样。
     * 计算 F1-score (Micro 和 Macro) 作为评估指标。
  7. **检查点管理**: 支持模型的保存、从检查点恢复训练 (`--resume_path`) 以及加载模型进行纯测试 (`--test_model_path`)。

### 顶层模型 (`spikenet_x/model.py`)

* **Class**: `SpikeTDANet`
* **目的**: 作为顶层容器，堆叠多个 `SpikeTDANetLayer`，并处理模型的输入投影和最终输出读出。
* **`__init__` 输入**:
  * `d_in`: 原始节点特征维度。
  * `d`: 模型内部的工作维度。
  * `layers`: `SpikeTDANetLayer` 的堆叠层数。
  * `heads`: 注意力头的数量。
  * `W`: 注意力机制的时间窗口大小。
  * `out_dim`: 最终分类任务的类别数。
  * `readout`: 最终节点表示的生成方式 (`'mean'` 或 `'last'`)。
* **`forward` 逻辑**:
  1. **输入投影**: 使用一个线性层 `input_proj` 将原始特征 `H` `[T, N, d_in]` 映射到模型工作维度 `d`。
  2. **逐层处理**: 循环遍历 `self.layers`，将上一层的输出 `(features, spikes)` 作为下一层的输入。
  3. **读出 (Readout)**:
     * 如果 `readout == 'mean'`，则对最后一层输出的特征在时间维度上取平均，得到每个节点的最终表示 `z` `[N, d]`。
     * 如果 `readout == 'last'`，则取最后一个时间步的特征作为最终表示。
  4. **分类头**: 将节点表示 `z` 通过一个线性层 `head` 得到最终的分类 logits `[N, out_dim]`。
* **输出**: 一个包含 `repr` (节点表示), `Y_last` (最后一层特征), `S_list` (每层脉冲) 和 `logits` 的字典。

### 核心编排层 (`spikenet_x/spiketdanet_layer.py`)

* **Class**: `SpikeTDANetLayer`
* **目的**: 实现[数据处理流程](#数据处理流程-单层)中描述的单层逻辑，是模型的核心编排单元。
* **`__init__` 输入**:
  * `channels`: 层的特征维度 `d`。
  * `heads`: 注意力头数量。
  * `W`: 时间窗口大小。
  * `delay_kernel`: `DelayLine` 的卷积核大小。
  * `**kwargs`: 其他传递给子模块的参数，如LIF神经元参数 (`lif_tau_theta`, `lif_gamma` 等)。
* **`forward` 逻辑**: 严格按照[数据处理流程](#数据处理流程-单层)所述，依次调用 `spatial_gnn`, `delay_line`, `aggregator`, `lif_cell`, 和 `ffn`，并正确处理它们之间的残差连接和层归一化。
* **输出**: `(output_features, new_spikes)` 元组，传递给下一层。

### 空间GNN预处理器 (`spikenet_x/new_modules/spatial_gnn_wrapper.py`)

* **Class**: `SpatialGNNWrapper`
* **目的**: 在时序图的每个时间步上高效地应用标准GNN卷积（如 `SAGEConv`）。
* **`__init__` 输入**:
  * `in_channels`, `out_channels`: 输入输出特征维度。
* **`forward` 输入**:
  * `x`: 节点特征 `[T, N, d_in]`。
  * `edge_index`: 图结构 `[2, E]`。
* **处理逻辑**:
  1. 将 `x` 重塑为 `[T*N, d_in]`，将时间维和节点维合并，形成一个大的批次。
  2. 通过对 `edge_index` 添加偏移量 `t * N`，将其扩展为对应 `T` 个图快照的批处理边索引。
  3. 调用 `self.conv` 对重塑后的 `x` 和扩展后的 `edge_index` 进行一次高效的批处理GNN计算。
  4. 将结果恢复为 `[T, N, d_out]` 并返回。
* **输出**: 空间聚合后的节点特征 `[T, N, d_out]`。

### 可学习延迟通路 (`spikenet_x/new_modules/delay_line.py`)

* **Class**: `DelayLine`
* **目的**: 使用因果深度可分离1D卷积，低成本地建模多种时间延迟。
* **`__init__` 输入**:
  * `channels`: 特征维度 `d`。
  * `kernel_size`: 卷积核大小，决定了能建模的最大延迟。
* **`forward` 输入**:
  * `x`: 节点特征 `[T, N, d]`。
* **处理逻辑**:
  1. 将 `x` 的维度重排为 `[N, d, T]`，以适应1D卷积对时间维的操作。
  2. 应用 `padding = kernel_size - 1` 的左侧填充，确保卷积的**因果性**（`t` 时刻的输出只依赖于 `<=t` 的输入）。
  3. 执行深度卷积 (`groups=channels`) 和逐点卷积 (`kernel_size=1`)。
  4. 对输出进行切片，移除因填充而多出的部分，保持时间长度为 `T`。
  5. 将结果维度恢复为 `[T, N, d]`。
* **输出**: 经过延迟建模后的特征 `[T, N, d]`。

### 脉冲时序注意力聚合器 (`spikenet_x/new_modules/sta_gnn_agg.py`)

* **Class**: `STAGNNAggregator`
* **目的**: 模型的计算核心。执行稀疏化的、带脉冲门控的因果时空注意力。
* **`__init__` 输入**:
  * `d_in`, `d`: 输入输出维度。
  * `heads`: 注意力头数。
  * `W`: 时间窗口大小。
  * `attn_drop`: Attention权重的Dropout概率。
* **`forward` 输入**:
  * `H_tilde`: 经过 `DelayLine`处理的特征 `[T, N, d_in]`。
  * `S`: 脉冲门控信号 `[T, N]`。
  * `edge_index`: 图结构 `[2, E]`。
  * `time_idx`: 时间索引 `[T]`。
* **处理逻辑 (基于稀疏实现)**:
  1. **预计算**: 对所有时间步的 `H_tilde` 计算出 Q, K, V 投影。
  2. **时间步循环**: 对每个目标时间步 `t`：
     * **Pass 1 (计算最大值)**: 在时间窗口 `t' in [t-W, t]` 内，对每条边 `j->i` (`src->dst`) 计算其注意力分数 `score(i, j, t, t')`。此分数包含Q-K点积、相对时间偏置和源节点 `j` 在 `t'` 时刻的脉冲门控 `log(S[t', j])`。然后，对每个目标节点 `i`，计算其在所有时间窗口内所有邻居中收到的最大分数 `max_score(i)`。
     * **Pass 2 (计算聚合)**: 再次遍历时间窗口和边，计算 `exp(score - max_score(i))`。使用 `scatter_add` 操作稳定地计算每个目标节点 `i` 的归一化分母和加权后的V值分子。
     * 最终得到聚合后的消息 `M[t, i, :]`。
  3. 将所有时间步的消息 `M[t, :, :]` 组合起来。
* **输出**: 聚合后的消息 `aggregated_message` `[T, N, d]`。

### LIF神经元 (`spikenet_x/lif_cell.py`)

* **Class**: `LIFCell`
* **目的**: 纯粹的Leaky Integrate-and-Fire (LIF) 神经元动力学模型。
* **`__init__` 输入**:
  * `lif_tau_theta`: 膜电位发放阈值 `V_th`。
  * `lif_gamma`: 脉冲后膜电位的重置衰减因子。
  * `lif_beta`: 膜电位泄漏因子。
* **`forward` 输入**:
  * `I_in`: 输入电流 `[T, N]`。
* **处理逻辑**:
  1. 初始化膜电位 `v` 和脉冲 `s` 为零。
  2. 按时间步 `t` 循环：
     a. **泄漏 (Leak)**: `v = v * beta`。
     b. **积分 (Integrate)**: `v = v + I_in[t]`。
     c. **重置 (Reset)**: `v = v - s * gamma` (减去上一步脉冲的影响)。
     d. **发放 (Fire)**: `s = (v > V_th).float()`，如果电位超过阈值，则发放脉冲。
     e. **硬重置**: `v = v * (1.0 - s)`，发放脉冲后电位归零。
  3. 记录并返回所有时间步的脉冲和膜电位历史。
* **输出**: `(spikes, v_mem, spike_history)` 元组，均为 `[T, N]` 张量。

### 辅助模块

* **`spikenet_x/rel_time.py` -> `RelativeTimeEncoding`**:

  * **作用**: 生成相对时间编码。对于一个时间差 `Δt`，它能产生一个唯一的编码向量，编码方式结合了指数衰减和正/余弦函数。同时，它维护一个可学习的相对偏置向量 `b[Δt]`。
  * **输出**: `(pe_table, rel_bias)`，一个编码查找表和一个偏置查找表，供 `STAGNNAggregator` 使用。
* **`spikenet/dataset.py`**:

  * **作用**: 负责加载和预处理DBLP, Tmall, Patent等时序图数据集。它将原始数据文件解析为一系列图快照，每个快照包含节点特征 `x` 和边 `edge_index`。
* **`spikenet/utils.py`**:

  * **作用**: 提供工具函数，如邻居采样器 (`Sampler`)、随机游走采样器 (`RandomWalkSampler`) 和自环处理函数。

## 6. 如何运行

1. **环境配置**: 确保已安装 `torch`, `torch_geometric`, `numpy`, `scikit-learn`, `texttable` 等依赖库。如果 `STAGNNAggregator` 报错，请确保 `torch` 版本 >= 1.12 或安装 `torch_scatter`。
2. **数据准备**:

   * 将数据集（如DBLP）放置在 `data/` 目录下。
   * 如果节点特征文件 (`.npy`) 不存在，运行特征生成脚本：
     ```bash
     python generate_feature.py --dataset DBLP
     ```
3. **模型训练**:

   * 运行 `main.py` 脚本，并指定模型为 `spiketdanet`。
   * **示例命令**:
     ```bash
     python main.py --model spiketdanet --dataset DBLP --epochs 100 --lr 5e-4 --batch_size 512 --hids 64 --heads 4 --W 16
     ```
   * **关键参数**:
     * `--model spiketdanet`: **必须指定**，以使用新模型。
     * `--dataset`: 选择数据集 (DBLP, Tmall, Patent)。
     * `--hids`: 模型内部特征维度 `d` (只取第一个值)。
     * `--heads`: 注意力头数。
     * `--W`: 时间窗口大小。
     * `--checkpoint_dir`: 保存最佳模型的目录。
4. **恢复训练**:

   * 如果训练中断，可以使用 `--resume_path` 参数从上次保存的检查点继续。
     ```bash
     python main.py ... --resume_path checkpoints/best_model_DBLP.pth
     ```
5. **模型测试**:

   * 使用 `--test_model_path` 加载一个已训练好的模型并进行评估。
     ```bash
     python main.py --model spiketdanet --dataset DBLP --test_model_path checkpoints/best_model_DBLP.pth
     ```

> provided by [EasyChat](http://iSq7n9s0OQ.site.llm99.com/)
>


# Spike-TDANet: 面向动态图的脉冲时序延迟注意力网络

**Spike-TDANet** (Spiking Temporal Delay Attention Network) 是一个专为动态图（时序图）设计的深度学习模型。它创新性地结合了图神经网络（GNN）强大的空间特征提取能力和脉冲神经网络（SNN）在处理时序信息上的高效性与生物可解释性。模型通过独特的 **时序延迟注意力机制 (Temporal Delay Attention)** 和 **脉冲门控聚合 (Spike-Gated Aggregation)**，能够高效地捕捉节点在时空维度上的复杂依赖关系，适用于动态图上的节点分类等任务。

![模型结构图](https://user-images.githubusercontent.com/assets/your-image-placeholder.png)

## 目录

- [项目亮点](#项目亮点)
- [模型架构](#模型架构)
  - [整体流程](#整体流程)
  - [核心模块详解 (SpikeTDANetLayer)](#核心模块详解-spiketdanetlayer)
- [代码模块详解](#代码模块详解)
  - [`SpikeTDANet` (模型主类)](#spiketdanet-模型主类)
  - [`SpikeTDANetLayer` (核心层)](#spiketdanetlayer-核心层)
  - [`SpatialGNNWrapper` (空间特征提取)](#spatialgnnwrapper-空间特征提取)
  - [`DelayLine` (时序延迟建模)](#delayline-时序延迟建模)
  - [`STAGNNAggregator_Optimized` (时空注意力聚合)](#stagnnaggregator_optimized-时空注意力聚合)
  - [`SurrogateLIFCell` (脉冲发放单元)](#surrogatelifcell-脉冲发放单元)
- [如何使用](#如何使用)
  - [环境配置](#环境配置)
  - [1. 数据准备 (特征生成)](#1-数据准备-特征生成)
  - [2. 模型训练](#2-模型训练)
  - [3. 模型测试](#3-模型测试)
  - [4. 超参数搜索 (Optuna)](#4-超参数搜索-optuna)
- [命令行参数详解](#命令行参数详解)
- [项目文件结构](#项目文件结构)
- [许可](#许可)

## 项目亮点

1. **混合式架构**: 结合 GNN 强大的空间建模能力和 SNN 在时序处理上的优势，实现了对动态图时空信息的深度融合。
2. **时序延迟注意力**: 通过 `DelayLine` 模块和 `STAGNNAggregator` 中的相对时间编码，模型不仅关注“是否连接”，还关注“何时连接”，并对不同时间延迟的邻居信息进行加权聚合。
3. **脉冲门控机制**: 上一层神经元的脉冲发放（Spike）被用作下一层注意力聚合的门控信号。这种机制模拟了生物神经元的信息筛选过程，使得信息流动更加动态和高效。
4. **高维脉冲神经元**: 采用 `SurrogateLIFCell` 直接处理高维特征向量，避免了将信息压缩至标量电流时的瓶颈，保留了更丰富的特征信息用于脉冲发放决策。
5. **高效的稀疏实现**: 核心的 `STAGNNAggregator` 模块基于 `edge_index` 进行稀疏计算，避免了构造稠密的 `(N, N)` 邻接矩阵，使其能够扩展到大规模图上。

## 模型架构

### 整体流程

Spike-TDANet 的核心思想是逐层处理动态图数据，每一层都完成一次完整的“空间聚合 -> 时间建模 -> 时空融合 -> 脉冲发放”过程。

1. **输入**: 模型接收一个时序图数据，包括：

   * **节点特征序列 `H`**: 形状为 `[T, N, d_in]`，`T` 是时间步数，`N` 是节点数，`d_in` 是初始特征维度。
   * **全局边索引 `edge_index`**: 形状为 `[2, E]`，包含了所有时间步中出现过的边。
   * **时间戳索引 `time_idx`**: 形状为 `[T]`，通常是 `torch.arange(T)`。
2. **输入投影**: 初始节点特征 `H` 首先通过一个线性层 (`input_proj`) 投影到模型的隐藏维度 `d`。
3. **堆叠 `SpikeTDANetLayer`**: 经过投影的特征被送入一个由多个 `SpikeTDANetLayer` 堆叠而成的网络。

   * 每一层接收上一层的 **输出特征 `features`** 和 **脉冲信号 `spikes`** 作为输入。
   * 第一层没有 `spikes` 输入，因此默认所有节点都在发放脉冲（即不进行门控）。
   * 每一层都会输出新的特征和脉冲，传递给下一层。
4. **读出与分类**:

   * 最后一层输出的节点特征序列 `features` 经过一个读出（Readout）操作（如对时间维度取平均 `mean` 或取最后时刻 `last`），得到每个节点的最终表示 `z`。
   * 最终表示 `z` 被送入一个分类头 (`head`)，得到预测的 `logits`。

### 核心模块详解 (SpikeTDANetLayer)

`SpikeTDANetLayer` 是模型的核心，其内部处理流程如下：


1. **空间GNN预处理 (`SpatialGNNWrapper`)**:

   * **功能**: 对每个时间步 `t` 的节点特征 `x[t]` 独立地进行一次图卷积（如 GraphSAGE），聚合来自直接邻居的瞬时空间信息。
   * **目的**: 捕捉图在当前时刻的静态空间结构。
   * **实现**: 通过巧妙地重塑张量和扩展 `edge_index`，可以在整个时间序列上进行一次批处理的图卷积，非常高效。
2. **时序延迟建模 (`DelayLine`)**:

   * **功能**: 对经过空间处理后的特征序列，沿着时间轴对每个节点独立地进行一次因果一维卷积。
   * **目的**: 建模节点自身特征在短期内的演化模式，即“记忆”效应。
   * **实现**: 采用深度可分离卷积，计算成本低廉，同时能有效捕捉多种时间延迟。
3. **时空注意力聚合 (`STAGNNAggregator_Optimized`)**:

   * **功能**: 这是模型的核心。对于每个目标节点 `i` 在 `t` 时刻，它会关注其邻居 `j` 在过去时间窗口 `[t-W, t]` 内的特征。
   * **注意力权重计算**: 权重不仅取决于 `i` 和 `j` 的特征相似度，还受到以下因素调制：
     * **相对时间 `t-t'`**: 通过 `RelativeTimeEncoding` 模块，为不同的时间差赋予独特的编码和可学习偏置。
     * **上一层脉冲 `S[t', j]`**: 邻居 `j` 在 `t'` 时刻的脉冲强度会作为门控信号，调节其信息的重要性。脉冲越强，权重越高。
   * **目的**: 动态地、有选择地从时空邻域中聚合最相关的信息。
4. **脉冲发放 (`SurrogateLIFCell`)**:

   * **功能**: 将聚合后的高维时空消息 `aggregated_message` 视为输入电流，注入一个 Leaky Integrate-and-Fire (LIF) 神经元模型。
   * **LIF 动态**: 神经元的膜电位根据输入电流进行累积，当超过阈值时发放一个脉冲，并重置电位。
   * **替代梯度**: 由于脉冲发放是不可导的，训练时采用 Sigmoid 或 Triangle 等平滑函数作为其替代梯度，以实现端到端的反向传播。
   * **输出**:
     * **高维脉冲 `spikes_hd` (`[T, N, C]`)**: 用于后续的特征门控。
     * **标量脉冲 `new_spikes` (`[T, N]`)**: 通过对高维脉冲在特征维度上取平均得到，传递给下一层作为门控信号。
5. **输出处理 (FFN & 残差连接)**:

   * 聚合后的消息 `aggregated_message` 会被 `spikes_hd` 门控，然后通过一个前馈网络 (FFN/MLP) 进行非线性变换。
   * 最后，通过残差连接和层归一化 (LayerNorm) 产生该层的最终输出特征。

## 代码模块详解

下面是核心代码模块的功能、参数和代码位置的详细说明。

### `SpikeTDANet` (模型主类)

- **文件**: `spikenet_x/model.py`
- **功能**: 作为模型的容器，负责搭建和管理多个 `SpikeTDANetLayer`。
- **`__init__` 参数**:
  - `d_in` (int): 输入特征维度。
  - `d` (int): 模型隐藏层维度。
  - `layers` (int): `SpikeTDANetLayer` 的层数。
  - `heads` (int): 注意力机制的头数。
  - `W` (int): 注意力机制的时间窗口大小。
  - `out_dim` (int): 输出维度（类别数）。
  - `readout` (str): 读出方式，`'mean'` 或 `'last'`。
  - `lif_tau` (float): LIF神经元的膜电位衰减因子。
  - `lif_v_threshold` (float): LIF神经元的脉冲阈值。
  - `lif_alpha` (float): 替代梯度的平滑度因子。
  - `lif_surrogate` (str): 替代梯度函数类型，如 `'sigmoid'` 或 `'triangle'`。

### `SpikeTDANetLayer` (核心层)

- **文件**: `spikenet_x/spiketdanet_layer.py`
- **功能**: 实现单层的时空信息处理流程。
- **`__init__` 参数**:
  - `channels` (int): 层的输入/输出维度（等于模型的 `d`）。
  - `heads` (int): 注意力头数。
  - `W` (int): 时间窗口大小。
  - `delay_kernel` (int): `DelayLine` 的卷积核大小。
  - `lif_*` 参数: 传递给 `SurrogateLIFCell`。
- **`forward` 方法**:
  - **输入**: `x` (特征), `spikes` (上一层脉冲), `edge_index`, `time_idx`。
  - **输出**: `layer_output_features` (本层输出特征), `new_spikes_for_next_layer` (本层输出脉冲)。

### `SpatialGNNWrapper` (空间特征提取)

- **文件**: `spikenet_x/new_modules/spatial_gnn_wrapper.py`
- **功能**: 在每个时间步上应用 SAGEConv。
- **`__init__` 参数**:
  - `in_channels`, `out_channels` (int): 输入输出维度。
  - `aggr` (str): SAGEConv 的聚合方式，如 `'mean'`。

### `DelayLine` (时序延迟建模)

- **文件**: `spikenet_x/new_modules/delay_line.py`
- **功能**: 使用因果深度可分离1D卷积建模时间延迟。
- **`__init__` 参数**:
  - `channels` (int): 特征维度。
  - `kernel_size` (int): 卷积核大小，决定了建模的延迟范围。

### `STAGNNAggregator_Optimized` (时空注意力聚合)

- **文件**: `spikenet_x/new_modules/sta_gnn_agg_optimized.py`
- **功能**: 高效的、基于稀疏边的、脉冲门控的时空注意力聚合。
- **`__init__` 参数**:
  - `d_in`, `d` (int): 输入输出维度。
  - `heads` (int): 注意力头数。
  - `W` (int): 时间窗口。
  - `attn_drop` (float): 注意力权重的 Dropout 概率。
  - `temp` (float): Softmax 的温度系数。
  - `pe_*` 参数: 传递给 `RelativeTimeEncoding`。

### `SurrogateLIFCell` (脉冲发放单元)

- **文件**: `spikenet_x/surrogate_lif_cell.py`
- **功能**: 支持高维特征输入和替代梯度训练的 LIF 神经元。
- **`__init__` 参数**:
  - `channels` (int): 输入特征维度。
  - `v_threshold` (float): 膜电位阈值。
  - `tau` (float): 膜电位时间常数（衰减因子）。
  - `alpha` (float): 替代梯度平滑度。
  - `surrogate` (str): 替代梯度函数名。

## 如何使用

### 环境配置

建议使用 `conda` 创建虚拟环境，并安装必要的依赖。

```bash
# 1. 创建并激活 conda 环境
conda create -n csegnn python=3.8
conda activate csegnn

# 2. 安装 PyTorch (请根据你的 CUDA 版本从官网选择合适的命令)
# 例如 CUDA 11.7
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# 3. 安装 PyTorch Geometric (PyG)
pip install torch_geometric
# 可选：安装更快的 GNN 算子
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# 4. 安装其他依赖
pip install numpy scikit-learn tqdm texttable optuna
```

### 1. 数据准备 (特征生成)

本模型使用 DeepWalk 预先为动态图的每个快照生成节点特征。如果你的数据集没有提供时序特征，请运行此脚本。

```bash
python generate_feature.py --dataset DBLP
```

* 将 `--dataset` 替换为 `Tmall` 或 `Patent` 以处理其他数据集。
* 这会在相应的数据集目录（如 `./data/DBLP/`）下生成一个 `DBLP.npy` 文件，其中包含了 `[T, N, F]` 形状的节点特征。

### 2. 模型训练

使用 `main.py` 脚本进行模型训练和评估。

```bash
python main.py \
    --model spiketdanet \
    --dataset DBLP \
    --hids 128 \
    --heads 4 \
    --W 32 \
    --layers 2 \
    --lr 0.001 \
    --wd 0.0001 \
    --epochs 100 \
    --batch_size 256 \
    --checkpoint_dir ./checkpoints
```

* 训练过程中，在验证集上表现最好的模型将被保存到 `--checkpoint_dir` 指定的目录中，文件名为 `best_model_{数据集名称}.pth`。
* 你可以通过 `--resume_path` 参数从一个检查点继续训练。

### 3. 模型测试

如果你只想测试一个已经训练好的模型，可以使用 `--test_model_path` 参数。

```bash
python main.py \
    --model spiketdanet \
    --dataset DBLP \
    --test_model_path ./checkpoints/best_model_DBLP.pth
```

脚本将加载模型，在测试集上运行一次评估，并打印 F1 分数。

### 4. 超参数搜索 (Optuna)

我们提供了 `main_optuna.py` 脚本来自动搜索最佳超参数组合。

```bash
python main_optuna.py
```

* 该脚本会针对 `DBLP` 数据集（可在脚本内修改 `DATASET_NAME` 常量）进行 200 次试验。
* 搜索过程会被持久化到一个 SQLite 数据库文件（如 `spiketdanet_opt_DBLP.db`）中，可以随时中断和恢复。
* 搜索结束后，会打印最佳参数组合，并生成 `optuna_report_*.html` 可视化报告，帮助你分析参数的重要性。

## 命令行参数详解

以下是 `main.py` 中与 Spike-TDANet 相关的主要参数：

* `--model`: (str) 要使用的模型，选择 `spiketdanet`。
* `--dataset`: (str) 数据集名称，如 `DBLP`, `Tmall`, `Patent`。
* `--hids`: (int) 模型的隐藏维度 `d`。
* `--layers`: (int) `SpikeTDANetLayer` 的层数。
* `--heads`: (int) 注意力头数。
* `--W`: (int) 时空注意力的回顾窗口大小。
* `--readout`: (str) 最终节点表示的读出方式 (`'mean'` 或 `'last'`)。
* `--lr`, `--wd`, `--epochs`, `--batch_size`: 标准的训练超参数。
* `--surrogate`: (str) LIF 神经元使用的替代梯度函数，如 `'sigmoid'`。
* `--checkpoint_dir`: (str) 保存最佳模型的目录。
* `--resume_path`: (str) 用于继续训练的检查点路径。
* `--test_model_path`: (str) 仅用于测试的模型路径。

## 项目文件结构

```
CSGNN/
│
├── main.py                   # Spike-TDANet 和基线模型的训练/评估主脚本
├── main_optuna.py            # 使用 Optuna 进行超参数优化的脚本
├── generate_feature.py       # 使用 DeepWalk 生成节点特征
├── spikenet_x/               # Spike-TDANet 模型核心代码
│   ├── model.py              # SpikeTDANet 整体模型定义
│   ├── spiketdanet_layer.py  # SpikeTDANetLayer 核心层定义
│   ├── surrogate_lif_cell.py # 带替代梯度的 LIF 神经元
│   ├── new_modules/          # SpikeTDANetLayer 的构建模块
│   │   ├── spatial_gnn_wrapper.py
│   │   ├── delay_line.py
│   │   ├── sta_gnn_agg_optimized.py # 优化的时空注意力聚合器
│   │   └── ...
│   └── rel_time.py           # 相对时间编码
│
├── spikenet/                 # 基线模型 (CSGNN) 的代码
│   ├── dataset.py            # 数据集加载和预处理
│   ├── layers.py             # 基线模型的层
│   └── ...
│
└── data/                     # 数据集存放目录
```

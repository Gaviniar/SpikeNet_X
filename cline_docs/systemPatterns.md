# 系统模式 (System Patterns)

## 系统如何构建？
SpikeNet 项目是一个基于 PyTorch 实现的动态图表示学习框架，核心在于利用脉冲神经网络 (SNNs) 处理时间图数据。其主要组件和构建方式如下：

1.  **数据层 (`spikenet/dataset.py`)**：
    *   提供 `Dataset` 基类，以及 DBLP、Tmall、Patent 等具体数据集的实现。
    *   负责从文件中读取节点特征（`.npy`）、边（`.txt` 或 `.json`）和标签（`node2label.txt` 或 `.json`）。
    *   支持对节点特征进行标准化。
    *   将边列表转换为稀疏邻接矩阵 (`scipy.sparse.csr_matrix`)。
    *   实现节点和边的时间切片与划分，以模拟图的动态演化。
    *   数据集迭代器允许按时间步访问图快照。

2.  **核心模型组件 (`spikenet/neuron.py`, `spikenet/layers.py`)**：
    *   **神经元模型 (`spikenet/neuron.py`)**：定义了基本的脉冲神经元（如 IF, LIF, PLIF）。这些神经元模型负责电压积分、发放脉冲和重置。
    *   **替代梯度 (`spikenet/neuron.py`)**：由于 SNN 的脉冲函数不可导，使用了多种替代梯度技术（如 SuperSpike, MultiGaussSpike, TriangleSpike, ArctanSpike, SigmoidSpike）来实现反向传播训练。
    *   **图聚合器 (`spikenet/layers.py`)**：包含了 `SAGEAggregator`，表明网络层可能采用了 GraphSAGE 风格的邻居特征聚合机制。它将中心节点特征与聚合后的邻居特征进行组合。

3.  **图采样器 (`spikenet/utils.py`, `spikenet/sample_neighber.cpp`)**：
    *   `spikenet/utils.py` 中定义了 `Sampler` 和 `RandomWalkSampler` 类，用于从邻接矩阵中采样邻居。
    *   `Sampler` 类利用了外部 C++ 实现 `sample_neighber_cpu` 进行高效的邻居采样，这可能是为了性能优化。
    *   `RandomWalkSampler` 在可选依赖 `torch_cluster` 存在时提供随机游走采样功能。

4.  **特征生成 (`generate_feature.py`, `spikenet/deepwalk.py`)**：
    *   `generate_feature.py` 脚本用于为不带原始特征的数据集生成节点特征，通过无监督的 DeepWalk 方法实现，其核心逻辑可能在 `spikenet/deepwalk.py` 中。

5.  **训练入口 (`main.py`, `main_static.py`)**：
    *   `main.py` 是用于动态图训练的主脚本，配置数据集、模型参数和训练过程。
    *   `main_static.py` 是用于静态图训练的脚本，可能适配了不同的数据集和训练流程。

## 关键技术决策
*   **SNNs 用于动态图**：核心创新是将 SNNs 应用于动态图表示学习，以解决传统 RNNs 在大规模图上的计算和内存效率问题。
*   **替代梯度**：采用替代梯度方法来训练 SNNs，使其能够通过反向传播进行优化。
*   **GraphSAGE 风格聚合**：使用聚合器从邻居节点收集信息，这是图神经网络中的常见模式。
*   **C++ 优化采样**：通过 `sample_neighber.cpp` 提供的 C++ 实现进行邻居采样，以提高性能和处理大规模图的能力。
*   **模块化设计**：将神经元模型、网络层、数据处理和采样器等功能分别封装在不同的模块中，提高了代码的可维护性和可扩展性。
*   **数据集支持**：设计了通用的 `Dataset` 接口，并为多个真实世界大型时间图数据集提供了具体实现。

## 架构模式
*   **时间序列图处理**：通过迭代时间步来处理图快照，捕获图的动态演化。
*   **消息传递范式**：聚合器（如 `SAGEAggregator`）遵循图神经网络的消息传递范式，其中节点通过聚合邻居信息来更新其表示。
*   **分离的数据加载与模型逻辑**：`dataset.py` 负责数据管理，而 `neuron.py` 和 `layers.py` 负责模型核心逻辑，实现了关注点分离。
*   **参数化神经元行为**：神经元模型（如 LIF）通过可配置的参数（如 `tau`, `v_threshold`, `alpha`）和可选择的替代梯度类型，提供了灵活性。
*   **命令行参数配置**：`main.py` 和 `main_static.py` 通过命令行参数 (`argparse`) 配置训练过程，方便实验和调优。

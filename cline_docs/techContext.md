# 技术背景 (Tech Context)

## 使用的技术
*   **Python**：主要的编程语言。
*   **PyTorch**：核心深度学习框架，用于构建和训练神经网络。
*   **NumPy**：用于数值计算和数组操作。
*   **SciPy**：用于科学计算，特别是稀疏矩阵操作 (`scipy.sparse`)。
*   **Scikit-learn**：用于数据预处理（如 `LabelEncoder`）和模型评估。
*   **tqdm**：用于显示进度条。
*   **texttable**：用于命令行参数的表格化输出。
*   **Numba**：一个 JIT 编译器，可能用于加速某些 Python 代码。
*   **C++**：用于高性能的邻居采样模块 (`sample_neighber.cpp`)，通过 `setup.py` 进行编译和集成。
*   **torch_cluster (可选)**：如果安装，用于更高级的图采样操作，如随机游走。

## 开发设置
*   **环境**：项目支持在 PyTorch 环境下运行。
*   **依赖**：`requirements` 部分列出了具体的包及其版本，包括 `tqdm`, `scipy`, `texttable`, `torch`, `numpy`, `numba`, `scikit_learn` 和可选的 `torch_cluster`。
*   **邻居采样器构建**：需要通过运行 `python setup.py install` 来编译和安装 C++ 实现的邻居采样器。
*   **数据准备**：数据集需要下载并放置在 `data/` 目录下。对于没有原始节点特征的数据集，可以通过 `generate_feature.py` 脚本使用 DeepWalk 生成特征。

## 技术约束
*   **大规模图处理**：设计目标是处理包含数百万节点和数千万边的大型时间图，对计算和内存效率有较高要求。
*   **SNN 训练挑战**：脉冲函数不可导，需要依赖替代梯度方法进行训练。
*   **数据格式**：需要适配不同数据集的特定文件格式（`.txt`, `.json`, `.npy`）。
*   **PyTorch 版本兼容性**：代码应兼容 PyTorch 1.6-1.12 版本。
*   **C++ 依赖**：邻居采样器依赖 C++ 编译，可能需要相应的编译环境。
*   **`torch_cluster` 依赖 (可选)**：随机游走采样功能依赖于 `torch_cluster` 库，如果未安装则无法使用该功能。

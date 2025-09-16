# main_optuna.py
import argparse
import time
import os

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm


import optuna
from optuna.trial import TrialState

from spikenet import dataset
# 更改导入，直接使用新模型名称
from spikenet_x.model import SpikeTDANet
from texttable import Texttable
import numpy as np
from torch_geometric.utils import subgraph
from typing import Tuple

# --- [Optuna] 将固定的参数定义为全局常量 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = "DBLP"  # 您可以在这里修改数据集, e.g., "Tmall", "Patent"
DATAPATH = './data'
EPOCHS = 50  # 建议在调参时适当减少epoch数量以加快速度
BATCH_SIZE = 1024
TRAIN_SIZE = 0.4
VAL_SIZE = 0.05
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE
SPLIT_SEED = 42
SEED = 2022

def sample_subgraph(nodes: torch.Tensor, edge_index_full: torch.Tensor, num_neighbors: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    高效的1-hop子图采样函数。
    - 强制限制每个节点的邻居数量，以控制计算复杂度。
    - 使用 torch_geometric.utils.subgraph 进行快速的边提取和节点重标签。
    """
    row, col = edge_index_full
    device = row.device
    nodes = nodes.to(device)

    # 1) 找到所有与种子节点相连的边和邻居
    node_mask = torch.isin(row, nodes)
    edge_index_subset = edge_index_full[:, node_mask]
    neighbors_all = torch.unique(edge_index_subset[1])
    
    # 2) 如果需要，对邻居进行采样
    if num_neighbors > 0 and neighbors_all.numel() > 0:
        target_num_neighbors = nodes.numel() * num_neighbors
        if neighbors_all.numel() > target_num_neighbors:
            perm = torch.randperm(neighbors_all.numel(), device=device)[:target_num_neighbors]
            neighbors = neighbors_all[perm]
        else:
            neighbors = neighbors_all
    else:
        neighbors = neighbors_all

    # 3) 构建子图节点集并获取重标签后的边
    subgraph_nodes = torch.unique(torch.cat([nodes, neighbors]))
    
    subgraph_edge_index, _ = subgraph(
        subset=subgraph_nodes,
        edge_index=edge_index_full,
        relabel_nodes=True,
        num_nodes=None
    )

    # 4) 找到原始种子节点在子图中的新索引
    subgraph_nodes_sorted, _ = torch.sort(subgraph_nodes)
    nodes_local_index = torch.searchsorted(subgraph_nodes_sorted, nodes)
    
    return subgraph_nodes_sorted, subgraph_edge_index, nodes_local_index


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# --- [Optuna] 数据加载和预处理（只执行一次） ---
print("--- Loading and Preparing Data (Once) ---")
set_seed(SEED)

if DATASET_NAME.lower() == "dblp":
    data = dataset.DBLP(root=DATAPATH)
elif DATASET_NAME.lower() == "tmall":
    data = dataset.Tmall(root=DATAPATH)
elif DATASET_NAME.lower() == "patent":
    data = dataset.Patent(root=DATAPATH)
else:
    raise ValueError(f"{DATASET_NAME} is invalid.")

data.split_nodes(train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=SPLIT_SEED)

y = data.y.to(DEVICE)
train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(data.val_nodes.tolist(), pin_memory=False, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=BATCH_SIZE, shuffle=False)

# 准备全图数据
T = len(data)
N = data.num_nodes
d_in = data.num_features
edge_list = [snapshot.edge_index for snapshot in data]
edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(DEVICE)
H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(DEVICE)
time_idx_full = torch.arange(T, device=DEVICE)
print("--- Data Ready ---")


# --- [Optuna] 定义 Objective 函数 ---
def objective(trial: optuna.Trial) -> float:
    """
    Optuna的objective函数，执行一次完整的训练和验证流程。
    """
    # 1. 定义超参数搜索空间
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
    d = trial.suggest_categorical("d", [32, 64, 128])
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    W = trial.suggest_int("W", 8, 48, step=8)
    surrogate = trial.suggest_categorical("surrogate", ["sigmoid", "triangle"])
    readout = trial.suggest_categorical("readout", ["mean", "last"])
    lif_tau = trial.suggest_float("lif_tau", 0.8, 0.99)
    
    # 确保 d 可以被 heads 整除
    if d % heads != 0:
        # 如果d不能被heads整除，则此组合无效，提前终止
        raise optuna.exceptions.TrialPruned()

    # 2. 根据建议的超参数构建模型和优化器
    model = SpikeTDANet(
        d_in=d_in,
        d=d,
        layers=2,  # 固定层数为2，也可以加入搜索空间
        heads=heads,
        W=W,
        out_dim=data.num_classes,
        readout=readout,
        lif_tau=lif_tau,
        lif_v_threshold=1.0,  # 固定
        lif_alpha=1.0,        # 固定
        lif_surrogate=surrogate
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    # 3. 定义嵌套的训练和测试函数（它们依赖于每个trial创建的model）
    def train_model_step(epoch):
        model.train()
        total_loss = 0
        num_neighbors_to_sample = 25
        for nodes in train_loader:
            nodes = nodes.to(DEVICE)
            subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
            H0_subgraph = H0_full[:, subgraph_nodes, :]
            
            optimizer.zero_grad()
            output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
            subgraph_logits = output['logits']
            loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])
            
            spike_rate = output['S_list'].float().mean()
            if epoch > 10:
                loss = loss + 2e-5 * (spike_rate - 0.15).abs()
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test_model_step(loader):
        model.eval()
        logits_list = []
        labels_list = []
        num_neighbors_to_sample = 25
        for nodes in loader:
            nodes = nodes.to(DEVICE)
            subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
            H0_subgraph = H0_full[:, subgraph_nodes, :]
            
            output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
            subgraph_logits = output['logits']
            
            logits_list.append(subgraph_logits[nodes_local_index].cpu())
            labels_list.append(y[nodes].cpu())
            
        logits = torch.cat(logits_list, dim=0).argmax(1)
        labels = torch.cat(labels_list, dim=0)
        
        micro = metrics.f1_score(labels, logits, average='micro', zero_division=0)
        macro = metrics.f1_score(labels, logits, average='macro', zero_division=0)
        return macro, micro

    # 4. 训练循环
    best_val_micro = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_model_step(epoch)
        val_macro, val_micro = test_model_step(val_loader)

        # 更新最佳验证分数
        if val_micro > best_val_micro:
            best_val_micro = val_micro

        # --- [Optuna] 向Optuna报告中间结果 ---
        trial.report(val_micro, epoch)

        # --- [Optuna] 处理剪枝请求 ---
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 5. 返回最终要优化的指标
    return best_val_micro


if __name__ == "__main__":
    # --- [Optuna] 创建并运行 Study ---
    
    # 1. 创建一个Study对象，我们想要最大化目标函数（验证集Micro-F1）
    # MedianPruner是一种简单有效的剪枝策略
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5) # 前5个epoch不剪枝
    )

    # 2. 开始优化，n_trials是总共要尝试的超参数组合数量
    study.optimize(objective, n_trials=100, timeout=3600*2) # 运行100次trial，或最多2小时

    # 3. 打印优化结果
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "="*40)
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value (Max Validation Micro-F1): {trial.value:.4f}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*40)

    # 你也可以使用 Optuna提供的可视化工具来分析结果
    # pip install plotly kaleido
    # study.trials_dataframe().to_csv("optuna_results.csv")
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()


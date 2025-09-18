# main_optuna.py (支持持久化和可视化的最终版)
import argparse
import time
import os

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入 optuna
import optuna
from optuna.trial import TrialState

# [新增] 导入可视化模块
import optuna.visualization as vis

from spikenet import dataset
from spikenet_x.model import SpikeTDANet
from texttable import Texttable
import numpy as np
from torch_geometric.utils import subgraph
from typing import Tuple

# --- [Optuna] 将固定的参数定义为全局常量 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = "Patent"
DATAPATH = '/data4/zhengzhuoyu/data'
EPOCHS = 30
BATCH_SIZE = 256
TRAIN_SIZE = 0.8
VAL_SIZE = 0.05
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE
SPLIT_SEED = 4222
SEED = 2025


def sample_subgraph(nodes: torch.Tensor, edge_index_full: torch.Tensor, num_neighbors: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = edge_index_full
    device = row.device
    nodes = nodes.to(device)
    node_mask = torch.isin(row, nodes)
    edge_index_subset = edge_index_full[:, node_mask]
    neighbors_all = torch.unique(edge_index_subset[1])
    if num_neighbors > 0 and neighbors_all.numel() > 0:
        target_num_neighbors = nodes.numel() * num_neighbors
        if neighbors_all.numel() > target_num_neighbors:
            perm = torch.randperm(neighbors_all.numel(), device=device)[:target_num_neighbors]
            neighbors = neighbors_all[perm]
        else:
            neighbors = neighbors_all
    else:
        neighbors = neighbors_all
    subgraph_nodes = torch.unique(torch.cat([nodes, neighbors]))
    subgraph_edge_index, _ = subgraph(
        subset=subgraph_nodes,
        edge_index=edge_index_full,
        relabel_nodes=True,
        num_nodes=None
    )
    subgraph_nodes_sorted, _ = torch.sort(subgraph_nodes)
    nodes_local_index = torch.searchsorted(subgraph_nodes_sorted, nodes)
    return subgraph_nodes_sorted, subgraph_edge_index, nodes_local_index

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
T = len(data)
N = data.num_nodes
d_in = data.num_features
edge_list = [snapshot.edge_index for snapshot in data]
edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(DEVICE)
H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(DEVICE)
time_idx_full = torch.arange(T, device=DEVICE)
print("--- Data Ready ---")


# --- [Optuna] 超参 ---
def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    d = 128
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    layers = trial.suggest_int("layers", 1, 4) 
    W = trial.suggest_int("W", 8, 48, step=8)
    readout = trial.suggest_categorical("readout", ["mean", "last"])
    lif_tau = trial.suggest_float("lif_tau", 0.8, 0.99)
    lif_alpha = trial.suggest_float("lif_alpha", 0.5, 2.0)
    surrogate = trial.suggest_categorical("surrogate", ["sigmoid", "triangle"])
    num_neighbors = trial.suggest_int("num_neighbors", 15, 35, step=5)
    spike_reg_coeff = trial.suggest_float("spike_reg_coeff", 1e-6, 1e-4, log=True)
    target_spike_rate = trial.suggest_float("target_spike_rate", 0.1, 0.3)
    
    if d % heads != 0:
        raise optuna.exceptions.TrialPruned()

    model = SpikeTDANet(
        d_in=d_in, d=d, layers=layers, heads=heads, W=W, out_dim=data.num_classes,
        readout=readout, lif_tau=lif_tau, lif_v_threshold=1.0, lif_alpha=lif_alpha,
        lif_surrogate=surrogate
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    def train_model_step(epoch):
        model.train()
        total_loss = 0
        for nodes in train_loader:
            nodes = nodes.to(DEVICE)
            subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors)
            H0_subgraph = H0_full[:, subgraph_nodes, :]
            optimizer.zero_grad()
            output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
            subgraph_logits = output['logits']
            loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])
            spike_rate = output['S_list'].float().mean()
            if epoch > 10:
                loss = loss + spike_reg_coeff * (spike_rate - target_spike_rate).abs()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test_model_step(loader):
        model.eval()
        logits_list, labels_list = [], []
        for nodes in loader:
            nodes = nodes.to(DEVICE)
            subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors)
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

    best_val_micro = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_model_step(epoch)
        val_macro, val_micro = test_model_step(val_loader)
        if val_micro > best_val_micro:
            best_val_micro = val_micro
        trial.report(val_micro, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_val_micro


if __name__ == "__main__":
    study_name = f"spiketdanet_opt_{DATASET_NAME}"  # e.g., "spiketdanet_opt_DBLP"
    storage_name = f"sqlite:///{study_name}.db"
    
    print(f"Study Name: {study_name}")
    print(f"Storage: {storage_name}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, 
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=3)
    )

    study.optimize(objective
                #    , n_trials=200
                   , timeout=3600*8
                   )

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

    print("\nGenerating visualization reports...")
    
    # 1. 优化历史图：展示每次试验的分数
    fig_history = vis.plot_optimization_history(study)
    fig_history.write_html("optuna_report_history.html")

    # 2. 参数重要性图：分析哪个超参数对结果影响最大
    try:
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html("optuna_report_importance.html")
    except Exception as e:
        print(f"Could not generate importance plot: {e}")


    # 3. 参数切片图：观察单个超参数与分数的关系
    fig_slice = vis.plot_slice(study)
    fig_slice.write_html("optuna_report_slice.html")
    
    print(f"Reports saved to .html files. You can open them with your browser.")
    print(f"Database saved to '{study_name}.db'.")


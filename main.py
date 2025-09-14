import argparse
import time
import os

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
# 更改导入，直接使用新模型名称
from spikenet_x.model import SpikeTDANet
from texttable import Texttable
import numpy as np


def sample_subgraph(nodes: torch.Tensor, edge_index_full: torch.Tensor, num_neighbors: int = -1):
    """
    以一批种子节点 nodes 抽 1-hop 子图，并返回：
      - subgraph_nodes: 子图包含的全局节点 id，形状 [N_sub]
      - subgraph_edge_index: 子图边(局部id)，形状 [2, E_sub]
      - nodes_local_index: 种子在子图里的局部索引，形状 [B]
    关键改动：保留子图内部的“所有边”（src/dst 都在子图内），解锁多层传播。
    """
    row, col = edge_index_full
    device = row.device
    nodes = nodes.to(device)

    # 1) 先收集邻居（全收或限量）
    if num_neighbors == -1:
        mask = torch.isin(row, nodes)
        neighbors = col[mask]
    else:
        # 简洁做法：对所有与种子相连的邻居做全局采样，期望规模 ≈ B * num_neighbors
        mask = torch.isin(row, nodes)
        neighbors_all = col[mask]
        target = nodes.numel() * int(num_neighbors)
        if neighbors_all.numel() > target > 0:
            perm = torch.randperm(neighbors_all.numel(), device=device)[:target]
            neighbors = neighbors_all[perm]
        else:
            neighbors = neighbors_all

    # 2) 子图节点集合：种子 ∪ 采样邻居
    subgraph_nodes = torch.unique(torch.cat([nodes, neighbors], dim=0))

    # 3) **关键修复**：仅保留子图内部的边（src/dst 都在 subgraph_nodes）
    mask_src = torch.isin(row, subgraph_nodes)
    mask_dst = torch.isin(col, subgraph_nodes)
    edge_mask = mask_src & mask_dst
    subgraph_edge_index_global = edge_index_full[:, edge_mask]  # 仍是“全局 id”

    # 4) 将全局 id 映射到局部 id（纯 Torch，避免 Python 循环/字典）
    subgraph_nodes_sorted, _ = torch.sort(subgraph_nodes)  # searchsorted 需要有序
    src_global = subgraph_edge_index_global[0]
    dst_global = subgraph_edge_index_global[1]
    src_local = torch.searchsorted(subgraph_nodes_sorted, src_global)
    dst_local = torch.searchsorted(subgraph_nodes_sorted, dst_global)
    subgraph_edge_index = torch.stack([src_local, dst_local], dim=0)

    # 5) 种子节点的局部索引
    nodes_local_index = torch.searchsorted(subgraph_nodes_sorted, nodes)

    return subgraph_nodes_sorted, subgraph_edge_index, nodes_local_index


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def tab_printer(args):
    """Function to print the logs in a nice tabular format."""
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
    print(t.draw())


class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):

        super().__init__()

        from spikenet.utils import RandomWalkSampler, Sampler, add_selfloops

        tau = 1.0
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(
                add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(
                adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix))
                            for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                              for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        for hid in hids:
            aggregators.append(SAGEAggregator(in_features, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.p = p
        self.pooling = nn.Linear(len(data) * in_features, out_features)

    def encode(self, nodes):
        spikes = []
        sizes = self.sizes
        for time_step in range(len(data)):

            snapshot = data[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]

            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in sizes:
                size_1 = max(int(size * self.p), 1)
                size_2 = size - size_1

                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)

                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            for i, aggregator in enumerate(self.aggregators):
                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))

                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)
        return self.pooling(spikes)


parser = argparse.ArgumentParser()
# 更新命令行参数以反映新模型名称
parser.add_argument("--model", nargs="?", default="spikenet",
                    help="Model to use ('spikenet', 'spiketdanet'). (default: spikenet)")
parser.add_argument("--dataset", nargs="?", default="DBLP",
                    help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+',
                    default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
parser.add_argument("--aggr", nargs="?", default="mean",
                    help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage",
                    help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--train_size', type=float, default=0.4,
                    help='Ratio of nodes for training. (default: 0.4)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--p', type=float, default=0.5,
                    help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='Dropout probability. (default: 0.65)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')
parser.add_argument('--datapath', type=str, default='./data',
                    help='Wheres your data?, Default is ./data')

# SpikeTDANet specific args
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for SpikeTDANet. (default: 4)')
parser.add_argument('--W', type=int, default=32, help='Time window size for SpikeTDANet. (default: 32)')
parser.add_argument('--readout', type=str, default='mean', choices=['last','mean'],
                    help="Temporal readout for logits. (default: 'mean')")

# 新增：模型保存、加载与测试参数
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='Directory to save model checkpoints. (default: checkpoints)')
parser.add_argument('--resume_path', type=str, default=None,
                    help='Path to a checkpoint file to resume training from. (default: None)')
parser.add_argument('--test_model_path', type=str, default=None,
                    help='Path to a model file to load for testing only. (default: None)')


try:
    args = parser.parse_args()
    args.test_size = 1 - args.train_size
    args.train_size = args.train_size - 0.05
    args.val_size = 0.05
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

if args.dataset.lower() == "dblp":
    data = dataset.DBLP(root = args.datapath)
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall(root = args.datapath)
elif args.dataset.lower() == "patent":
    data = dataset.Patent(root = args.datapath)
else:
    raise ValueError(
        f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")

# train:val:test
data.split_nodes(train_size=args.train_size, val_size=args.val_size,
                 test_size=args.test_size, random_state=args.split_seed)

set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y = data.y.to(device)

train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
                        pin_memory=False, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=False)

# 更新模型选择逻辑
if args.model == 'spiketdanet':

    def train_model():
        model.train()
        total_loss = 0
        num_neighbors_to_sample = 10
        for nodes in tqdm(train_loader, desc='Training'):
            nodes = nodes.to(device)
            subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)

            H0_subgraph = H0_full[:, subgraph_nodes, :]

            N_sub = subgraph_nodes.numel()
            if subgraph_edge_index.numel() > 0:
                assert int(subgraph_edge_index.max().item()) < N_sub and int(subgraph_edge_index.min().item()) >= 0, \
                    f"边索引越界：[{int(subgraph_edge_index.min())}, {int(subgraph_edge_index.max())}]，但 N_sub={N_sub}"
            assert H0_subgraph.size(1) == N_sub, f"H0_subgraph 第二维应等于 N_sub，但拿到 {H0_subgraph.size()} vs N_sub={N_sub}"

            optimizer.zero_grad()

            output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
            subgraph_logits = output['logits']

            loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])

            spike_rate = output['S_list'].float().mean()
            if epoch > 10:
                loss = loss + 2e-5 * (spike_rate - 0.1).abs()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test_model(loader):
        model.eval()
        logits_list = []
        labels_list = []
        num_neighbors_to_sample = 25
        for nodes in tqdm(loader, desc='Testing'):
            nodes = nodes.to(device)
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

    # --- SpikeTDANet Training and Evaluation ---

    # 1. Data Preparation (Full graph data)
    print("Preparing data for SpikeTDANet...")
    T = len(data)
    N = data.num_nodes
    d_in = data.num_features

    edge_list = [snapshot.edge_index for snapshot in data]
    edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(device)
    H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(device)
    time_idx_full = torch.arange(T, device=device)

    # 2. Model, Optimizer, Loss
    model = SpikeTDANet(
        d_in=d_in,
        d=args.hids[0],
        layers=len(args.sizes),
        heads=args.heads,
        W=args.W,
        out_dim=data.num_classes,
        readout=args.readout,
        lif_tau_theta=0.95,
        lif_gamma=0.20,
        lif_beta=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()


    # --- Test-only mode ---
    if args.test_model_path:
        print(f"Loading model from {args.test_model_path} for testing...")
        checkpoint = torch.load(args.test_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_macro, test_micro = test_model(test_loader)
        print(f"Test Results: Macro-F1={test_macro:.4f}, Micro-F1={test_micro:.4f}")
        exit(0)

    # 3. Training Loop
    start_epoch = 1
    best_val_metric = 0
    best_test_metric = (0, 0)

    # --- Resume from checkpoint ---
    if args.resume_path:
        if os.path.exists(args.resume_path):
            print(f"Resuming training from {args.resume_path}...")
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_metric = checkpoint.get('best_val_metric', 0)
            print(f"Resumed from epoch {start_epoch-1}. Best val metric so far: {best_val_metric:.4f}")
        else:
            print(f"Warning: Checkpoint path {args.resume_path} not found. Starting from scratch.")

    print("Starting SpikeTDANet training...")
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == int(0.1 * args.epochs):
            for layer in model.layers:
                layer.lif_cell.beta = 2.0

        train_model()
        val_metric = test_model(val_loader)
        test_metric = test_model(test_loader)

        is_best = val_metric[1] > best_val_metric
        if is_best:
            best_val_metric = val_metric[1]
            best_test_metric = test_metric

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.dataset}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'test_metric_at_best_val': test_metric,
            }, checkpoint_path)
            print(f"Epoch {epoch:03d}: New best model saved to {checkpoint_path} with Val Micro: {best_val_metric:.4f}")

        end = time.time()
        print(
            f'Epoch: {epoch:03d}, Val Micro: {val_metric[1]:.4f}, Test Micro: {test_metric[1]:.4f}, '
            f'Best Test: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time: {end-start:.2f}s'
        )

else:
    # --- Original SpikeNet Training and Evaluation ---
    model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
                     dropout=args.dropout, sampler=args.sampler, p=args.p,
                     aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
                     hids=args.hids, act=args.neuron, bias=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def train():
        model.train()
        for nodes in tqdm(train_loader, desc='Training'):
            optimizer.zero_grad()
            loss_fn(model(nodes), y[nodes]).backward()
            optimizer.step()

    @torch.no_grad()
    def test(loader):
        model.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(model(nodes))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        metric_macro = metrics.f1_score(labels, logits, average='macro')
        metric_micro = metrics.f1_score(labels, logits, average='micro')
        return metric_macro, metric_micro

    best_val_metric = test_metric = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train()
        val_metric, test_metric = test(val_loader), test(test_loader)
        if val_metric[1] > best_val_metric:
            best_val_metric = val_metric[1]
            best_test_metric = test_metric
        end = time.time()
        print(
            f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')

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
from spikenet_x.model import SpikeNetX
from texttable import Texttable # Added for tab_printer
import numpy as np # Added for set_seed


def sample_subgraph(nodes, edge_index_full, num_neighbors=-1):
    """
    Samples a 1-hop subgraph for the given seed nodes, with an optional
    limit on the number of neighbors per node.
    
    Args:
        nodes (Tensor): Seed nodes of shape [B].
        edge_index_full (Tensor): Full graph edge index of shape [2, E_full].
        num_neighbors (int): Number of neighbors to sample for each node.
                             -1 means all neighbors.

    Returns:
        subgraph_nodes (Tensor): Unique nodes in the subgraph, shape [N_sub].
        subgraph_edge_index (Tensor): Edge index of the subgraph, shape [2, E_sub].
        nodes_local_index (Tensor): Indices of the seed nodes within subgraph_nodes.
    """
    row, col = edge_index_full
    
    # Efficiently find neighbors for each node in the batch
    sampled_neighbors = []
    if num_neighbors == -1:
        # Get all neighbors
        node_mask = torch.isin(row, nodes)
        neighbors = edge_index_full[1, node_mask]
        sampled_neighbors.append(neighbors)
    else:
        # Sample a fixed number of neighbors for each node
        for node_id in nodes:
            node_mask = (row == node_id)
            node_neighbors = col[node_mask]
            if node_neighbors.numel() > num_neighbors:
                # Randomly sample neighbors
                perm = torch.randperm(node_neighbors.numel(), device=nodes.device)[:num_neighbors]
                node_neighbors = node_neighbors[perm]
            sampled_neighbors.append(node_neighbors)

    if sampled_neighbors:
        neighbors = torch.cat(sampled_neighbors)
        subgraph_nodes = torch.cat([nodes, neighbors]).unique()
    else:
        subgraph_nodes = nodes.unique()
    
    # This is the correct set of edges for the 1-hop subgraph
    edge_mask = torch.isin(row, nodes) & torch.isin(col, subgraph_nodes)
    subgraph_edge_index_global = edge_index_full[:, edge_mask]

    # Map global node indices to local indices in the subgraph
    node_map = {global_id.item(): local_id for local_id, global_id in enumerate(subgraph_nodes)}
    
    # Remap edge_index to local subgraph indices
    subgraph_edge_index = torch.tensor(
        [[node_map[src.item()] for src in subgraph_edge_index_global[0]],
         [node_map[dst.item()] for dst in subgraph_edge_index_global[1]]],
        dtype=torch.long, device=nodes.device
    )

    # Get local indices of the seed nodes
    nodes_local_index = torch.tensor([node_map[n.item()] for n in nodes], dtype=torch.long, device=nodes.device)

    return subgraph_nodes, subgraph_edge_index, nodes_local_index


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
parser.add_argument("--model", nargs="?", default="spikenet",
                    help="Model to use ('spikenet', 'spikenetx'). (default: spikenet)")
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
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout probability. (default: 0.7)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')

# SpikeNet-X specific args
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for SpikeNet-X. (default: 4)')
parser.add_argument('--topk', type=int, default=8, help='Top-k neighbors for SpikeNet-X attention. (default: 8)')
parser.add_argument('--W', type=int, default=8, help='Time window size for SpikeNet-X. (default: 8)')
parser.add_argument('--attn_impl', type=str, default='sparse', choices=['dense','sparse'],
                    help='Attention kernel for SpikeNet-X: "dense" (fallback) or "sparse" (scales to big graphs). (default: "sparse")')

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
    data = dataset.DBLP()
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall()
elif args.dataset.lower() == "patent":
    data = dataset.Patent()
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
                        pin_memory=False, batch_size=200000, shuffle=False)
test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)

if args.model == 'spikenetx':
    # --- SpikeNet-X Training and Evaluation (with batching) ---

    # 1. Data Preparation (Full graph data)
    print("Preparing data for SpikeNet-X...")
    T = len(data)
    N = data.num_nodes
    d_in = data.num_features
    
    edge_list = [snapshot.edge_index for snapshot in data]
    edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(device)
    H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(device)
    time_idx_full = torch.arange(T, device=device)

    # 2. Model, Optimizer, Loss
    model = SpikeNetX(
        d_in=d_in,
        d=args.hids[0],
        layers=len(args.sizes),
        heads=args.heads,
        out_dim=data.num_classes,
        topk=args.topk,
        W=args.W,
        attn_impl=args.attn_impl
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # --- Test-only mode ---
    if args.test_model_path:
        print(f"Loading model from {args.test_model_path} for testing...")
        checkpoint = torch.load(args.test_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_macro, test_micro = test_spikenetx(test_loader)
        print(f"Test Results: Macro-F1={test_macro:.4f}, Micro-F1={test_micro:.4f}")
        exit(0)

    def train_spikenetx():
        model.train()
        total_loss = 0
        # Let's use a fixed number of neighbors for now to control memory
        num_neighbors_to_sample = 10 
        for nodes in tqdm(train_loader, desc='Training'):
            nodes = nodes.to(device)
            subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
            
            H0_subgraph = H0_full[:, subgraph_nodes, :]
            
            optimizer.zero_grad()
            
            # The model's output `repr` and `logits` are for all nodes in the subgraph
            output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
            subgraph_logits = output['logits']

            # We only compute the loss on the seed nodes of the batch
            loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test_spikenetx(loader):
        model.eval()
        logits_list = []
        labels_list = []
        num_neighbors_to_sample = 10 # Use the same for testing
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
            best_val_metric = checkpoint.get('best_val_metric', 0) # Use .get for backward compatibility
            print(f"Resumed from epoch {start_epoch-1}. Best val metric so far: {best_val_metric:.4f}")
        else:
            print(f"Warning: Checkpoint path {args.resume_path} not found. Starting from scratch.")

    print("Starting SpikeNet-X training...")
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_spikenetx()
        val_metric = test_spikenetx(val_loader)
        test_metric = test_spikenetx(test_loader)
        
        is_best = val_metric[1] > best_val_metric
        if is_best:
            best_val_metric = val_metric[1]
            best_test_metric = test_metric

            # --- Save checkpoint ---
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

    # save bianry node embeddings (spikes)
    # emb = model.encode(torch.arange(data.num_nodes)).cpu()
    # torch.save(emb, 'emb.pth')

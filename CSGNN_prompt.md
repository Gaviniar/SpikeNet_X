# Table of Contents
- F:\SomeProjects\CSGNN\.gitignore
- F:\SomeProjects\CSGNN\generate_feature.py
- F:\SomeProjects\CSGNN\LICENSE
- F:\SomeProjects\CSGNN\main.py
- F:\SomeProjects\CSGNN\main_static.py
- F:\SomeProjects\CSGNN\setup.py
- F:\SomeProjects\CSGNN\spikenet\dataset.py
- F:\SomeProjects\CSGNN\spikenet\deepwalk.py
- F:\SomeProjects\CSGNN\spikenet\layers.py
- F:\SomeProjects\CSGNN\spikenet\neuron.py
- F:\SomeProjects\CSGNN\spikenet\sample_neighber.cpp
- F:\SomeProjects\CSGNN\spikenet\utils.py
- F:\SomeProjects\CSGNN\spikenet_x\delayline.py
- F:\SomeProjects\CSGNN\spikenet_x\lif_cell.py
- F:\SomeProjects\CSGNN\spikenet_x\masked_ops.py
- F:\SomeProjects\CSGNN\spikenet_x\minimal_example.py
- F:\SomeProjects\CSGNN\spikenet_x\model.py
- F:\SomeProjects\CSGNN\spikenet_x\rel_time.py
- F:\SomeProjects\CSGNN\spikenet_x\spikenetx_layer.py
- F:\SomeProjects\CSGNN\spikenet_x\spiketdanet_layer.py
- F:\SomeProjects\CSGNN\spikenet_x\sta.py
- F:\SomeProjects\CSGNN\spikenet_x\sta_sparse.py
- F:\SomeProjects\CSGNN\spikenet_x\surrogate_lif_cell.py
- F:\SomeProjects\CSGNN\spikenet_x\__init__.py
- F:\SomeProjects\CSGNN\spikenet_x\new_modules\delay_line.py
- F:\SomeProjects\CSGNN\spikenet_x\new_modules\spatial_gnn_wrapper.py
- F:\SomeProjects\CSGNN\spikenet_x\new_modules\sta_gnn_agg.py
- F:\SomeProjects\CSGNN\spikenet_x\new_modules\sta_gnn_agg_optimized.py
- F:\SomeProjects\CSGNN\spikenet_x\new_modules\__init__.py

## File: F:\SomeProjects\CSGNN\.gitignore

- Extension: 
- Language: unknown
- Size: 1281 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```unknown
  1 | # Custom
  2 | *.idea
  3 | *.pdf
  4 | *.txt
  5 | *.npy
  6 | !requirements.txt
  7 | data/
  8 | # Byte-compiled / optimized / DLL files
  9 | __pycache__/
 10 | *.py[cod]
 11 | *$py.class
 12 | 
 13 | # C extensions
 14 | *.so
 15 | 
 16 | # Distribution / packaging
 17 | .Python
 18 | env/
 19 | build/
 20 | develop-eggs/
 21 | dist/
 22 | downloads/
 23 | eggs/
 24 | .eggs/
 25 | lib/
 26 | lib64/
 27 | parts/
 28 | sdist/
 29 | var/
 30 | *.egg-info/
 31 | .installed.cfg
 32 | *.egg
 33 | 
 34 | # PyInstaller
 35 | #  Usually these files are written by a python script from a template
 36 | #  before PyInstaller builds the exe, so as to inject date/other infos into it.
 37 | *.manifest
 38 | *.spec
 39 | 
 40 | # Installer logs
 41 | pip-log.txt
 42 | pip-delete-this-directory.txt
 43 | 
 44 | # Unit test / coverage reports
 45 | htmlcov/
 46 | .tox/
 47 | .coverage
 48 | .coverage.*
 49 | .cache
 50 | nosetests.xml
 51 | coverage.xml
 52 | *,cover
 53 | .hypothesis/
 54 | 
 55 | # Translations
 56 | *.mo
 57 | *.pot
 58 | 
 59 | # Django stuff:
 60 | *.log
 61 | local_settings.py
 62 | 
 63 | # Flask stuff:
 64 | instance/
 65 | .webassets-cache
 66 | 
 67 | # Scrapy stuff:
 68 | .scrapy
 69 | 
 70 | # Sphinx documentation
 71 | docs/build/
 72 | 
 73 | # PyBuilder
 74 | target/
 75 | 
 76 | # IPython Notebook
 77 | .ipynb_checkpoints
 78 | 
 79 | # pyenv
 80 | .python-version
 81 | 
 82 | # celery beat schedule file
 83 | celerybeat-schedule
 84 | 
 85 | # dotenv
 86 | .env
 87 | 
 88 | # virtualenv
 89 | venv/
 90 | ENV/
 91 | 
 92 | # Spyder project settings
 93 | .spyderproject
 94 | 
 95 | # Rope project settings
 96 | .ropeproject
 97 | 
 98 | *.pickle
 99 | .vscode
100 | 
101 | # checkpoint
102 | *.h5
103 | *.pkl
104 | *.pth
105 | 
106 | # Mac files
107 | .DS_Store
```

## File: F:\SomeProjects\CSGNN\generate_feature.py

- Extension: .py
- Language: python
- Size: 1176 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | import argparse
 2 | 
 3 | import numpy as np
 4 | from tqdm import tqdm
 5 | 
 6 | from spikenet import dataset
 7 | from spikenet.deepwalk import DeepWalk
 8 | 
 9 | parser = argparse.ArgumentParser()
10 | parser.add_argument("--dataset", nargs="?", default="DBLP",
11 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
12 | parser.add_argument('--normalize', action='store_true',
13 |                     help='Whether to normalize output embedding. (default: False)')
14 | 
15 | 
16 | args = parser.parse_args()
17 | if args.dataset.lower() == "dblp":
18 |     data = dataset.DBLP()
19 | elif args.dataset.lower() == "tmall":
20 |     data = dataset.Tmall()
21 | elif args.dataset.lower() == "patent":
22 |     data = dataset.Patent()
23 | else:
24 |     raise ValueError(
25 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
26 | 
27 | 
28 | model = DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16)
29 | xs = []
30 | for g in tqdm(data.adj):
31 |     model.fit(g)
32 |     x = model.get_embedding(normalize=args.normalize)
33 |     xs.append(x)
34 | 
35 | 
36 | file_path = f'{data.root}/{data.name}/{data.name}.npy'
37 | np.save(file_path, np.stack(xs, axis=0)) # [T, N, F]
38 | print(f"Generated node feautures saved at {file_path}")
```

## File: F:\SomeProjects\CSGNN\LICENSE

- Extension: 
- Language: unknown
- Size: 1112 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```unknown
 1 | MIT License
 2 | 
 3 | Copyright (c) 2022 Jintang Li, Sun Yat-sen University
 4 | 
 5 | Permission is hereby granted, free of charge, to any person obtaining a copy
 6 | of this software and associated documentation files (the "Software"), to deal
 7 | in the Software without restriction, including without limitation the rights
 8 | to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 9 | copies of the Software, and to permit persons to whom the Software is
10 | furnished to do so, subject to the following conditions:
11 | 
12 | The above copyright notice and this permission notice shall be included in all
13 | copies or substantial portions of the Software.
14 | 
15 | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
16 | IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
17 | FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
18 | AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
19 | LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
20 | OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
21 | SOFTWARE.
```

## File: F:\SomeProjects\CSGNN\main.py

- Extension: .py
- Language: python
- Size: 21049 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2025-09-16 22:21:02

### Code

```python
  1 | import argparse
  2 | import time
  3 | import os
  4 | 
  5 | import torch
  6 | import torch.nn as nn
  7 | from sklearn import metrics
  8 | from torch.utils.data import DataLoader
  9 | from tqdm import tqdm
 10 | 
 11 | from spikenet import dataset, neuron
 12 | from spikenet.layers import SAGEAggregator
 13 | # 更改导入，直接使用新模型名称
 14 | from spikenet_x.model import SpikeTDANet
 15 | from texttable import Texttable
 16 | import numpy as np
 17 | from torch_geometric.utils import subgraph
 18 | from typing import Tuple
 19 | 
 20 | def sample_subgraph(nodes: torch.Tensor, edge_index_full: torch.Tensor, num_neighbors: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 21 |     """
 22 |     高效的1-hop子图采样函数。
 23 |     - 强制限制每个节点的邻居数量，以控制计算复杂度。
 24 |     - 使用 torch_geometric.utils.subgraph 进行快速的边提取和节点重标签。
 25 |     """
 26 |     row, col = edge_index_full
 27 |     device = row.device
 28 |     nodes = nodes.to(device)
 29 | 
 30 |     # 1) 找到所有与种子节点相连的边和邻居
 31 |     node_mask = torch.isin(row, nodes)
 32 |     edge_index_subset = edge_index_full[:, node_mask]
 33 |     neighbors_all = torch.unique(edge_index_subset[1])
 34 |     
 35 |     # 2) 如果需要，对邻居进行采样
 36 |     if num_neighbors > 0 and neighbors_all.numel() > 0:
 37 |         # 为了简化，我们进行全局采样。更复杂的实现可以做到per-node采样。
 38 |         # 即使是全局采样，也能有效控制子图规模。
 39 |         target_num_neighbors = nodes.numel() * num_neighbors
 40 |         if neighbors_all.numel() > target_num_neighbors:
 41 |             perm = torch.randperm(neighbors_all.numel(), device=device)[:target_num_neighbors]
 42 |             neighbors = neighbors_all[perm]
 43 |         else:
 44 |             neighbors = neighbors_all
 45 |     else:
 46 |         neighbors = neighbors_all
 47 | 
 48 |     # 3) 构建子图节点集并获取重标签后的边
 49 |     subgraph_nodes = torch.unique(torch.cat([nodes, neighbors]))
 50 |     
 51 |     # 使用PyG的subgraph函数，它会返回重标签后的边索引
 52 |     # relabel_nodes=True 是关键
 53 |     subgraph_edge_index, _ = subgraph(
 54 |         subset=subgraph_nodes,
 55 |         edge_index=edge_index_full,
 56 |         relabel_nodes=True,
 57 |         num_nodes=None # PyG会自动推断
 58 |     )
 59 | 
 60 |     # 4) 找到原始种子节点在子图中的新索引
 61 |     # 我们需要创建一个从全局ID到新局部ID的映射
 62 |     # torch.searchsorted 是一个高效的方法
 63 |     subgraph_nodes_sorted, _ = torch.sort(subgraph_nodes)
 64 |     nodes_local_index = torch.searchsorted(subgraph_nodes_sorted, nodes)
 65 |     
 66 |     return subgraph_nodes_sorted, subgraph_edge_index, nodes_local_index
 67 | 
 68 | 
 69 | def set_seed(seed):
 70 |     np.random.seed(seed)
 71 |     torch.manual_seed(seed)
 72 |     torch.cuda.manual_seed(seed)
 73 | 
 74 | def tab_printer(args):
 75 |     """Function to print the logs in a nice tabular format."""
 76 |     args = vars(args)
 77 |     keys = sorted(args.keys())
 78 |     t = Texttable()
 79 |     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
 80 |     print(t.draw())
 81 | 
 82 | 
 83 | class SpikeNet(nn.Module):
 84 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
 85 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 86 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):
 87 | 
 88 |         super().__init__()
 89 | 
 90 |         from spikenet.utils import RandomWalkSampler, Sampler, add_selfloops
 91 | 
 92 |         tau = 1.0
 93 |         if sampler == 'rw':
 94 |             self.sampler = [RandomWalkSampler(
 95 |                 add_selfloops(adj_matrix)) for adj_matrix in data.adj]
 96 |             self.sampler_t = [RandomWalkSampler(add_selfloops(
 97 |                 adj_matrix)) for adj_matrix in data.adj_evolve]
 98 |         elif sampler == 'sage':
 99 |             self.sampler = [Sampler(add_selfloops(adj_matrix))
100 |                             for adj_matrix in data.adj]
101 |             self.sampler_t = [Sampler(add_selfloops(adj_matrix))
102 |                               for adj_matrix in data.adj_evolve]
103 |         else:
104 |             raise ValueError(sampler)
105 | 
106 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
107 | 
108 |         for hid in hids:
109 |             aggregators.append(SAGEAggregator(in_features, hid,
110 |                                               concat=concat, bias=bias,
111 |                                               aggr=aggr))
112 | 
113 |             if act == "IF":
114 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
115 |             elif act == 'LIF':
116 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
117 |             elif act == 'PLIF':
118 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
119 |             else:
120 |                 raise ValueError(act)
121 | 
122 |             in_features = hid * 2 if concat else hid
123 | 
124 |         self.aggregators = aggregators
125 |         self.dropout = nn.Dropout(dropout)
126 |         self.snn = snn
127 |         self.sizes = sizes
128 |         self.p = p
129 |         self.pooling = nn.Linear(len(data) * in_features, out_features)
130 | 
131 |     def encode(self, nodes):
132 |         spikes = []
133 |         sizes = self.sizes
134 |         for time_step in range(len(data)):
135 | 
136 |             snapshot = data[time_step]
137 |             sampler = self.sampler[time_step]
138 |             sampler_t = self.sampler_t[time_step]
139 | 
140 |             x = snapshot.x
141 |             h = [x[nodes].to(device)]
142 |             num_nodes = [nodes.size(0)]
143 |             nbr = nodes
144 |             for size in sizes:
145 |                 size_1 = max(int(size * self.p), 1)
146 |                 size_2 = size - size_1
147 | 
148 |                 if size_2 > 0:
149 |                     nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
150 |                     nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
151 |                     nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
152 |                 else:
153 |                     nbr = sampler(nbr, size_1).view(-1)
154 | 
155 |                 num_nodes.append(nbr.size(0))
156 |                 h.append(x[nbr].to(device))
157 | 
158 |             for i, aggregator in enumerate(self.aggregators):
159 |                 self_x = h[:-1]
160 |                 neigh_x = []
161 |                 for j, n_x in enumerate(h[1:]):
162 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
163 | 
164 |                 out = self.snn[i](aggregator(self_x, neigh_x))
165 |                 if i != len(sizes) - 1:
166 |                     out = self.dropout(out)
167 |                     h = torch.split(out, num_nodes[:-(i + 1)])
168 | 
169 |             spikes.append(out)
170 |         spikes = torch.cat(spikes, dim=1)
171 |         neuron.reset_net(self)
172 |         return spikes
173 | 
174 |     def forward(self, nodes):
175 |         spikes = self.encode(nodes)
176 |         return self.pooling(spikes)
177 | 
178 | 
179 | parser = argparse.ArgumentParser()
180 | # 更新命令行参数以反映新模型名称
181 | parser.add_argument("--model", nargs="?", default="spikenet",
182 |                     help="Model to use ('spikenet', 'spiketdanet'). (default: spikenet)")
183 | parser.add_argument("--dataset", nargs="?", default="DBLP",
184 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
185 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
186 | parser.add_argument('--hids', type=int, nargs='+',
187 |                     default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
188 | parser.add_argument("--aggr", nargs="?", default="mean",
189 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
190 | parser.add_argument("--sampler", nargs="?", default="sage",
191 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
192 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
193 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
194 | parser.add_argument("--neuron", nargs="?", default="LIF",
195 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
196 | parser.add_argument('--batch_size', type=int, default=1024,
197 |                     help='Batch size for training. (default: 1024)')
198 | parser.add_argument('--lr', type=float, default=5e-3,
199 |                     help='Learning rate for training. (default: 5e-3)')
200 | parser.add_argument('--wd', type=float, default=1e-4,
201 |                     help='weight decay for training. (default: 1e-4)')
202 | parser.add_argument('--train_size', type=float, default=0.4,
203 |                     help='Ratio of nodes for training. (default: 0.4)')
204 | parser.add_argument('--alpha', type=float, default=1.0,
205 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
206 | parser.add_argument('--p', type=float, default=0.5,
207 |                     help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
208 | parser.add_argument('--dropout', type=float, default=0.65,
209 |                     help='Dropout probability. (default: 0.65)')
210 | parser.add_argument('--epochs', type=int, default=100,
211 |                     help='Number of training epochs. (default: 100)')
212 | parser.add_argument('--concat', action='store_true',
213 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
214 | parser.add_argument('--seed', type=int, default=2022,
215 |                     help='Random seed for model. (default: 2022)')
216 | parser.add_argument('--datapath', type=str, default='./data',
217 |                     help='Wheres your data?, Default is ./data')
218 | 
219 | # SpikeTDANet specific args
220 | parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for SpikeTDANet. (default: 4)')
221 | parser.add_argument('--W', type=int, default=32, help='Time window size for SpikeTDANet. (default: 32)')
222 | parser.add_argument('--readout', type=str, default='mean', choices=['last','mean'],
223 |                     help="Temporal readout for logits. (default: 'mean')")
224 | 
225 | # 新增：模型保存、加载与测试参数
226 | parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
227 |                     help='Directory to save model checkpoints. (default: checkpoints)')
228 | parser.add_argument('--resume_path', type=str, default=None,
229 |                     help='Path to a checkpoint file to resume training from. (default: None)')
230 | parser.add_argument('--test_model_path', type=str, default=None,
231 |                     help='Path to a model file to load for testing only. (default: None)')
232 | 
233 | 
234 | try:
235 |     args = parser.parse_args()
236 |     args.test_size = 1 - args.train_size
237 |     args.train_size = args.train_size - 0.05
238 |     args.val_size = 0.05
239 |     args.split_seed = 42
240 |     tab_printer(args)
241 | except:
242 |     parser.print_help()
243 |     exit(0)
244 | 
245 | assert len(args.hids) == len(args.sizes), "must be equal!"
246 | 
247 | if args.dataset.lower() == "dblp":
248 |     data = dataset.DBLP(root = args.datapath)
249 | elif args.dataset.lower() == "tmall":
250 |     data = dataset.Tmall(root = args.datapath)
251 | elif args.dataset.lower() == "patent":
252 |     data = dataset.Patent(root = args.datapath)
253 | else:
254 |     raise ValueError(
255 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
256 | 
257 | # train:val:test
258 | data.split_nodes(train_size=args.train_size, val_size=args.val_size,
259 |                  test_size=args.test_size, random_state=args.split_seed)
260 | 
261 | set_seed(args.seed)
262 | 
263 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
264 | 
265 | y = data.y.to(device)
266 | 
267 | train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
268 | val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
269 |                         pin_memory=False, batch_size=args.batch_size, shuffle=False)
270 | test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=False)
271 | 
272 | # 更新模型选择逻辑
273 | if args.model == 'spiketdanet':
274 | 
275 |     # 替换 main.py 中的 train_model 函数
276 |     def train_model():
277 |         model.train()
278 |         total_loss = 0
279 |         total_spike_rate = 0  # <--- 新增：用于累加脉冲率
280 |         num_batches = 0       # <--- 新增：用于计算平均值
281 |         num_neighbors_to_sample = 25
282 |         for nodes in tqdm(train_loader, desc='Training'):
283 |             nodes = nodes.to(device)
284 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
285 | 
286 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
287 | 
288 |             N_sub = subgraph_nodes.numel()
289 |             if subgraph_edge_index.numel() > 0:
290 |                 assert int(subgraph_edge_index.max().item()) < N_sub and int(subgraph_edge_index.min().item()) >= 0, \
291 |                     f"边索引越界：[{int(subgraph_edge_index.min())}, {int(subgraph_edge_index.max())}]，但 N_sub={N_sub}"
292 |             assert H0_subgraph.size(1) == N_sub, f"H0_subgraph 第二维应等于 N_sub，但拿到 {H0_subgraph.size()} vs N_sub={N_sub}"
293 | 
294 |             optimizer.zero_grad()
295 | 
296 |             output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
297 |             subgraph_logits = output['logits']
298 | 
299 |             loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])
300 | 
301 |             spike_rate = output['S_list'].float().mean()
302 |             if epoch > 10:
303 |                 loss = loss + 2e-5 * (spike_rate - 0.15).abs()
304 |             loss.backward()
305 |             optimizer.step()
306 |             total_loss += loss.item()
307 |             total_spike_rate += spike_rate.item() # <--- 新增：累加当前batch的脉冲率
308 |             num_batches += 1                      # <--- 新增：批次计数
309 |             
310 |         avg_loss = total_loss / len(train_loader)
311 |         avg_spike_rate = total_spike_rate / num_batches 
312 |         return avg_loss, avg_spike_rate 
313 | 
314 |     @torch.no_grad()
315 |     def test_model(loader):
316 |         model.eval()
317 |         logits_list = []
318 |         labels_list = []
319 |         num_neighbors_to_sample = 25
320 |         for nodes in tqdm(loader, desc='Testing'):
321 |             nodes = nodes.to(device)
322 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
323 | 
324 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
325 | 
326 |             output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
327 |             subgraph_logits = output['logits']
328 | 
329 |             logits_list.append(subgraph_logits[nodes_local_index].cpu())
330 |             labels_list.append(y[nodes].cpu())
331 | 
332 |         logits = torch.cat(logits_list, dim=0).argmax(1)
333 |         labels = torch.cat(labels_list, dim=0)
334 | 
335 |         micro = metrics.f1_score(labels, logits, average='micro', zero_division=0)
336 |         macro = metrics.f1_score(labels, logits, average='macro', zero_division=0)
337 |         return macro, micro
338 | 
339 |     # --- SpikeTDANet Training and Evaluation ---
340 | 
341 |     # 1. Data Preparation (Full graph data)
342 |     print("Preparing data for SpikeTDANet...")
343 |     T = len(data)
344 |     N = data.num_nodes
345 |     d_in = data.num_features
346 | 
347 |     edge_list = [snapshot.edge_index for snapshot in data]
348 |     edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(device)
349 |     H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(device)
350 |     time_idx_full = torch.arange(T, device=device)
351 | 
352 |     # 2. Model, Optimizer, Loss
353 |     model = SpikeTDANet(
354 |         d_in=d_in,
355 |         d=args.hids[0],
356 |         layers=len(args.sizes),
357 |         heads=args.heads,
358 |         W=args.W,
359 |         out_dim=data.num_classes,
360 |         readout=args.readout,
361 |         # 传递新的LIF超参数
362 |         lif_tau=0.95,             # 膜电位衰减因子
363 |         lif_v_threshold=1.0,      # 脉冲阈值
364 |         lif_alpha=1.0,            # 替代梯度平滑度
365 |         lif_surrogate=args.surrogate # 从命令行参数获取替代函数类型
366 |     ).to(device)
367 | 
368 |     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
369 |     loss_fn = nn.CrossEntropyLoss()
370 | 
371 | 
372 |     # --- Test-only mode ---
373 |     if args.test_model_path:
374 |         print(f"Loading model from {args.test_model_path} for testing...")
375 |         checkpoint = torch.load(args.test_model_path, map_location=device)
376 |         model.load_state_dict(checkpoint['model_state_dict'])
377 |         test_macro, test_micro = test_model(test_loader)
378 |         print(f"Test Results: Macro-F1={test_macro:.4f}, Micro-F1={test_micro:.4f}")
379 |         exit(0)
380 | 
381 |     # 3. Training Loop
382 |     start_epoch = 1
383 |     best_val_metric = 0
384 |     best_test_metric = (0, 0)
385 | 
386 |     # --- Resume from checkpoint ---
387 |     if args.resume_path:
388 |         if os.path.exists(args.resume_path):
389 |             print(f"Resuming training from {args.resume_path}...")
390 |             checkpoint = torch.load(args.resume_path, map_location=device)
391 |             model.load_state_dict(checkpoint['model_state_dict'])
392 |             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
393 |             start_epoch = checkpoint['epoch'] + 1
394 |             best_val_metric = checkpoint.get('best_val_metric', 0)
395 |             print(f"Resumed from epoch {start_epoch-1}. Best val metric so far: {best_val_metric:.4f}")
396 |         else:
397 |             print(f"Warning: Checkpoint path {args.resume_path} not found. Starting from scratch.")
398 | 
399 |     print("Starting SpikeTDANet training...")
400 |     start = time.time()
401 |     # 替换 main.py 中的主训练循环
402 |     for epoch in range(start_epoch, args.epochs + 1):
403 |         if epoch == int(0.1 * args.epochs):
404 |             for layer in model.layers:
405 |                 layer.lif_cell.beta = torch.tensor(2.0, device=layer.lif_cell.beta.device)
406 | 
407 |         # ========== [修改开始] ==========
408 |         train_loss, train_spike_rate = train_model() # 接收返回的脉冲率
409 |         # ========== [修改结束] ==========
410 |         
411 |         val_metric = test_model(val_loader)
412 |         test_metric = test_model(test_loader)
413 | 
414 |         is_best = val_metric[1] > best_val_metric
415 |         if is_best:
416 |             best_val_metric = val_metric[1]
417 |             best_test_metric = test_metric
418 | 
419 |             os.makedirs(args.checkpoint_dir, exist_ok=True)
420 |             checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.dataset}.pth')
421 |             torch.save({
422 |                 'epoch': epoch,
423 |                 'model_state_dict': model.state_dict(),
424 |                 'optimizer_state_dict': optimizer.state_dict(),
425 |                 'best_val_metric': best_val_metric,
426 |                 'test_metric_at_best_val': test_metric,
427 |             }, checkpoint_path)
428 |             print(f"Epoch {epoch:03d}: New best model saved to {checkpoint_path} with Val Micro: {best_val_metric:.4f}")
429 | 
430 |         end = time.time()
431 |         # ========== [修改开始] ==========
432 |         # 在打印信息中加入 train_loss 和 train_spike_rate
433 |         print(
434 |             f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Spike Rate: {train_spike_rate:.4f}, '
435 |             f'Val Micro: {val_metric[1]:.4f}, Test Micro: {test_metric[1]:.4f}, '
436 |             f'Best Test: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time: {end-start:.2f}s'
437 |         )
438 |         # ========== [修改结束] ==========
439 | 
440 | else:
441 |     # --- Original SpikeNet Training and Evaluation ---
442 |     model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
443 |                      dropout=args.dropout, sampler=args.sampler, p=args.p,
444 |                      aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
445 |                      hids=args.hids, act=args.neuron, bias=True).to(device)
446 | 
447 |     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
448 |     loss_fn = nn.CrossEntropyLoss()
449 | 
450 |     def train():
451 |         model.train()
452 |         for nodes in tqdm(train_loader, desc='Training'):
453 |             optimizer.zero_grad()
454 |             loss_fn(model(nodes), y[nodes]).backward()
455 |             optimizer.step()
456 | 
457 |     @torch.no_grad()
458 |     def test(loader):
459 |         model.eval()
460 |         logits = []
461 |         labels = []
462 |         for nodes in loader:
463 |             logits.append(model(nodes))
464 |             labels.append(y[nodes])
465 |         logits = torch.cat(logits, dim=0).cpu()
466 |         labels = torch.cat(labels, dim=0).cpu()
467 |         logits = logits.argmax(1)
468 |         metric_macro = metrics.f1_score(labels, logits, average='macro')
469 |         metric_micro = metrics.f1_score(labels, logits, average='micro')
470 |         return metric_macro, metric_micro
471 | 
472 |     best_val_metric = test_metric = 0
473 |     start = time.time()
474 |     for epoch in range(1, args.epochs + 1):
475 |         train()
476 |         val_metric, test_metric = test(val_loader), test(test_loader)
477 |         if val_metric[1] > best_val_metric:
478 |             best_val_metric = val_metric[1]
479 |             best_test_metric = test_metric
480 |         end = time.time()
481 |         print(
482 |             f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
```

## File: F:\SomeProjects\CSGNN\main_static.py

- Extension: .py
- Language: python
- Size: 8038 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
  1 | import argparse
  2 | import os.path as osp
  3 | import time
  4 | 
  5 | import torch
  6 | import torch.nn as nn
  7 | from sklearn import metrics
  8 | from spikenet import dataset, neuron
  9 | from spikenet.layers import SAGEAggregator
 10 | from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
 11 |                             set_seed, tab_printer)
 12 | from torch.utils.data import DataLoader
 13 | from torch_geometric.datasets import Flickr, Reddit
 14 | from torch_geometric.utils import to_scipy_sparse_matrix
 15 | from tqdm import tqdm
 16 | 
 17 | 
 18 | class SpikeNet(nn.Module):
 19 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, T=5,
 20 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 21 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):
 22 | 
 23 |         super().__init__()
 24 | 
 25 |         tau = 1.0
 26 |         if sampler == 'rw':
 27 |             self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
 28 |         elif sampler == 'sage':
 29 |             self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
 30 |         else:
 31 |             raise ValueError(sampler)
 32 | 
 33 |         del data.edge_index
 34 | 
 35 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 36 | 
 37 |         for hid in hids:
 38 |             aggregators.append(SAGEAggregator(in_features, hid,
 39 |                                               concat=concat, bias=bias,
 40 |                                               aggr=aggr))
 41 | 
 42 |             if act == "IF":
 43 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
 44 |             elif act == 'LIF':
 45 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
 46 |             elif act == 'PLIF':
 47 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
 48 |             else:
 49 |                 raise ValueError(act)
 50 | 
 51 |             in_features = hid * 2 if concat else hid
 52 | 
 53 |         self.aggregators = aggregators
 54 |         self.dropout = nn.Dropout(dropout)
 55 |         self.snn = snn
 56 |         self.sizes = sizes
 57 |         self.T = T
 58 |         self.pooling = nn.Linear(T * in_features, out_features)
 59 | 
 60 |     def encode(self, nodes):
 61 |         spikes = []
 62 |         sizes = self.sizes
 63 |         x = data.x
 64 | 
 65 |         for time_step in range(self.T):
 66 |             h = [x[nodes].to(device)]
 67 |             num_nodes = [nodes.size(0)]
 68 |             nbr = nodes
 69 |             for size in sizes:
 70 |                 nbr = self.sampler(nbr, size)
 71 |                 num_nodes.append(nbr.size(0))
 72 |                 h.append(x[nbr].to(device))
 73 | 
 74 |             for i, aggregator in enumerate(self.aggregators):
 75 |                 self_x = h[:-1]
 76 |                 neigh_x = []
 77 |                 for j, n_x in enumerate(h[1:]):
 78 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
 79 | 
 80 |                 out = self.snn[i](aggregator(self_x, neigh_x))
 81 |                 if i != len(sizes) - 1:
 82 |                     out = self.dropout(out)
 83 |                     h = torch.split(out, num_nodes[:-(i + 1)])
 84 | 
 85 |             spikes.append(out)
 86 |         spikes = torch.cat(spikes, dim=1)
 87 |         neuron.reset_net(self)
 88 |         return spikes
 89 | 
 90 |     def forward(self, nodes):
 91 |         spikes = self.encode(nodes)
 92 |         return self.pooling(spikes)
 93 | 
 94 | 
 95 | parser = argparse.ArgumentParser()
 96 | parser.add_argument("--dataset", nargs="?", default="flickr",
 97 |                     help="Datasets (Reddit and Flickr only). (default: Flickr)")
 98 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2],
 99 |                     help='Neighborhood sampling size for each layer. (default: [5, 2])')
100 | parser.add_argument('--hids', type=int, nargs='+',
101 |                     default=[512, 10], help='Hidden units for each layer. (default: [128, 10])')
102 | parser.add_argument("--aggr", nargs="?", default="mean",
103 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
104 | parser.add_argument("--sampler", nargs="?", default="sage",
105 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
106 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
107 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
108 | parser.add_argument("--neuron", nargs="?", default="LIF",
109 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
110 | parser.add_argument('--batch_size', type=int, default=2048,
111 |                     help='Batch size for training. (default: 1024)')
112 | parser.add_argument('--lr', type=float, default=5e-3,
113 |                     help='Learning rate for training. (default: 5e-3)')
114 | parser.add_argument('--alpha', type=float, default=1.0,
115 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
116 | parser.add_argument('--T', type=int, default=15,
117 |                     help='Number of time steps. (default: 15)')
118 | parser.add_argument('--dropout', type=float, default=0.5,
119 |                     help='Dropout probability. (default: 0.5)')
120 | parser.add_argument('--epochs', type=int, default=100,
121 |                     help='Number of training epochs. (default: 100)')
122 | parser.add_argument('--concat', action='store_true',
123 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
124 | parser.add_argument('--seed', type=int, default=2022,
125 |                     help='Random seed for model. (default: 2022)')
126 | 
127 | 
128 | try:
129 |     args = parser.parse_args()
130 |     args.split_seed = 42
131 |     tab_printer(args)
132 | except:
133 |     parser.print_help()
134 |     exit(0)
135 | 
136 | assert len(args.hids) == len(args.sizes), "must be equal!"
137 | 
138 | root = "data/"  # Specify your root path
139 | 
140 | if args.dataset.lower() == "reddit":
141 |     dataset = Reddit(osp.join(root, 'Reddit'))
142 |     data = dataset[0]
143 | elif args.dataset.lower() == "flickr":
144 |     dataset = Flickr(osp.join(root, 'Flickr'))
145 |     data = dataset[0]
146 |     
147 | data.x = torch.nn.functional.normalize(data.x, dim=1)
148 | 
149 | set_seed(args.seed)
150 | 
151 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
152 | 
153 | y = data.y.to(device)
154 | 
155 | train_loader = DataLoader(data.train_mask.nonzero().view(-1), pin_memory=False, batch_size=args.batch_size, shuffle=True)
156 | val_loader = DataLoader(data.val_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
157 | test_loader = DataLoader(data.test_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
158 | 
159 | 
160 | model = SpikeNet(dataset.num_features, dataset.num_classes, alpha=args.alpha,
161 |                  dropout=args.dropout, sampler=args.sampler, T=args.T,
162 |                  aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
163 |                  hids=args.hids, act=args.neuron, bias=True).to(device)
164 | 
165 | optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
166 | loss_fn = nn.CrossEntropyLoss()
167 | 
168 | 
169 | def train():
170 |     model.train()
171 |     for nodes in tqdm(train_loader, desc='Training'):
172 |         optimizer.zero_grad()
173 |         loss_fn(model(nodes), y[nodes]).backward()
174 |         optimizer.step()
175 | 
176 | 
177 | @torch.no_grad()
178 | def test(loader):
179 |     model.eval()
180 |     logits = []
181 |     labels = []
182 |     for nodes in loader:
183 |         logits.append(model(nodes))
184 |         labels.append(y[nodes])
185 |     logits = torch.cat(logits, dim=0).cpu()
186 |     labels = torch.cat(labels, dim=0).cpu()
187 |     logits = logits.argmax(1)
188 |     metric_macro = metrics.f1_score(labels, logits, average='macro')
189 |     metric_micro = metrics.f1_score(labels, logits, average='micro')
190 |     return metric_macro, metric_micro
191 | 
192 | 
193 | best_val_metric = test_metric = 0
194 | start = time.time()
195 | for epoch in range(1, args.epochs + 1):
196 |     train()
197 |     val_metric, test_metric = test(val_loader), test(test_loader)
198 |     if val_metric[1] > best_val_metric:
199 |         best_val_metric = val_metric[1]
200 |         best_test_metric = test_metric
201 |     end = time.time()
202 |     print(
203 |         f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
204 | 
205 | # save bianry node embeddings (spikes)
206 | # emb = model.encode(torch.arange(data.num_nodes)).cpu()
207 | # torch.save(emb, 'emb.pth')
```

## File: F:\SomeProjects\CSGNN\setup.py

- Extension: .py
- Language: python
- Size: 327 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | from setuptools import setup
 2 | from torch.utils.cpp_extension import BuildExtension, CppExtension
 3 | 
 4 | setup(
 5 |     name="sample_neighber",
 6 |     ext_modules=[
 7 |         CppExtension("sample_neighber", sources=["spikenet/sample_neighber.cpp"], extra_compile_args=['-g']),
 8 | 
 9 |     ],
10 |     cmdclass={
11 |         "build_ext": BuildExtension
12 |     }
13 | )
```

## File: F:\SomeProjects\CSGNN\spikenet\dataset.py

- Extension: .py
- Language: python
- Size: 11862 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2025-09-14 23:11:56

### Code

```python
  1 | import math
  2 | import os.path as osp
  3 | from collections import defaultdict, namedtuple
  4 | from typing import Optional
  5 | 
  6 | import numpy as np
  7 | import scipy.sparse as sp
  8 | import torch
  9 | from sklearn import preprocessing
 10 | from sklearn.model_selection import train_test_split
 11 | from sklearn.preprocessing import LabelEncoder
 12 | from tqdm import tqdm
 13 | 
 14 | Data = namedtuple('Data', ['x', 'edge_index'])
 15 | 
 16 | 
 17 | def standard_normalization(arr):
 18 |     n_steps, n_node, n_dim = arr.shape
 19 |     arr_norm = preprocessing.scale(np.reshape(arr, [n_steps, n_node * n_dim]), axis=1)
 20 |     arr_norm = np.reshape(arr_norm, [n_steps, n_node, n_dim])
 21 |     return arr_norm
 22 | 
 23 | 
 24 | def edges_to_adj(edges, num_nodes, undirected=True):
 25 |     row, col = edges
 26 |     data = np.ones(len(row))
 27 |     N = num_nodes
 28 |     adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
 29 |     if undirected:
 30 |         adj = adj.maximum(adj.T)
 31 |     adj[adj > 1] = 1
 32 |     return adj
 33 | 
 34 | 
 35 | class Dataset:
 36 |     def __init__(self, name=None, root="./data"):
 37 |         self.name = name
 38 |         self.root = root
 39 |         self.x = None
 40 |         self.y = None
 41 |         self.num_features = None
 42 |         self.adj = []
 43 |         self.adj_evolve = []
 44 |         self.edges = []
 45 |         self.edges_evolve = []
 46 | 
 47 |     def _read_feature(self):
 48 |         filename = osp.join(self.root, self.name, f"{self.name}.npy")
 49 |         if osp.exists(filename):
 50 |             return np.load(filename)
 51 |         else:
 52 |             return None
 53 | 
 54 |     def split_nodes(
 55 |         self,
 56 |         train_size: float = 0.4,
 57 |         val_size: float = 0.0,
 58 |         test_size: float = 0.6,
 59 |         random_state: Optional[int] = None,
 60 |     ):
 61 |         val_size = 0. if val_size is None else val_size
 62 |         assert train_size + val_size + test_size <= 1.0
 63 | 
 64 |         y = self.y
 65 |         train_nodes, test_nodes = train_test_split(
 66 |             torch.arange(y.size(0)),
 67 |             train_size=train_size + val_size,
 68 |             test_size=test_size,
 69 |             random_state=random_state,
 70 |             stratify=y)
 71 | 
 72 |         if val_size:
 73 |             train_nodes, val_nodes = train_test_split(
 74 |                 train_nodes,
 75 |                 train_size=train_size / (train_size + val_size),
 76 |                 random_state=random_state,
 77 |                 stratify=y[train_nodes])
 78 |         else:
 79 |             val_nodes = None
 80 | 
 81 |         self.train_nodes = train_nodes
 82 |         self.val_nodes = val_nodes
 83 |         self.test_nodes = test_nodes
 84 | 
 85 |     def split_edges(
 86 |         self,
 87 |         train_stamp: float = 0.7,
 88 |         train_size: float = None,
 89 |         val_size: float = 0.1,
 90 |         test_size: float = 0.2,
 91 |         random_state: int = None,
 92 |     ):
 93 | 
 94 |         if random_state is not None:
 95 |             torch.manual_seed(random_state)
 96 | 
 97 |         num_edges = self.edges[-1].size(-1)
 98 |         train_stamp = train_stamp if train_stamp >= 1 else math.ceil(len(self) * train_stamp)
 99 | 
100 |         train_edges = torch.LongTensor(np.hstack(self.edges_evolve[:train_stamp]))
101 |         if train_size is not None:
102 |             assert 0 < train_size < 1
103 |             num_train = math.floor(train_size * num_edges)
104 |             perm = torch.randperm(train_edges.size(1))[:num_train]
105 |             train_edges = train_edges[:, perm]
106 | 
107 |         num_val = math.floor(val_size * num_edges)
108 |         num_test = math.floor(test_size * num_edges)
109 |         testing_edges = torch.LongTensor(np.hstack(self.edges_evolve[train_stamp:]))
110 |         perm = torch.randperm(testing_edges.size(1))
111 | 
112 |         assert num_val + num_test <= testing_edges.size(1)
113 | 
114 |         self.train_stamp = train_stamp
115 |         self.train_edges = train_edges
116 |         self.val_edges = testing_edges[:, perm[:num_val]]
117 |         self.test_edges = testing_edges[:, perm[num_val:num_val + num_test]]
118 | 
119 |     def __getitem__(self, time_index: int):
120 |         x = self.x[time_index]
121 |         edge_index = self.edges[time_index]
122 |         snapshot = Data(x=x, edge_index=edge_index)
123 |         return snapshot
124 | 
125 |     def __next__(self):
126 |         if self.t < len(self):
127 |             snapshot = self.__getitem__(self.t)
128 |             self.t = self.t + 1
129 |             return snapshot
130 |         else:
131 |             self.t = 0
132 |             raise StopIteration
133 | 
134 |     def __iter__(self):
135 |         self.t = 0
136 |         return self
137 | 
138 |     def __len__(self):
139 |         return len(self.adj)
140 | 
141 |     def __repr__(self):
142 |         return self.name
143 | 
144 | 
145 | class DBLP(Dataset):
146 |     def __init__(self, root="./data", normalize=True):
147 |         super().__init__(name='dblp', root=root)
148 |         edges_evolve, self.num_nodes = self._read_graph()
149 |         x = self._read_feature()
150 |         y = self._read_label()
151 | 
152 |         if x is not None:
153 |             if normalize:
154 |                 x = standard_normalization(x)
155 |             self.num_features = x.shape[-1]
156 |             self.x = torch.FloatTensor(x)
157 | 
158 |         self.num_classes = y.max() + 1
159 | 
160 |         edges = [edges_evolve[0]]
161 |         for e_now in edges_evolve[1:]:
162 |             e_last = edges[-1]
163 |             edges.append(np.hstack([e_last, e_now]))
164 | 
165 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
166 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
167 |         self.edges = [torch.LongTensor(edge) for edge in edges]
168 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
169 | 
170 |         self.y = torch.LongTensor(y)
171 | 
172 |     def _read_graph(self):
173 |         filename = osp.join(self.root, self.name, f"{self.name}.txt")
174 |         d = defaultdict(list)
175 |         N = 0
176 |         with open(filename) as f:
177 |             for line in f:
178 |                 x, y, t = line.strip().split()
179 |                 x, y = int(x), int(y)
180 |                 d[t].append((x, y))
181 |                 N = max(N, x)
182 |                 N = max(N, y)
183 |         N += 1
184 |         edges = []
185 |         for time in sorted(d):
186 |             row, col = zip(*d[time])
187 |             edge_now = np.vstack([row, col])
188 |             edges.append(edge_now)
189 |         return edges, N
190 | 
191 |     def _read_label(self):
192 |         filename = osp.join(self.root, self.name, "node2label.txt")
193 |         nodes = []
194 |         labels = []
195 |         with open(filename) as f:
196 |             for line in f:
197 |                 node, label = line.strip().split()
198 |                 nodes.append(int(node))
199 |                 labels.append(label)
200 | 
201 |         nodes = np.array(nodes)
202 |         labels = LabelEncoder().fit_transform(labels)
203 | 
204 |         assert np.allclose(nodes, np.arange(nodes.size))
205 |         return labels
206 | 
207 | 
208 | def merge(edges, step=1):
209 |     if step == 1:
210 |         return edges
211 |     i = 0
212 |     length = len(edges)
213 |     out = []
214 |     while i < length:
215 |         e = edges[i:i + step]
216 |         if len(e):
217 |             out.append(np.hstack(e))
218 |         i += step
219 |     print(f'Edges has been merged from {len(edges)} timestamps to {len(out)} timestamps')
220 |     return out
221 | 
222 | 
223 | class Tmall(Dataset):
224 |     def __init__(self, root="./data", normalize=True):
225 |         super().__init__(name='tmall', root=root)
226 |         edges_evolve, self.num_nodes = self._read_graph()
227 |         x = self._read_feature()
228 | 
229 |         y, labeled_nodes = self._read_label()
230 |         # reindexing
231 |         others = set(range(self.num_nodes)) - set(labeled_nodes.tolist())
232 |         new_index = np.hstack([labeled_nodes, list(others)])
233 |         whole_nodes = np.arange(self.num_nodes)
234 |         mapping_dict = dict(zip(new_index, whole_nodes))
235 |         mapping = np.vectorize(mapping_dict.get)(whole_nodes)
236 |         edges_evolve = [mapping[e] for e in edges_evolve]
237 | 
238 |         edges_evolve = merge(edges_evolve, step=10)
239 | 
240 |         if x is not None:
241 |             if normalize:
242 |                 x = standard_normalization(x)
243 |             self.num_features = x.shape[-1]
244 |             self.x = torch.FloatTensor(x)
245 | 
246 |         self.num_classes = y.max() + 1
247 | 
248 |         edges = [edges_evolve[0]]
249 |         for e_now in edges_evolve[1:]:
250 |             e_last = edges[-1]
251 |             edges.append(np.hstack([e_last, e_now]))
252 | 
253 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
254 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
255 |         self.edges = [torch.LongTensor(edge) for edge in edges]
256 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
257 | 
258 |         self.mapping = mapping
259 |         self.y = torch.LongTensor(y)
260 | 
261 |     def _read_graph(self):
262 |         filename = osp.join(self.root, self.name, f"{self.name}.txt")
263 |         d = defaultdict(list)
264 |         N = 0
265 |         with open(filename) as f:
266 |             for line in tqdm(f, desc='loading edges'):
267 |                 x, y, t = line.strip().split()
268 |                 x, y = int(x), int(y)
269 |                 d[t].append((x, y))
270 |                 N = max(N, x)
271 |                 N = max(N, y)
272 |         N += 1
273 |         edges = []
274 |         for time in sorted(d):
275 |             row, col = zip(*d[time])
276 |             edge_now = np.vstack([row, col])
277 |             edges.append(edge_now)
278 |         return edges, N
279 | 
280 |     def _read_label(self):
281 |         filename = osp.join(self.root, self.name, "node2label.txt")
282 |         nodes = []
283 |         labels = []
284 |         with open(filename) as f:
285 |             for line in tqdm(f, desc='loading nodes'):
286 |                 node, label = line.strip().split()
287 |                 nodes.append(int(node))
288 |                 labels.append(label)
289 | 
290 |         labeled_nodes = np.array(nodes)
291 |         labels = LabelEncoder().fit_transform(labels)
292 |         return labels, labeled_nodes
293 | 
294 | 
295 | class Patent(Dataset):
296 |     def __init__(self, root="./data", normalize=True):
297 |         super().__init__(name='patent', root=root)
298 |         edges_evolve = self._read_graph()
299 |         y = self._read_label()
300 |         edges_evolve = merge(edges_evolve, step=2)
301 |         x = self._read_feature()
302 | 
303 |         if x is not None:
304 |             if normalize:
305 |                 x = standard_normalization(x)
306 |             self.num_features = x.shape[-1]
307 |             self.x = torch.FloatTensor(x)
308 | 
309 |         self.num_nodes = y.size
310 |         self.num_features = x.shape[-1]
311 |         self.num_classes = y.max() + 1
312 | 
313 |         edges = [edges_evolve[0]]
314 |         for e_now in edges_evolve[1:]:
315 |             e_last = edges[-1]
316 |             edges.append(np.hstack([e_last, e_now]))
317 | 
318 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
319 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
320 |         self.edges = [torch.LongTensor(edge) for edge in edges]
321 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
322 | 
323 |         self.x = torch.FloatTensor(x)
324 |         self.y = torch.LongTensor(y)
325 | 
326 |     def _read_graph(self):
327 |         filename = osp.join(self.root, self.name, f"{self.name}_edges.json")
328 |         time_edges = defaultdict(list)
329 |         with open(filename) as f:
330 |             for line in tqdm(f, desc='loading patent_edges'):
331 |                 # src nodeID, dst nodeID, date, src originalID, dst originalID
332 |                 src, dst, date, _, _ = eval(line)
333 |                 date = date // 1e4
334 |                 time_edges[date].append((src, dst))
335 | 
336 |         edges = []
337 |         for time in sorted(time_edges):
338 |             edges.append(np.transpose(time_edges[time]))
339 |         return edges
340 | 
341 |     def _read_label(self):
342 |         filename = osp.join(self.root, self.name, f"{self.name}_nodes.json")
343 |         labels = []
344 |         with open(filename) as f:
345 |             for line in tqdm(f, desc='loading patent_nodes'):
346 |                 # nodeID, originalID, date, node class
347 |                 node, _, date, label = eval(line)
348 |                 date = date // 1e4
349 |                 labels.append(label - 1)
350 |         labels = np.array(labels)
351 |         return labels
```

## File: F:\SomeProjects\CSGNN\spikenet\deepwalk.py

- Extension: .py
- Language: python
- Size: 5290 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
  1 | from distutils.version import LooseVersion
  2 | 
  3 | import gensim
  4 | import numpy as np
  5 | import scipy.sparse as sp
  6 | from gensim.models import Word2Vec as _Word2Vec
  7 | from numba import njit
  8 | from sklearn import preprocessing
  9 | 
 10 | 
 11 | class DeepWalk:
 12 |     r"""Implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
 13 |     from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
 14 |     The procedure uses random walks to approximate the pointwise mutual information
 15 |     matrix obtained by pooling normalized adjacency matrix powers. This matrix
 16 |     is decomposed by an approximate factorization technique.
 17 |     """
 18 | 
 19 |     def __init__(self, dimensions: int = 64,
 20 |                  walk_length: int = 80,
 21 |                  walk_number: int = 10,
 22 |                  workers: int = 3,
 23 |                  window_size: int = 5,
 24 |                  epochs: int = 1,
 25 |                  learning_rate: float = 0.025,
 26 |                  negative: int = 1,
 27 |                  name: str = None,
 28 |                  seed: int = None):
 29 | 
 30 |         kwargs = locals()
 31 |         kwargs.pop("self")
 32 |         kwargs.pop("__class__", None)
 33 | 
 34 |         self.set_hyparas(kwargs)
 35 | 
 36 |     def set_hyparas(self, kwargs: dict):
 37 |         for k, v in kwargs.items():
 38 |             setattr(self, k, v)
 39 |         self.hyparas = kwargs
 40 | 
 41 |     def fit(self, graph: sp.csr_matrix):
 42 |         walks = RandomWalker(walk_length=self.walk_length,
 43 |                              walk_number=self.walk_number).walk(graph)
 44 |         sentences = [list(map(str, walk)) for walk in walks]
 45 |         model = Word2Vec(sentences,
 46 |                          sg=1,
 47 |                          hs=0,
 48 |                          alpha=self.learning_rate,
 49 |                          iter=self.epochs,
 50 |                          size=self.dimensions,
 51 |                          window=self.window_size,
 52 |                          workers=self.workers,
 53 |                          negative=self.negative,
 54 |                          seed=self.seed)
 55 |         self._embedding = model.get_embedding()
 56 | 
 57 |     def get_embedding(self, normalize=True) -> np.array:
 58 |         """Getting the node embedding."""
 59 |         embedding = self._embedding
 60 |         if normalize:
 61 |             embedding = preprocessing.normalize(embedding)
 62 |         return embedding
 63 | 
 64 | 
 65 | class RandomWalker:
 66 |     """Fast first-order random walks in DeepWalk
 67 | 
 68 |     Parameters:
 69 |     -----------
 70 |     walk_number (int): Number of random walks. Default is 10.
 71 |     walk_length (int): Length of random walks. Default is 80.
 72 |     """
 73 | 
 74 |     def __init__(self, walk_length: int = 80, walk_number: int = 10):
 75 |         self.walk_length = walk_length
 76 |         self.walk_number = walk_number
 77 | 
 78 |     def walk(self, graph: sp.csr_matrix):
 79 |         walks = self.random_walk(graph.indices,
 80 |                                  graph.indptr,
 81 |                                  walk_length=self.walk_length,
 82 |                                  walk_number=self.walk_number)
 83 |         return walks
 84 | 
 85 |     @staticmethod
 86 |     @njit(nogil=True)
 87 |     def random_walk(indices,
 88 |                     indptr,
 89 |                     walk_length,
 90 |                     walk_number):
 91 |         N = len(indptr) - 1
 92 |         for _ in range(walk_number):
 93 |             for n in range(N):
 94 |                 walk = [n]
 95 |                 current_node = n
 96 |                 for _ in range(walk_length - 1):
 97 |                     neighbors = indices[
 98 |                         indptr[current_node]:indptr[current_node + 1]]
 99 |                     if neighbors.size == 0:
100 |                         break
101 |                     current_node = np.random.choice(neighbors)
102 |                     walk.append(current_node)
103 | 
104 |                 yield walk
105 | 
106 | 
107 | class Word2Vec(_Word2Vec):
108 |     """A compatible version of Word2Vec"""
109 | 
110 |     def __init__(self, sentences=None, sg=0, hs=0, alpha=0.025, iter=5, size=100, window=5, workers=3, negative=5, seed=None, **kwargs):
111 |         if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
112 |             super().__init__(sentences,
113 |                              size=size,
114 |                              window=window,
115 |                              min_count=0,
116 |                              alpha=alpha,
117 |                              sg=sg,
118 |                              workers=workers,
119 |                              iter=iter,
120 |                              negative=negative,
121 |                              hs=hs,
122 |                              compute_loss=True,
123 |                              seed=seed, **kwargs)
124 | 
125 |         else:
126 |             super().__init__(sentences,
127 |                              vector_size=size,
128 |                              window=window,
129 |                              min_count=0,
130 |                              alpha=alpha,
131 |                              sg=sg,
132 |                              workers=workers,
133 |                              epochs=iter,
134 |                              negative=negative,
135 |                              hs=hs,
136 |                              compute_loss=True,
137 |                              seed=seed, **kwargs)
138 | 
139 |     def get_embedding(self):
140 |         if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
141 |             embedding = self.wv.vectors[np.fromiter(
142 |                 map(int, self.wv.index2word), np.int32).argsort()]
143 |         else:
144 |             embedding = self.wv.vectors[np.fromiter(
145 |                 map(int, self.wv.index_to_key), np.int32).argsort()]
146 | 
147 |         return embedding
```

## File: F:\SomeProjects\CSGNN\spikenet\layers.py

- Extension: .py
- Language: python
- Size: 1225 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | import torch
 2 | import torch.nn as nn
 3 | 
 4 | 
 5 | class SAGEAggregator(nn.Module):
 6 |     def __init__(self, in_features, out_features,
 7 |                  aggr='mean',
 8 |                  concat=False,
 9 |                  bias=False):
10 | 
11 |         super().__init__()
12 |         self.in_features = in_features
13 |         self.out_features = out_features
14 |         self.concat = concat
15 | 
16 |         self.aggr = aggr
17 |         self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]
18 | 
19 |         self.lin_l = nn.Linear(in_features, out_features, bias=bias)
20 |         self.lin_r = nn.Linear(in_features, out_features, bias=bias)
21 | 
22 |     def forward(self, x, neigh_x):
23 |         if not isinstance(x, torch.Tensor):
24 |             x = torch.cat(x, dim=0)
25 | 
26 |         if not isinstance(neigh_x, torch.Tensor):
27 |             neigh_x = torch.cat([self.aggregator(h, dim=1)
28 |                                 for h in neigh_x], dim=0)
29 |         else:
30 |             neigh_x = self.aggregator(neigh_x, dim=1)
31 | 
32 |         x = self.lin_l(x)
33 |         neigh_x = self.lin_r(neigh_x)
34 |         out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
35 |         return out
36 | 
37 |     def __repr__(self):
38 |         return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"
```

## File: F:\SomeProjects\CSGNN\spikenet\neuron.py

- Extension: .py
- Language: python
- Size: 7039 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
  1 | from math import pi
  2 | 
  3 | import torch
  4 | import torch.nn as nn
  5 | 
  6 | gamma = 0.2
  7 | thresh_decay = 0.7
  8 | 
  9 | 
 10 | def reset_net(net: nn.Module):
 11 |     for m in net.modules():
 12 |         if hasattr(m, 'reset'):
 13 |             m.reset()
 14 | 
 15 | 
 16 | def heaviside(x: torch.Tensor):
 17 |     return x.ge(0)
 18 | 
 19 | 
 20 | def gaussian(x, mu, sigma):
 21 |     """
 22 |     Gaussian PDF with broadcasting.
 23 |     """
 24 |     return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))
 25 | 
 26 | 
 27 | class BaseSpike(torch.autograd.Function):
 28 |     """
 29 |     Baseline spiking function.
 30 |     """
 31 | 
 32 |     @staticmethod
 33 |     def forward(ctx, x, alpha):
 34 |         ctx.save_for_backward(x, alpha)
 35 |         return x.gt(0).float()
 36 | 
 37 |     @staticmethod
 38 |     def backward(ctx, grad_output):
 39 |         raise NotImplementedError
 40 | 
 41 | 
 42 | class SuperSpike(BaseSpike):
 43 |     """
 44 |     Spike function with SuperSpike surrogate gradient from
 45 |     "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.
 46 | 
 47 |     Design choices:
 48 |     - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
 49 |     - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
 50 |     """
 51 | 
 52 |     @staticmethod
 53 |     def backward(ctx, grad_output):
 54 |         x, alpha = ctx.saved_tensors
 55 |         grad_input = grad_output.clone()
 56 |         sg = 1 / (1 + alpha * x.abs()) ** 2
 57 |         return grad_input * sg, None
 58 | 
 59 | 
 60 | class MultiGaussSpike(BaseSpike):
 61 |     """
 62 |     Spike function with multi-Gaussian surrogate gradient from
 63 |     "Accurate and efficient time-domain classification...", Yin et al. 2021.
 64 | 
 65 |     Design choices:
 66 |     - Hyperparameters determined through grid search (Yin et al. 2021)
 67 |     """
 68 | 
 69 |     @staticmethod
 70 |     def backward(ctx, grad_output):
 71 |         x, alpha = ctx.saved_tensors
 72 |         grad_input = grad_output.clone()
 73 |         zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
 74 |         sg = (
 75 |             1.15 * gaussian(x, zero, alpha)
 76 |             - 0.15 * gaussian(x, alpha, 6 * alpha)
 77 |             - 0.15 * gaussian(x, -alpha, 6 * alpha)
 78 |         )
 79 |         return grad_input * sg, None
 80 | 
 81 | 
 82 | class TriangleSpike(BaseSpike):
 83 |     """
 84 |     Spike function with triangular surrogate gradient
 85 |     as in Bellec et al. 2020.
 86 |     """
 87 | 
 88 |     @staticmethod
 89 |     def backward(ctx, grad_output):
 90 |         x, alpha = ctx.saved_tensors
 91 |         grad_input = grad_output.clone()
 92 |         sg = torch.nn.functional.relu(1 - alpha * x.abs())
 93 |         return grad_input * sg, None
 94 | 
 95 | 
 96 | class ArctanSpike(BaseSpike):
 97 |     """
 98 |     Spike function with derivative of arctan surrogate gradient.
 99 |     Featured in Fang et al. 2020/2021.
100 |     """
101 | 
102 |     @staticmethod
103 |     def backward(ctx, grad_output):
104 |         x, alpha = ctx.saved_tensors
105 |         grad_input = grad_output.clone()
106 |         sg = 1 / (1 + alpha * x * x)
107 |         return grad_input * sg, None
108 | 
109 | 
110 | class SigmoidSpike(BaseSpike):
111 | 
112 |     @staticmethod
113 |     def backward(ctx, grad_output):
114 |         x, alpha = ctx.saved_tensors
115 |         grad_input = grad_output.clone()
116 |         sgax = (x * alpha).sigmoid_()
117 |         sg = (1. - sgax) * sgax * alpha
118 |         return grad_input * sg, None
119 | 
120 | 
121 | def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
122 |     return SuperSpike.apply(x - thresh, alpha)
123 | 
124 | 
125 | def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
126 |     return MultiGaussSpike.apply(x - thresh, alpha)
127 | 
128 | 
129 | def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
130 |     return SigmoidSpike.apply(x - thresh, alpha)
131 | 
132 | 
133 | def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
134 |     return TriangleSpike.apply(x - thresh, alpha)
135 | 
136 | 
137 | def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
138 |     return ArctanSpike.apply(x - thresh, alpha)
139 | 
140 | 
141 | SURROGATE = {'sigmoid': sigmoidspike, 'triangle': trianglespike, 'arctan': arctanspike,
142 |              'mg': mgspike, 'super': superspike}
143 | 
144 | 
145 | class IF(nn.Module):
146 |     def __init__(self, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
147 |         super().__init__()
148 |         self.v_threshold = v_threshold
149 |         self.v_reset = v_reset
150 |         self.surrogate = SURROGATE.get(surrogate)
151 |         self.register_buffer("alpha", torch.as_tensor(
152 |             alpha, dtype=torch.float32))
153 |         self.reset()
154 | 
155 |     def reset(self):
156 |         self.v = 0.
157 |         self.v_th = self.v_threshold
158 | 
159 |     def forward(self, dv):
160 |         # 1. charge
161 |         self.v += dv
162 |         # 2. fire
163 |         spike = self.surrogate(self.v, self.v_threshold, self.alpha)
164 |         # 3. reset
165 |         self.v = (1 - spike) * self.v + spike * self.v_reset
166 |         # 4. threhold updates
167 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
168 |         self.v_th = gamma * spike + self.v_th * thresh_decay
169 |         return spike
170 | 
171 | 
172 | class LIF(nn.Module):
173 |     def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
174 |         super().__init__()
175 |         self.v_threshold = v_threshold
176 |         self.v_reset = v_reset
177 |         self.surrogate = SURROGATE.get(surrogate)
178 |         self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
179 |         self.register_buffer("alpha", torch.as_tensor(
180 |             alpha, dtype=torch.float32))
181 |         self.reset()
182 | 
183 |     def reset(self):
184 |         self.v = 0.
185 |         self.v_th = self.v_threshold
186 | 
187 |     def forward(self, dv):
188 |         # 1. charge
189 |         self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
190 |         # 2. fire
191 |         spike = self.surrogate(self.v, self.v_th, self.alpha)
192 |         # 3. reset
193 |         self.v = (1 - spike) * self.v + spike * self.v_reset
194 |         # 4. threhold updates
195 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
196 |         self.v_th = gamma * spike + self.v_th * thresh_decay
197 |         return spike
198 | 
199 | 
200 | class PLIF(nn.Module):
201 |     def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
202 |         super().__init__()
203 |         self.v_threshold = v_threshold
204 |         self.v_reset = v_reset
205 |         self.surrogate = SURROGATE.get(surrogate)
206 |         self.register_parameter("tau", nn.Parameter(
207 |             torch.as_tensor(tau, dtype=torch.float32)))
208 |         self.register_buffer("alpha", torch.as_tensor(
209 |             alpha, dtype=torch.float32))
210 |         self.reset()
211 | 
212 |     def reset(self):
213 |         self.v = 0.
214 |         self.v_th = self.v_threshold
215 | 
216 |     def forward(self, dv):
217 |         # 1. charge
218 |         self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
219 |         # 2. fire
220 |         spike = self.surrogate(self.v, self.v_th, self.alpha)
221 |         # 3. reset
222 |         self.v = (1 - spike) * self.v + spike * self.v_reset
223 |         # 4. threhold updates
224 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
225 |         self.v_th = gamma * spike + self.v_th * thresh_decay
226 |         return spike
```

## File: F:\SomeProjects\CSGNN\spikenet\sample_neighber.cpp

- Extension: .cpp
- Language: cpp
- Size: 5664 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```cpp
  1 | #include <torch/extension.h>
  2 | #define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
  3 | #define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
  4 | 
  5 | #define AT_DISPATCH_HAS_VALUE(optional_value, ...) \
  6 |     [&] {                                          \
  7 |         if (optional_value.has_value())            \
  8 |         {                                          \
  9 |             const bool HAS_VALUE = true;           \
 10 |             return __VA_ARGS__();                  \
 11 |         }                                          \
 12 |         else                                       \
 13 |         {                                          \
 14 |             const bool HAS_VALUE = false;          \
 15 |             return __VA_ARGS__();                  \
 16 |         }                                          \
 17 |     }()
 18 | 
 19 | torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
 20 |                int64_t num_neighbors, bool replace);
 21 | 
 22 | // Returns `rowptr`, `col`, `n_id`, `e_id`
 23 | torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
 24 |                int64_t num_neighbors, bool replace)
 25 | {
 26 |     CHECK_CPU(rowptr);
 27 |     CHECK_CPU(col);
 28 |     CHECK_CPU(idx);
 29 |     CHECK_INPUT(idx.dim() == 1);
 30 | 
 31 |     auto rowptr_data = rowptr.data_ptr<int64_t>();
 32 |     auto col_data = col.data_ptr<int64_t>();
 33 |     auto idx_data = idx.data_ptr<int64_t>();
 34 | 
 35 |     std::vector<int64_t> n_ids;
 36 | 
 37 |     int64_t i;
 38 |     
 39 | 
 40 |     int64_t n, c, e, row_start, row_end, row_count;
 41 | 
 42 |     if (num_neighbors < 0)
 43 |     { // No sampling ======================================
 44 | 
 45 |         for (int64_t i = 0; i < idx.numel(); i++)
 46 |         {
 47 |             n = idx_data[i];
 48 |             row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
 49 |             row_count = row_end - row_start;
 50 | 
 51 |             for (int64_t j = 0; j < row_count; j++)
 52 |             {
 53 |                 e = row_start + j;
 54 |                 c = col_data[e];
 55 |                 n_ids.push_back(c);
 56 |             }
 57 |         }
 58 |     }
 59 | 
 60 |     else if (replace)
 61 |     { // Sample with replacement ===============================
 62 |         for (int64_t i = 0; i < idx.numel(); i++)
 63 |         {
 64 |             n = idx_data[i];
 65 |             row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
 66 |             row_count = row_end - row_start;
 67 | 
 68 |             std::unordered_set<int64_t> perm;
 69 |             if (row_count <= num_neighbors)
 70 |             {
 71 |                 for (int64_t j = 0; j < row_count; j++)
 72 |                     perm.insert(j);
 73 |                 for (int64_t j = 0; j < num_neighbors-row_count; j++){
 74 |                     e = row_start + rand() % row_count;
 75 |                     c = col_data[e];
 76 |                     n_ids.push_back(c);
 77 |                 }
 78 |             }
 79 |             else
 80 |             { // See: https://www.nowherenearithaca.com/2013/05/
 81 |                 //      robert-floyds-tiny-and-beautiful.html
 82 |                 for (int64_t j = row_count - num_neighbors; j < row_count; j++)
 83 |                 {
 84 |                     if (!perm.insert(rand() % j).second)
 85 |                         perm.insert(j);
 86 |                 }
 87 |             }
 88 | 
 89 |             
 90 |             for (const int64_t &p : perm)
 91 |             {
 92 |                 e = row_start + p;
 93 |                 c = col_data[e];
 94 |                 n_ids.push_back(c);
 95 |             }
 96 |             
 97 |         }
 98 |         // for (int64_t i = 0; i < idx.numel(); i++)
 99 |         // {
100 |         //     n = idx_data[i];
101 |         //     row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
102 |         //     row_count = row_end - row_start;
103 |         //     // std::vector<int64_t>().swap(temp);
104 |         //     // for (int64_t j = 0; j < row_count; j++)
105 |         //     // {
106 |         //     //     temp.push_back(j);
107 |         //     // }
108 |         //     // if (row_count<num_neighbors){
109 |         //     //     for (int64_t j = 0; j <num_neighbors-row_count; j++){
110 |         //     //         temp.push_back(rand() % row_count);
111 |         //     //     }
112 |         //     // }
113 |         //     // std::random_shuffle(temp.begin(), temp.end());
114 |         //     std::unordered_set<int64_t> perm;
115 |         //     for (int64_t j = 0; j < num_neighbors; j++)
116 |         //     {
117 |         //         e = row_start + rand() % row_count;
118 |         //         // e = row_start + temp[j];
119 |         //         c = col_data[e];
120 |         //         n_ids.push_back(c);
121 |         //     }
122 |         // }
123 |     }
124 |     else
125 |     { // Sample without replacement via Robert Floyd algorithm ============
126 | 
127 |         for (int64_t i = 0; i < idx.numel(); i++)
128 |         {
129 |             n = idx_data[i];
130 |             row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
131 |             row_count = row_end - row_start;
132 | 
133 |             std::unordered_set<int64_t> perm;
134 |             if (row_count <= num_neighbors)
135 |             {
136 |                 for (int64_t j = 0; j < row_count; j++)
137 |                     perm.insert(j);
138 |             }
139 |             else
140 |             { // See: https://www.nowherenearithaca.com/2013/05/
141 |                 //      robert-floyds-tiny-and-beautiful.html
142 |                 for (int64_t j = row_count - num_neighbors; j < row_count; j++)
143 |                 {
144 |                     if (!perm.insert(rand() % j).second)
145 |                         perm.insert(j);
146 |                 }
147 |             }
148 | 
149 |             for (const int64_t &p : perm)
150 |             {
151 |                 e = row_start + p;
152 |                 c = col_data[e];
153 |                 n_ids.push_back(c);
154 |             }
155 |         }
156 |     }
157 | 
158 |     int64_t N = n_ids.size();
159 |     auto out_n_id = torch::from_blob(n_ids.data(), {N}, col.options()).clone();
160 | 
161 |     return out_n_id;
162 | }
163 | PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
164 |     m.def("sample_neighber_cpu", &sample_neighber_cpu, "Node neighborhood sampler");
165 | }
```

## File: F:\SomeProjects\CSGNN\spikenet\utils.py

- Extension: .py
- Language: python
- Size: 2759 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

### Code

```python
 1 | import numpy as np
 2 | import scipy.sparse as sp
 3 | import torch
 4 | from sample_neighber import sample_neighber_cpu
 5 | from texttable import Texttable
 6 | 
 7 | try:
 8 |     import torch_cluster
 9 | except ImportError:
10 |     torch_cluster = None
11 | 
12 | 
13 | def set_seed(seed):
14 |     np.random.seed(seed)
15 |     torch.manual_seed(seed)
16 |     torch.cuda.manual_seed(seed)
17 | 
18 | def tab_printer(args):
19 |     """Function to print the logs in a nice tabular format.
20 |     
21 |     Note
22 |     ----
23 |     Package `Texttable` is required.
24 |     Run `pip install Texttable` if was not installed.
25 |     
26 |     Parameters
27 |     ----------
28 |     args: Parameters used for the model.
29 |     """
30 |     args = vars(args)
31 |     keys = sorted(args.keys())
32 |     t = Texttable() 
33 |     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
34 |     print(t.draw())
35 | 
36 |     
37 | class Sampler:
38 |     def __init__(self, adj_matrix: sp.csr_matrix):
39 |         self.rowptr = torch.LongTensor(adj_matrix.indptr)
40 |         self.col = torch.LongTensor(adj_matrix.indices)
41 | 
42 |     def __call__(self, nodes, size, replace=True):
43 |         nbr = sample_neighber_cpu(self.rowptr, self.col, nodes, size, replace)
44 |         return nbr
45 |     
46 |     
47 | class RandomWalkSampler:
48 |     def __init__(self, adj_matrix: sp.csr_matrix, p: float = 1.0, q: float = 1.0):
49 |         self.rowptr = torch.LongTensor(adj_matrix.indptr)
50 |         self.col = torch.LongTensor(adj_matrix.indices)
51 |         self.p = p
52 |         self.q = q
53 |         assert torch_cluster, "Please install 'torch_cluster' first."
54 | 
55 |     def __call__(self, nodes, size, replace=True):
56 |         nbr = torch.ops.torch_cluster.random_walk(self.rowptr, self.col, nodes, size, self.p, self.q)[0][:, 1:] 
57 |         return nbr
58 | 
59 | 
60 | def eliminate_selfloops(adj_matrix):
61 |     """eliminate selfloops for adjacency matrix.
62 | 
63 |     >>>eliminate_selfloops(adj) # return an adjacency matrix without selfloops
64 | 
65 |     Parameters
66 |     ----------
67 |     adj_matrix: Scipy matrix or Numpy array
68 | 
69 |     Returns
70 |     -------
71 |     Single Scipy sparse matrix or Numpy matrix.
72 | 
73 |     """
74 |     if sp.issparse(adj_matrix):
75 |         adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
76 |         adj_matrix.eliminate_zeros()
77 |     else:
78 |         adj_matrix = adj_matrix - np.diag(adj_matrix)
79 |     return adj_matrix
80 | 
81 | 
82 | def add_selfloops(adj_matrix: sp.csr_matrix):
83 |     """add selfloops for adjacency matrix.
84 | 
85 |     >>>add_selfloops(adj) # return an adjacency matrix with selfloops
86 | 
87 |     Parameters
88 |     ----------
89 |     adj_matrix: Scipy matrix or Numpy array
90 | 
91 |     Returns
92 |     -------
93 |     Single sparse matrix or Numpy matrix.
94 | 
95 |     """
96 |     adj_matrix = eliminate_selfloops(adj_matrix)
97 | 
98 |     return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')
```

## File: F:\SomeProjects\CSGNN\spikenet_x\delayline.py

- Extension: .py
- Language: python
- Size: 4626 bytes
- Created: 2025-08-22 12:55:40
- Modified: 2025-08-22 12:56:05

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | LearnableDelayLine: 可学习多延迟通路（因果深度可分离 1D 卷积）
  4 | 
  5 | - 输入输出形状:
  6 |   forward(H: Float[T, N, d_in]) -> Float[T, N, d_in]
  7 | 
  8 | - 设计:
  9 |   对每个通道 c, 时间步 t:
 10 |       H_tilde[t, :, c] = sum_{k=0..K-1} w_c[k] * H[t-k, :, c]
 11 |   其中 w_c[k] = softplus(u_c[k]) * rho^k / sum_r softplus(u_c[r]) * rho^r
 12 |   并在 t-k < 0 使用因果左填充 0。
 13 | 
 14 | - 实现:
 15 |   将输入重排为 [N, d_in, T]，使用 groups=d_in 的 Conv1d 执行深度可分离因果卷积。
 16 | """
 17 | 
 18 | from __future__ import annotations
 19 | 
 20 | from typing import Optional
 21 | 
 22 | import torch
 23 | import torch.nn as nn
 24 | import torch.nn.functional as F
 25 | 
 26 | 
 27 | class LearnableDelayLine(nn.Module):
 28 |     def __init__(
 29 |         self,
 30 |         d_in: int,
 31 |         K: int = 5,
 32 |         rho: float = 0.85,
 33 |         per_channel: bool = True,
 34 |         eps: float = 1e-8,
 35 |     ) -> None:
 36 |         """
 37 |         参数
 38 |         ----
 39 |         d_in : 输入通道数（特征维）
 40 |         K : 延迟 tap 数 (>=1)
 41 |         rho : 指数折扣因子，(0,1)
 42 |         per_channel : True 表示逐通道独立权重；False 表示所有通道共享一组权重
 43 |         eps : 归一化时的数值稳定项
 44 |         """
 45 |         super().__init__()
 46 |         assert d_in >= 1, "d_in 必须 >= 1"
 47 |         assert K >= 1, "K 必须 >= 1"
 48 |         assert 0.0 < rho < 1.0, "rho 必须在 (0,1)"
 49 | 
 50 |         self.d_in = int(d_in)
 51 |         self.K = int(K)
 52 |         self.rho = float(rho)
 53 |         self.per_channel = bool(per_channel)
 54 |         self.eps = float(eps)
 55 | 
 56 |         # 原始可学习参数 u，经 softplus 后非负
 57 |         if self.per_channel:
 58 |             self.u = nn.Parameter(torch.zeros(self.d_in, self.K))
 59 |         else:
 60 |             self.u = nn.Parameter(torch.zeros(self.K))
 61 | 
 62 |         # 预先缓存 rho 的幂 [K]
 63 |         rho_pow = torch.tensor([self.rho ** k for k in range(self.K)], dtype=torch.float32)
 64 |         self.register_buffer("rho_pow", rho_pow, persistent=True)
 65 | 
 66 |     def extra_repr(self) -> str:
 67 |         return f"d_in={self.d_in}, K={self.K}, rho={self.rho}, per_channel={self.per_channel}"
 68 | 
 69 |     @torch.no_grad()
 70 |     def get_delay_weights(self) -> torch.Tensor:
 71 |         """
 72 |         返回当前归一化后的延迟权重 w，形状:
 73 |           - per_channel=True: [d_in, K]
 74 |           - per_channel=False: [K]
 75 |         便于监控/可视化。
 76 |         """
 77 |         if self.per_channel:
 78 |             # [d_in, K]
 79 |             sp = F.softplus(self.u)
 80 |             num = sp * self.rho_pow  # 广播到 [d_in, K]
 81 |             den = num.sum(dim=1, keepdim=True).clamp_min(self.eps)
 82 |             w = num / den
 83 |             return w
 84 |         else:
 85 |             # [K]
 86 |             sp = F.softplus(self.u)
 87 |             num = sp * self.rho_pow
 88 |             den = num.sum().clamp_min(self.eps)
 89 |             w = num / den
 90 |             return w
 91 | 
 92 |     def _build_depthwise_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
 93 |         """
 94 |         构造 Conv1d 的 depthwise 卷积核:
 95 |           - 形状 [d_in, 1, K]
 96 |           - 每个通道自有一条核（若 per_channel=True），否则共享一条核并复制
 97 |         """
 98 |         if self.per_channel:
 99 |             # [d_in, K] -> [d_in, 1, K]
100 |             w = self.get_delay_weights().to(device=device, dtype=dtype).unsqueeze(1)
101 |         else:
102 |             # 共享核 [K] -> [d_in, 1, K]
103 |             w_shared = self.get_delay_weights().to(device=device, dtype=dtype).view(1, 1, self.K)
104 |             w = w_shared.expand(self.d_in, 1, self.K).contiguous()
105 |         return w  # Float[d_in, 1, K]
106 | 
107 |     def forward(self, H: torch.Tensor) -> torch.Tensor:
108 |         """
109 |         参数
110 |         ----
111 |         H : Float[T, N, d_in]
112 | 
113 |         返回
114 |         ----
115 |         H_tilde : Float[T, N, d_in]
116 |         """
117 |         assert H.dim() == 3, "输入 H 形状应为 [T, N, d_in]"
118 |         T, N, Din = H.shape
119 |         assert Din == self.d_in, f"d_in 不匹配: 期望 {self.d_in}, 实得 {Din}"
120 | 
121 |         # [T, N, d] -> [N, d, T]
122 |         x = H.permute(1, 2, 0).contiguous()
123 | 
124 |         # 因果左填充 K-1
125 |         pad_left = self.K - 1
126 |         if pad_left > 0:
127 |             x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)  # 在时间维左侧填充
128 | 
129 |         # 深度可分离卷积 (groups=d_in)
130 |         weight = self._build_depthwise_kernel(H.device, H.dtype)  # [d, 1, K]
131 |         y = F.conv1d(x, weight=weight, bias=None, stride=1, padding=0, groups=self.d_in)
132 |         # y: [N, d, T]
133 | 
134 |         # 回到 [T, N, d]
135 |         H_tilde = y.permute(2, 0, 1).contiguous()
136 |         return H_tilde
```

## File: F:\SomeProjects\CSGNN\spikenet_x\lif_cell.py

- Extension: .py
- Language: python
- Size: 2766 bytes
- Created: 2025-08-22 12:57:19
- Modified: 2025-09-15 02:40:52

### Code

```python
 1 | import torch
 2 | import torch.nn as nn
 3 | from typing import Tuple
 4 | 
 5 | class LIFCell(nn.Module):
 6 |     """
 7 |     一个批处理的、基于循环的 Leaky Integrate-and-Fire (LIF) 神经元单元。
 8 |     它接收一个形状为 [T, N] 的输入电流，并按时间步进行处理。
 9 |     """
10 |     def __init__(self, lif_tau_theta: float = 1.0, lif_gamma: float = 0.95, lif_beta: float = 0.95):
11 |         """
12 |         初始化LIF神经元。
13 | 
14 |         Args:
15 |             lif_tau_theta (float): 膜电位阈值 (V_th)。
16 |             lif_gamma (float): 脉冲衰减因子，用于脉冲后的电位重置。
17 |             lif_beta (float): 膜电位泄漏/衰减因子。
18 |         """
19 |         super().__init__()
20 |         # 使用 register_buffer 将这些张量注册为模型的持久状态，但不是模型参数（不会被优化器更新）
21 |         self.register_buffer("tau_theta", torch.tensor(lif_tau_theta))
22 |         self.register_buffer("gamma", torch.tensor(lif_gamma))
23 |         self.register_buffer("beta", torch.tensor(lif_beta))
24 | 
25 |     def forward(self, I_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
26 |         """
27 |         LIF神经元的前向传播。
28 | 
29 |         Args:
30 |             I_in (torch.Tensor): 输入电流，形状为 [T, N]，T是时间步数，N是节点数。
31 | 
32 |         Returns:
33 |             Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
34 |             - spikes (torch.Tensor): 输出脉冲序列，形状 [T, N]。
35 |             - v_mem (torch.Tensor): 膜电位历史记录，形状 [T, N]。
36 |             - spike_history (torch.Tensor): 脉冲历史记录（与spikes相同，用于可能的内部调试）。
37 |         """
38 |         T, N = I_in.shape
39 |         device = I_in.device
40 | 
41 |         # 初始化膜电位和脉冲历史记录
42 |         v = torch.zeros(N, device=device)
43 |         s = torch.zeros(N, device=device)
44 | 
45 |         # 用于存储每个时间步结果的列表
46 |         v_mem_list = []
47 |         spike_list = []
48 | 
49 |         # 按时间步循环处理
50 |         for t in range(T):
51 |             # 计算新的膜电位
52 |             # v_new = v_old * beta (泄漏) + I_in (积分) - s_old * gamma (重置)
53 |             v = self.beta * v + I_in[t] - self.gamma * s
54 |             
55 |             # 检查是否超过阈值，产生脉冲
56 |             s = (v > self.tau_theta).float()
57 |             
58 |             # 脉冲发放后，重置膜电位 (硬重置)
59 |             v = v * (1.0 - s)
60 | 
61 |             # 存储当前时间步的结果
62 |             v_mem_list.append(v)
63 |             spike_list.append(s)
64 | 
65 |         # 将列表堆叠成张量
66 |         v_mem_out = torch.stack(v_mem_list, dim=0)
67 |         spikes_out = torch.stack(spike_list, dim=0)
68 |         
69 |         return spikes_out, v_mem_out, spikes_out
```

## File: F:\SomeProjects\CSGNN\spikenet_x\masked_ops.py

- Extension: .py
- Language: python
- Size: 4504 bytes
- Created: 2025-08-22 12:50:15
- Modified: 2025-09-14 19:29:19

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | 掩码与 Top-k 相关的张量操作（纯 PyTorch 实现）
  4 | 
  5 | 函数约定
  6 | --------
  7 | - 所有 logits/score 相关的函数在被掩蔽位置填充为 -inf（或非常负的数），
  8 |   再做 softmax，以确保数值与归一化正确。
  9 | - 所有张量形状均保持与输入一致，除非特别说明。
 10 | 
 11 | 作者: Cline
 12 | """
 13 | 
 14 | from typing import Optional, Tuple
 15 | 
 16 | import torch
 17 | import torch.nn.functional as F
 18 | 
 19 | 
 20 | NEG_INF = -1e30  # 作为 -inf 的数值近似，避免部分设备对 -inf 的不一致处理
 21 | 
 22 | 
 23 | def fill_masked_(logits: torch.Tensor, mask: torch.Tensor, value: float = NEG_INF) -> torch.Tensor:
 24 |     """
 25 |     原地将 mask==0 的位置填充为 value（默认近似 -inf）。
 26 | 
 27 |     参数
 28 |     ----
 29 |     logits : Float[...]
 30 |         任意形状的分数张量
 31 |     mask : Bool/Byte[...]
 32 |         与 logits 同形，True/1 表示可用，False/0 表示被掩蔽
 33 |     value : float
 34 |         被掩蔽位置写入的值
 35 | 
 36 |     返回
 37 |     ----
 38 |     logits : Float[...]
 39 |         与输入同一引用的张量（原地修改）
 40 |     """
 41 |     if mask.dtype != torch.bool:
 42 |         mask = mask != 0
 43 |     logits.masked_fill_(~mask, value)
 44 |     return logits
 45 | 
 46 | 
 47 | def masked_softmax(
 48 |     logits: torch.Tensor,
 49 |     mask: Optional[torch.Tensor] = None,
 50 |     dim: int = -1,
 51 |     temperature: float = 1.0,
 52 | ) -> torch.Tensor:
 53 |     """
 54 |     在给定维度上对带掩码的 logits 进行 softmax。
 55 | 
 56 |     - 先在掩蔽位置写入 -inf，再做 softmax，避免“先置零后归一”导致的数值偏差。
 57 |     - 支持温度缩放：logits / temperature
 58 | 
 59 |     参数
 60 |     ----
 61 |     logits : Float[...]
 62 |     mask : Bool/Byte[...] or None
 63 |         与 logits 广播兼容；为 None 时等价于全 True
 64 |     dim : int
 65 |     temperature : float
 66 | 
 67 |     返回
 68 |     ----
 69 |     probs : Float[...]
 70 |     """
 71 |     if temperature != 1.0:
 72 |         logits = logits / float(temperature)
 73 | 
 74 |     if mask is not None:
 75 |         # 为避免修改外部张量，做一份拷贝
 76 |         logits = logits.clone()
 77 |         fill_masked_(logits, mask, NEG_INF)
 78 | 
 79 |     # 数值稳定 softmax
 80 |     max_val, _ = torch.max(logits, dim=dim, keepdim=True)
 81 |     shifted = logits - max_val
 82 |     exp = torch.exp(shifted)
 83 |     if mask is not None:
 84 |         if mask.dtype != torch.bool:
 85 |             mask = mask != 0
 86 |         exp = exp * mask.to(dtype=exp.dtype)
 87 | 
 88 |     denom = torch.clamp(exp.sum(dim=dim, keepdim=True), min=1e-12)
 89 |     return exp / denom
 90 | 
 91 | 
 92 | def topk_mask_logits(logits: torch.Tensor, k: int, dim: int = -1, inplace: bool = False):
 93 |     """
 94 |     仅保留最后一维的 top-k 位置，其它位置填充 NEG_INF。
 95 |     返回：(new_logits, keep_mask)
 96 |     """
 97 |     if k is None or k <= 0:
 98 |         keep_mask = torch.ones_like(logits, dtype=torch.bool)
 99 |         return (logits, keep_mask) if not inplace else (logits, keep_mask)
100 | 
101 |     # 兼容负维、非连续存储
102 |     last_dim = logits.size(dim)
103 |     k_eff = int(min(k, last_dim))
104 |     if k_eff <= 0:
105 |         keep_mask = torch.ones_like(logits, dtype=torch.bool)
106 |         return (logits, keep_mask)
107 | 
108 |     x = logits.contiguous()  # 有些 CUDA 版本在非 contiguous + topk 下不稳定
109 |     # 取 topk 索引
110 |     topk_vals, topk_idx = torch.topk(x, k=k_eff, dim=dim)
111 | 
112 |     # 构造布尔 keep_mask
113 |     keep_mask = torch.zeros_like(x, dtype=torch.bool)
114 |     keep_mask.scatter_(dim, topk_idx, True)
115 | 
116 |     if inplace:
117 |         x.masked_fill_(~keep_mask, NEG_INF)
118 |         return x, keep_mask
119 |     else:
120 |         new_logits = x.clone()
121 |         new_logits.masked_fill_(~keep_mask, NEG_INF)
122 |         return new_logits, keep_mask
123 | 
124 | 
125 | 
126 | def masked_topk_softmax(
127 |     logits: torch.Tensor,
128 |     mask: Optional[torch.Tensor],
129 |     k: int,
130 |     dim: int = -1,
131 |     temperature: float = 1.0,
132 | ) -> torch.Tensor:
133 |     """
134 |     组合操作：先对 logits 进行掩码，随后 Top-k 截断，再做 softmax。
135 | 
136 |     等价步骤：
137 |       1) logits[~mask] = -inf
138 |       2) 仅保留维度 dim 上的 Top-k，其余 = -inf
139 |       3) softmax(dim)
140 | 
141 |     参数
142 |     ----
143 |     logits : Float[...]
144 |     mask : Optional[Bool/Byte[...] ]
145 |     k : int
146 |     dim : int
147 |     temperature : float
148 | 
149 |     返回
150 |     ----
151 |     probs : Float[...]
152 |     """
153 |     if mask is not None:
154 |         logits = logits.clone()
155 |         fill_masked_(logits, mask, NEG_INF)
156 |     logits, _ = topk_mask_logits(logits, k=k, dim=dim, inplace=False)
157 |     return masked_softmax(logits, mask=None, dim=dim, temperature=temperature)
```

## File: F:\SomeProjects\CSGNN\spikenet_x\minimal_example.py

- Extension: .py
- Language: python
- Size: 2571 bytes
- Created: 2025-08-22 13:11:54
- Modified: 2025-09-14 23:31:34

### Code

```python
 1 | # -*- coding: utf-8 -*-
 2 | """
 3 | SpikeNet-X 最小可运行示例
 4 | 
 5 | 运行方法：
 6 |     python -m spikenet_x.minimal_example
 7 | """
 8 | 
 9 | import torch
10 | 
11 | # 动态地添加 spikenet_x 包的父目录到 sys.path
12 | # 以便在 spikenet_x 目录外也能运行此脚本（例如从项目根目录）
13 | import os
14 | import sys
15 | if __package__ is None or __package__ == '':
16 |     # a bit of a hack to get relative imports working when running as a script
17 |     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
18 |     from spikenet_x.model import SpikeNetX
19 | else:
20 |     from .model import SpikeNetX
21 | 
22 | 
23 | def erdos_renyi_edge_index(num_nodes: int, p: float = 0.1, seed: int = 42) -> torch.Tensor:
24 |     g = torch.Generator().manual_seed(seed)
25 |     mask = torch.rand((num_nodes, num_nodes), generator=g) < p
26 |     # 去除自环
27 |     mask.fill_diagonal_(False)
28 |     src, dst = mask.nonzero(as_tuple=True)
29 |     # edge_index: [2, E]，列为 (src, dst)
30 |     return torch.stack([src, dst], dim=0).to(torch.long)
31 | 
32 | 
33 | def main():
34 |     print("--- SpikeNet-X Minimal Example & Shape Check ---")
35 |     
36 |     T, N, d_in, d, Hs, L = 16, 64, 32, 64, 4, 2
37 |     E = N * 5
38 | 
39 |     print(f"Params: T={T}, N={N}, d_in={d_in}, d={d}, heads={Hs}, layers={L}")
40 | 
41 |     H0 = torch.randn(T, N, d_in)
42 |     edge_index = erdos_renyi_edge_index(N, p=0.05, seed=1)
43 |     time_idx = torch.arange(T)
44 | 
45 |     print(f"Input shapes: H={H0.shape}, edge_index={edge_index.shape}, time_idx={time_idx.shape}")
46 | 
47 |     model = SpikeNetX(
48 |         d_in=d_in,
49 |         d=d,
50 |         layers=L,
51 |         heads=Hs,
52 |         topk=8,
53 |         W=8,
54 |         attn_impl="dense",
55 |         out_dim=5,
56 |     )
57 |     model.eval() # 禁用 dropout
58 |     print(f"\nModel:\n{model}\n")
59 | 
60 |     with torch.no_grad():
61 |         out = model(H0, edge_index=edge_index, time_idx=time_idx)
62 | 
63 |     print("--- Output Shape Check ---")
64 |     print(f"Y_last (final features): {out['Y_last'].shape}")
65 |     print(f"S_list (spikes per layer): {out['S_list'].shape}")
66 |     print(f"V_list (voltages per layer): {out['V_list'].shape}")
67 |     print(f"logits (readout): {out['logits'].shape if out.get('logits') is not None else 'N/A'}")
68 |     print(f"M_last (last layer msg): {out['M_last'].shape}")
69 | 
70 |     # 检查形状是否符合预期
71 |     assert out["Y_last"].shape == (T, N, d)
72 |     assert out["S_list"].shape == (L, T, N)
73 |     assert out["V_list"].shape == (L, T, N)
74 |     if out.get("logits") is not None:
75 |         assert out["logits"].shape == (N, 5)
76 | 
77 |     print("\n✅ All shapes are correct.")
78 | 
79 | 
80 | if __name__ == "__main__":
81 |     main()
```

## File: F:\SomeProjects\CSGNN\spikenet_x\model.py

- Extension: .py
- Language: python
- Size: 2493 bytes
- Created: 2025-08-22 13:07:03
- Modified: 2025-09-16 22:04:11

### Code

```python
 1 | # spikenet_x/model.py
 2 | from __future__ import annotations
 3 | from typing import Dict, List, Optional
 4 | import torch
 5 | import torch.nn as nn
 6 | from .spiketdanet_layer import SpikeTDANetLayer
 7 | 
 8 | class SpikeTDANet(nn.Module):
 9 |     """
10 |     Spike-TDANet: A Spiking Temporal Delay Attention Network.
11 |     This model stacks SpikeTDANetLayer blocks.
12 |     """
13 |     # [MODIFIED] 更新__init__以传递LIF参数
14 |     def __init__(
15 |         self,
16 |         d_in: int,
17 |         d: int,
18 |         layers: int = 2,
19 |         heads: int = 4,
20 |         W: int = 32,
21 |         out_dim: Optional[int] = None,
22 |         readout: str = "mean",
23 |         # LIF Hyperparameters
24 |         lif_tau: float = 0.95,
25 |         lif_v_threshold: float = 1.0,
26 |         lif_alpha: float = 1.0,
27 |         lif_surrogate: str = 'sigmoid',
28 |         **kwargs
29 |     ) -> None:
30 |         super().__init__()
31 |         assert layers >= 1
32 |         assert readout in ("last", "mean")
33 | 
34 |         self.readout = readout
35 |         self.d_in = d_in
36 |         self.d = d
37 |         
38 |         self.input_proj = nn.Linear(d_in, d)
39 |         
40 |         self.layers = nn.ModuleList()
41 |         for _ in range(layers):
42 |             self.layers.append(SpikeTDANetLayer(
43 |                 channels=d, heads=heads, W=W, 
44 |                 lif_tau=lif_tau, lif_v_threshold=lif_v_threshold,
45 |                 lif_alpha=lif_alpha, lif_surrogate=lif_surrogate,
46 |                 **kwargs
47 |             ))
48 | 
49 |         self.head = nn.Linear(d, out_dim) if out_dim is not None else None
50 | 
51 |     def forward(
52 |         self,
53 |         H: torch.Tensor,
54 |         edge_index: torch.Tensor,
55 |         time_idx: torch.Tensor,
56 |     ) -> Dict[str, torch.Tensor]:
57 |         
58 |         features = self.input_proj(H)
59 |         spikes = None  # 第一层没有前序脉冲
60 |         
61 |         spike_list = []
62 | 
63 |         for layer in self.layers:
64 |             # [MODIFIED] spikes的形状在层间传递时是[T, N]
65 |             features, spikes = layer(features, spikes, edge_index, time_idx)
66 |             spike_list.append(spikes)
67 | 
68 |         if self.readout == "last":
69 |             z = features[-1]
70 |         else:
71 |             z = features.mean(dim=0)
72 | 
73 |         logits = self.head(z) if self.head is not None else None
74 | 
75 |         out: Dict[str, torch.Tensor] = {
76 |             "repr": z,
77 |             "Y_last": features,
78 |             "S_list": torch.stack(spike_list) if spike_list else torch.empty(0),
79 |         }
80 |         if logits is not None:
81 |             out["logits"] = logits
82 | 
83 |         return out
```

## File: F:\SomeProjects\CSGNN\spikenet_x\rel_time.py

- Extension: .py
- Language: python
- Size: 6544 bytes
- Created: 2025-08-22 12:52:55
- Modified: 2025-09-15 03:49:37

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | 相对时间编码与相对偏置
  4 | 
  5 | - 提供 RelativeTimeEncoding(nn.Module)
  6 |   * forward(time_idx: Long[T], W:int) -> (pe_table: Float[W+1, d_pe], rel_bias: Float[W+1])
  7 |   * pe_table[k] 表示 Δt = k 的编码向量（仅用于 0..W）
  8 |   * rel_bias 为可学习标量偏置 b[Δt]，长度 W+1
  9 | 
 10 | - 设计：
 11 |   * 指数衰减对（tau_m, tau_s）
 12 |   * 对数间隔频率的 sin/cos 对（n_freq 个频率）
 13 |   * 可选 log-bucket one-hot（num_buckets=0 表示关闭）
 14 |   * 输出维度 d_pe = 2 + 2*n_freq + num_buckets
 15 | 
 16 | - 数值与工程：
 17 |   * 仅构造 0..W 的 Δt 子表，避免构建完整 [T,T] 矩阵
 18 |   * Δt 使用 float 计算后堆叠为编码
 19 | """
 20 | 
 21 | from __future__ import annotations
 22 | 
 23 | from typing import Tuple
 24 | 
 25 | import math
 26 | import torch
 27 | import torch.nn as nn
 28 | 
 29 | 
 30 | class RelativeTimeEncoding(nn.Module):
 31 |     """
 32 |     相对时间编码（仅依赖 Δt），并带可学习相对偏置表 b[0..W]。
 33 | 
 34 |     参数
 35 |     ----
 36 |     taus : Tuple[float, float]
 37 |         指数衰减的两个时间常数 (tau_m, tau_s)
 38 |     n_freq : int
 39 |         正弦/余弦的对数间隔频率个数
 40 |     num_buckets : int
 41 |         log-bucket one-hot 的桶数（0 表示关闭）
 42 |     """
 43 | 
 44 |     def __init__(
 45 |         self,
 46 |         taus: Tuple[float, float] = (4.0, 16.0),
 47 |         n_freq: int = 3,
 48 |         num_buckets: int = 0,
 49 |     ) -> None:
 50 |         super().__init__()
 51 |         assert len(taus) == 2 and taus[0] > 0 and taus[1] > 0
 52 |         assert n_freq >= 0
 53 |         assert num_buckets >= 0
 54 | 
 55 |         self.tau_m = float(taus[0])
 56 |         self.tau_s = float(taus[1])
 57 |         self.n_freq = int(n_freq)
 58 |         self.num_buckets = int(num_buckets)
 59 | 
 60 |         # 缓存最近一次构造的 W，以便重用不同 batch 的同一窗口
 61 |         self._cached_W = None
 62 |         self.register_buffer("_cached_pe", None, persistent=False)
 63 |         self.register_buffer("_cached_bias", None, persistent=False)
 64 | 
 65 |         # 注意：rel_bias 的长度依赖于 W，故在 forward 时按需创建/扩展为 Parameter
 66 |         self.register_parameter("_rel_bias", None)
 67 | 
 68 |     @property
 69 |     def d_pe(self) -> int:
 70 |         # 2 (双指数) + 2*n_freq (sin/cos 对) + num_buckets (one-hot)
 71 |         return 2 + 2 * self.n_freq + self.num_buckets
 72 | 
 73 |     @staticmethod
 74 |     def _log_spaced_frequencies(n_freq: int, W: int) -> torch.Tensor:
 75 |         """
 76 |         生成对数间隔频率（角频率 ω），范围大致覆盖 [1/(2W), 1/2]（经验值）。
 77 |         """
 78 |         if n_freq <= 0:
 79 |             return torch.empty(0)
 80 |         f_min = 1.0 / max(2.0 * W, 1.0)
 81 |         f_max = 0.5
 82 |         freqs = torch.logspace(
 83 |             start=math.log10(f_min),
 84 |             end=math.log10(f_max),
 85 |             steps=n_freq,
 86 |         )
 87 |         # 转为角频率
 88 |         return 2.0 * math.pi * freqs
 89 | 
 90 |     @staticmethod
 91 |     def _log_bucketize(delta: torch.Tensor, num_buckets: int) -> torch.Tensor:
 92 |         """
 93 |         将 Δt 做对数分桶并返回 one-hot；delta >= 0 的整数张量。
 94 |         """
 95 |         if num_buckets <= 0:
 96 |             return torch.empty(delta.shape + (0,), device=delta.device, dtype=delta.dtype)
 97 | 
 98 |         # +1 防止 log(0)
 99 |         logv = torch.log2(delta.to(torch.float32) + 1.0)
100 |         # 线性映射到桶 [0, num_buckets-1]
101 |         idx = torch.clamp((logv / torch.clamp(logv.max(), min=1.0e-6)) * (num_buckets - 1), 0, num_buckets - 1)
102 |         idx = idx.round().to(torch.long)
103 | 
104 |         one_hot = torch.zeros(delta.shape + (num_buckets,), device=delta.device, dtype=delta.dtype)
105 |         one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)
106 |         return one_hot
107 | 
108 |     def _build_pe_for_window(self, W: int, device: torch.device) -> torch.Tensor:
109 |         """
110 |         构建长度 W+1 的相对时间编码表：pe[k] = phi(Δt=k)，形状 [W+1, d_pe]
111 |         """
112 |         delta = torch.arange(0, W + 1, device=device, dtype=torch.float32)  # [0..W]
113 | 
114 |         # 双指数衰减
115 |         exp_m = torch.exp(-delta / self.tau_m)  # [W+1]
116 |         exp_s = torch.exp(-delta / self.tau_s)  # [W+1]
117 | 
118 |         # 正弦/余弦
119 |         omegas = self._log_spaced_frequencies(self.n_freq, W).to(device)
120 |         if omegas.numel() > 0:
121 |             # [W+1, n_freq]
122 |             arg = delta.unsqueeze(-1) * omegas.unsqueeze(0)
123 |             sinv = torch.sin(arg)
124 |             cosv = torch.cos(arg)
125 |             sincos = torch.cat([sinv, cosv], dim=-1)  # [W+1, 2*n_freq]
126 |         else:
127 |             sincos = torch.empty((W + 1, 0), device=device, dtype=torch.float32)
128 | 
129 |         # log-bucket one-hot
130 |         if self.num_buckets > 0:
131 |             buckets = self._log_bucketize(delta.to(torch.long), self.num_buckets).to(torch.float32)  # [W+1, B]
132 |         else:
133 |             buckets = torch.empty((W + 1, 0), device=device, dtype=torch.float32)
134 | 
135 |         pe = torch.cat(
136 |             [
137 |                 exp_m.unsqueeze(-1),
138 |                 exp_s.unsqueeze(-1),
139 |                 sincos,
140 |                 buckets,
141 |             ],
142 |             dim=-1,
143 |         )
144 |         return pe  # [W+1, d_pe]
145 | 
146 |     def _ensure_bias(self, W: int, device: torch.device) -> torch.Tensor:
147 |         """
148 |         确保存在长度 W+1 的可学习偏置表；如果已有更短表，做扩展并保留已学部分。
149 |         """
150 |         if self._rel_bias is None:
151 |             self._rel_bias = nn.Parameter(torch.zeros(W + 1, device=device))
152 |         elif self._rel_bias.numel() < (W + 1):
153 |             old = self._rel_bias.data
154 |             new = torch.zeros(W + 1, device=device)
155 |             new[: old.numel()] = old
156 |             self._rel_bias = nn.Parameter(new)
157 |         return self._rel_bias
158 | 
159 |     def forward(
160 |         self,
161 |         time_idx: torch.Tensor,  # Long[T]，通常为 arange(T)
162 |         W: int,
163 |     ) -> Tuple[torch.Tensor, torch.Tensor]:
164 |         """
165 |         构建窗口 0..W 的相对时间编码子表与相对偏置。
166 | 
167 |         返回
168 |         ----
169 |         pe_table : Float[W+1, d_pe]
170 |         rel_bias : Float[W+1]
171 |         """
172 |         assert time_idx.dim() == 1, "time_idx 应为一维 LongTensor [T]"
173 |         assert W >= 0, "W >= 0"
174 | 
175 |         device = time_idx.device
176 |         # 缓存与重用
177 |         if self._cached_W == W and self._cached_pe is not None and self._cached_pe.device == device:
178 |             pe = self._cached_pe
179 |         else:
180 |             pe = self._build_pe_for_window(W, device)
181 |             self._cached_W = W
182 |             self._cached_pe = pe
183 | 
184 |         rel_bias = self._ensure_bias(W, device)
185 |         return pe, rel_bias
```

## File: F:\SomeProjects\CSGNN\spikenet_x\spikenetx_layer.py

- Extension: .py
- Language: python
- Size: 5988 bytes
- Created: 2025-08-22 13:04:32
- Modified: 2025-08-22 23:31:01

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | SpikeNetXLayer: DelayLine + Spiking Temporal Attention (STA) + LIF 单层封装
  4 | 
  5 | 根据《提示词.md》的签名：
  6 |     class SpikeNetXLayer(nn.Module):
  7 |         def __init__(...):
  8 |             ...
  9 |         def forward(self, H, S_prev, edge_index, time_idx, adj_mask=None, batch=None):
 10 |             H̃ = self.delay(H)                                   # [T,N,d_in]
 11 |             M  = self.sta(H̃, S_prev, edge_index, time_idx, adj_mask)  # [T,N,d]
 12 |             S, V, aux = self.neuron(M)                           # [T,N], [T,N]
 13 |             Y = self.norm(M + self.ffn(M))                       # 残差 + 归一
 14 |             return S, V, Y, {"M": M, **aux}
 15 | 
 16 | 说明
 17 | ----
 18 | - 稠密 STA 回退实现在 spikenet_x/sta.py 中，适合小图或功能验证；
 19 | - 稀疏边 rolling-window 版本可在后续新增（接口保持一致）；
 20 | - 本层不改变时间长度 T 与节点数 N，仅改变通道维（d_in -> d）。
 21 | 
 22 | """
 23 | 
 24 | from __future__ import annotations
 25 | 
 26 | from typing import Dict, Optional, Tuple
 27 | 
 28 | import torch
 29 | import torch.nn as nn
 30 | import torch.nn.functional as F
 31 | 
 32 | from .delayline import LearnableDelayLine
 33 | from .sta import SpikingTemporalAttention
 34 | from .sta_sparse import SparseSpikingTemporalAttention
 35 | from .lif_cell import LIFCell
 36 | 
 37 | 
 38 | class MLP(nn.Module):
 39 |     def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.1) -> None:
 40 |         super().__init__()
 41 |         hidden = d * hidden_mult
 42 |         self.fc1 = nn.Linear(d, hidden, bias=True)
 43 |         self.fc2 = nn.Linear(hidden, d, bias=True)
 44 |         self.drop = nn.Dropout(dropout)
 45 | 
 46 |         self.reset_parameters()
 47 | 
 48 |     def reset_parameters(self) -> None:
 49 |         nn.init.xavier_uniform_(self.fc1.weight)
 50 |         nn.init.zeros_(self.fc1.bias)
 51 |         nn.init.xavier_uniform_(self.fc2.weight)
 52 |         nn.init.zeros_(self.fc2.bias)
 53 | 
 54 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 55 |         x = self.fc1(x)
 56 |         x = F.gelu(x)
 57 |         x = self.drop(x)
 58 |         x = self.fc2(x)
 59 |         return self.drop(x)
 60 | 
 61 | 
 62 | class SpikeNetXLayer(nn.Module):
 63 |     def __init__(
 64 |         self,
 65 |         d_in: int,
 66 |         d: int,
 67 |         heads: int = 4,
 68 |         topk: int = 16,
 69 |         W: int = 64,
 70 |         K: int = 5,
 71 |         rho: float = 0.85,
 72 |         use_rel_bias: bool = True,
 73 |         attn_drop: float = 0.1,
 74 |         temp: float = 1.0,
 75 |         attn_impl: str = "dense",
 76 |         per_channel: bool = True,
 77 |         ffn_hidden_mult: int = 4,
 78 |         ffn_drop: float = 0.1,
 79 |         lif_lambda_mem: float = 0.95,
 80 |         lif_tau_theta: float = 0.99,
 81 |         lif_gamma: float = 0.10,
 82 |         lif_adaptive: bool = True,
 83 |         lif_surrogate: str = "fast_tanh",
 84 |         lif_beta: float = 2.0,
 85 |     ) -> None:
 86 |         super().__init__()
 87 |         self.d_in = int(d_in)
 88 |         self.d = int(d)
 89 |         self.attn_impl = str(attn_impl)
 90 |         assert self.attn_impl in ("dense", "sparse"), "attn_impl must be 'dense' or 'sparse'"
 91 | 
 92 |         # 1) DelayLine（因果深度可分离 1D 卷积）
 93 |         self.delay = LearnableDelayLine(d_in=d_in, K=K, rho=rho, per_channel=per_channel)
 94 | 
 95 |         # 2) STA 聚合：根据 attn_impl 选择稠密/稀疏实现
 96 |         if self.attn_impl == "sparse":
 97 |             # 稀疏版本不支持 topk（后续可流式实现）
 98 |             self.sta = SparseSpikingTemporalAttention(
 99 |                 d_in=d_in,
100 |                 d=d,
101 |                 heads=heads,
102 |                 W=W,
103 |                 use_rel_bias=use_rel_bias,
104 |                 attn_drop=attn_drop,
105 |                 temp=temp,
106 |             )
107 |         else:
108 |             self.sta = SpikingTemporalAttention(
109 |                 d_in=d_in,
110 |                 d=d,
111 |                 heads=heads,
112 |                 topk=topk,
113 |                 W=W,
114 |                 use_rel_bias=use_rel_bias,
115 |                 attn_drop=attn_drop,
116 |                 temp=temp,
117 |             )
118 | 
119 |         # 3) 脉冲单元
120 |         self.neuron = LIFCell(
121 |             d=d,
122 |             lambda_mem=lif_lambda_mem,
123 |             tau_theta=lif_tau_theta,
124 |             gamma=lif_gamma,
125 |             adaptive=lif_adaptive,
126 |             surrogate=lif_surrogate,
127 |             beta=lif_beta,
128 |         )
129 | 
130 |         # 归一与 FFN（残差）
131 |         self.norm = nn.LayerNorm(d)
132 |         self.ffn = MLP(d=d, hidden_mult=ffn_hidden_mult, dropout=ffn_drop)
133 | 
134 |     def forward(
135 |         self,
136 |         H: torch.Tensor,                       # [T, N, d_in]
137 |         S_prev: Optional[torch.Tensor],        # [T, N] 或 None（None 时采用全 1 门控）
138 |         edge_index: Optional[torch.Tensor],    # [2, E] 或 None（若提供 adj_mask 可为 None）
139 |         time_idx: torch.Tensor,                # [T]
140 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool
141 |         batch: Optional[torch.Tensor] = None,     # 预留；当前未使用
142 |     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
143 |         assert H.dim() == 3, "H 形状应为 [T, N, d_in]"
144 |         T, N, Din = H.shape
145 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
146 | 
147 |         device = H.device
148 |         dtype = H.dtype
149 | 
150 |         # 若上一层脉冲缺省，则使用全 1 门控（不抑制注意力）
151 |         if S_prev is None:
152 |             S_gate = torch.ones((T, N), device=device, dtype=dtype)
153 |         else:
154 |             assert S_prev.shape == (T, N), "S_prev 形状应为 [T, N]"
155 |             S_gate = S_prev.to(device=device, dtype=dtype)
156 | 
157 |         # 1) DelayLine
158 |         H_tilde = self.delay(H)  # [T, N, d_in]
159 | 
160 |         # 2) STA 聚合为 d 维消息
161 |         M = self.sta(H_tilde, S_gate, edge_index=edge_index, time_idx=time_idx, adj_mask=adj_mask)  # [T, N, d]
162 | 
163 |         # 3) LIF 发放
164 |         S, V, aux = self.neuron(M)  # S:[T,N], V:[T,N]
165 |         aux = {"M": M, **aux}
166 | 
167 |         # 4) 残差 + 归一（Pre-LN 的一个简化变体）
168 |         Y = self.norm(M + self.ffn(M))  # [T, N, d]
169 | 
170 |         return S, V, Y, aux
```

## File: F:\SomeProjects\CSGNN\spikenet_x\spiketdanet_layer.py

- Extension: .py
- Language: python
- Size: 3978 bytes
- Created: 2025-09-15 02:24:13
- Modified: 2025-09-16 22:16:07

### Code

```python
 1 | # 文件: spikenet_x/spiketdanet_layer.py
 2 | 
 3 | import torch
 4 | import torch.nn as nn
 5 | from typing import Tuple, Optional
 6 | 
 7 | # from .new_modules import SpatialGNNWrapper, DelayLine, STAGNNAggregator
 8 | from .new_modules import SpatialGNNWrapper, DelayLine
 9 | from .new_modules.sta_gnn_agg_optimized import STAGNNAggregator_Optimized as STAGNNAggregator
10 | 
11 | from .surrogate_lif_cell import SurrogateLIFCell # [MODIFIED] 导入新的LIF单元
12 | 
13 | class MLP(nn.Module):
14 |     def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.1):
15 |         super().__init__()
16 |         self.fc1 = nn.Linear(d, d * hidden_mult)
17 |         self.fc2 = nn.Linear(d * hidden_mult, d)
18 |         self.drop = nn.Dropout(dropout)
19 |         self.act = nn.GELU()
20 | 
21 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
22 |         return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
23 | 
24 | class SpikeTDANetLayer(nn.Module):
25 |     # [MODIFIED] 更新__init__签名以接收LIF超参数
26 |     def __init__(self, channels: int, heads: int, W: int, delay_kernel: int = 5, 
27 |                  lif_tau=0.95, lif_v_threshold=1.0, lif_alpha=1.0, lif_surrogate='sigmoid', **kwargs):
28 |         super().__init__()
29 |         self.channels = channels
30 |         
31 |         # 1. 空间GNN预处理
32 |         self.spatial_gnn = SpatialGNNWrapper(channels, channels)
33 |         self.norm1 = nn.LayerNorm(channels)
34 | 
35 |         # 2. 时间延迟建模
36 |         self.delay_line = DelayLine(channels, kernel_size=delay_kernel)
37 |         self.norm2 = nn.LayerNorm(channels)
38 | 
39 |         # 3. 时空信息聚合
40 |         self.aggregator = STAGNNAggregator(d_in=channels, d=channels, heads=heads, W=W, **kwargs)
41 |         
42 |         # 4. 脉冲发放 (使用新的LIF单元)
43 |         # [REMOVED] self.msg_proj = nn.Linear(channels, 1) # 移除信息瓶颈
44 |         self.lif_cell = SurrogateLIFCell(
45 |             channels=channels,
46 |             tau=lif_tau,
47 |             v_threshold=lif_v_threshold,
48 |             alpha=lif_alpha,
49 |             surrogate=lif_surrogate
50 |         )
51 |         
52 |         # 5. FFN 和最终输出处理
53 |         self.ffn = MLP(channels)
54 |         self.final_norm = nn.LayerNorm(channels)
55 | 
56 |     def forward(self, x: torch.Tensor, spikes: Optional[torch.Tensor], edge_index: torch.Tensor, time_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
57 |         T, N, C = x.shape
58 |         
59 |         # --- [NEW] START: Handle None spikes for the first layer ---
60 |         if spikes is None:
61 |             # For the first layer, assume all-one spikes (no gating)
62 |             # Shape should be [T, N, C] to match features for the aggregator
63 |             spikes = torch.ones((T, N, C), device=x.device, dtype=x.dtype)
64 |         # --- [NEW] END ---
65 |         
66 |         # [MODIFIED] 如果上一层脉冲是[T,N]，需要扩展以匹配特征维度
67 |         # This part is now a fallback, but good to keep.
68 |         elif spikes.dim() == 2:
69 |             spikes = spikes.unsqueeze(-1).expand(-1, -1, C)
70 | 
71 |         # 1. 空间GNN预处理
72 |         x_spatial = self.spatial_gnn(x, edge_index)
73 |         x = self.norm1(x + x_spatial)
74 | 
75 |         # 2. 时间延迟建模
76 |         x_delayed = self.delay_line(x)
77 |         x_processed = self.norm2(x + x_delayed)
78 | 
79 |         # 3. 时空聚合
80 |         aggregated_message = self.aggregator(x_processed, spikes, edge_index, time_idx)
81 | 
82 |         # 4. 脉冲发放 (处理高维特征)
83 |         # [MODIFIED] 直接将高维消息送入LIF
84 |         spikes_hd, _ = self.lif_cell(aggregated_message) # spikes_hd shape: [T, N, C]
85 | 
86 |         # 5. 最终输出 (FFN + 宏观残差)
87 |         # [MODIFIED] 使用脉冲作为门控信号
88 |         ffn_input = aggregated_message * spikes_hd
89 |         ffn_out = self.ffn(ffn_input)
90 |         layer_output_features = self.final_norm(x_processed + ffn_out)
91 | 
92 |         # [MODIFIED] 为下一层准备脉冲信号 (通过平均降维)
93 |         new_spikes_for_next_layer = spikes_hd.mean(dim=-1)
94 | 
95 |         return layer_output_features, new_spikes_for_next_layer
96 | 
```

## File: F:\SomeProjects\CSGNN\spikenet_x\sta.py

- Extension: .py
- Language: python
- Size: 9674 bytes
- Created: 2025-08-22 12:59:28
- Modified: 2025-09-14 23:29:21

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | SpikingTemporalAttention（STA）——稠密回退实现（小图/验证用）
  4 | 
  5 | 功能
  6 | ----
  7 | - 在因果与邻接掩码下，对 (邻居 j, 过去时间 t') 的键值进行多头注意力；
  8 | - 将源端脉冲 S[j,t'] 作为门控（在 logit 上加 log(S+eps) 等价于概率缩放）；
  9 | - 支持 Top-k 稀疏化（在 (j,t') 的联合候选维度上执行）；
 10 | - 支持相对时间编码与可学习相对偏置 b[Δt]；
 11 | - 输出每个 (t,i) 的聚合消息 M[t,i,:]，形状 [T, N, d]。
 12 | 
 13 | 复杂度
 14 | ------
 15 | Dense 回退：O(T * (W+1) * H * N^2)。在大图上请实现/切换稀疏边版本。
 16 | 
 17 | 接口（与《提示词.md》一致）
 18 | ------------------------
 19 | forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E] 或 adj_mask:Bool[N,N],
 20 |         time_idx:Long[T]) -> M:[T,N,d]
 21 | """
 22 | 
 23 | from __future__ import annotations
 24 | 
 25 | from typing import Optional, Tuple
 26 | 
 27 | import torch
 28 | import torch.nn as nn
 29 | import torch.nn.functional as F
 30 | 
 31 | from .masked_ops import masked_topk_softmax
 32 | from .rel_time import RelativeTimeEncoding
 33 | 
 34 | 
 35 | def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
 36 |     # 形状检查
 37 |     assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
 38 |     if edge_index.numel() == 0:
 39 |         return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
 40 | 
 41 |     # --- 越界保护：确保传入的是“局部 id（0..num_nodes-1）” ---
 42 |     max_id = int(edge_index.max().item())
 43 |     min_id = int(edge_index.min().item())
 44 |     if max_id >= num_nodes or min_id < 0:
 45 |         raise RuntimeError(
 46 |             f"[dense adj] edge_index 越界：min={min_id}, max={max_id}, 但 num_nodes={num_nodes}。"
 47 |             "请确认子图边已映射为局部 id（0..N_sub-1），且 H0_subgraph 的 N 与之匹配。"
 48 |         )
 49 | 
 50 |     # 用 GPU 写稠密邻接（安全、快速）
 51 |     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
 52 |     src = edge_index[0].to(torch.long)
 53 |     dst = edge_index[1].to(torch.long)
 54 |     adj[dst, src] = True
 55 |     return adj
 56 | 
 57 | 
 58 | 
 59 | class SpikingTemporalAttention(nn.Module):
 60 |     def __init__(
 61 |         self,
 62 |         d_in: int,
 63 |         d: int,
 64 |         heads: int = 4,
 65 |         topk: int = 16,
 66 |         W: int = 64,
 67 |         use_rel_bias: bool = True,
 68 |         attn_drop: float = 0.1,
 69 |         temp: float = 1.0,
 70 |         # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8，符合提示词推荐）
 71 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
 72 |         pe_n_freq: int = 3,
 73 |         pe_num_buckets: int = 0,
 74 |     ) -> None:
 75 |         super().__init__()
 76 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
 77 |         assert topk >= 1, "topk 必须 >= 1"
 78 |         assert W >= 0, "W 必须 >= 0"
 79 | 
 80 |         self.d_in = int(d_in)
 81 |         self.d = int(d)
 82 |         self.heads = int(heads)
 83 |         self.d_head = self.d // self.heads
 84 |         self.topk = int(topk)
 85 |         self.W = int(W)
 86 |         self.use_rel_bias = bool(use_rel_bias)
 87 |         self.temp = float(temp)
 88 | 
 89 |         # 相对时间编码
 90 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq, num_buckets=pe_num_buckets)
 91 |         d_pe = self.rel_enc.d_pe
 92 | 
 93 |         # 线性投影（K 拼接相对时间编码）
 94 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
 95 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
 96 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
 97 | 
 98 |         self.attn_drop = nn.Dropout(attn_drop) if attn_drop and attn_drop > 0 else nn.Identity()
 99 |         self.scale = self.d_head ** -0.5
100 | 
101 |         self.reset_parameters()
102 | 
103 |     def reset_parameters(self) -> None:
104 |         nn.init.xavier_uniform_(self.W_q.weight)
105 |         nn.init.xavier_uniform_(self.W_k.weight)
106 |         nn.init.xavier_uniform_(self.W_v.weight)
107 | 
108 |     @staticmethod
109 |     def _build_adj_mask(
110 |         N: int,
111 |         edge_index: Optional[torch.Tensor],
112 |         adj_mask: Optional[torch.Tensor],
113 |         device: torch.device,
114 |     ) -> torch.Tensor:
115 |         """
116 |         构造邻接掩码 A: Bool[N,N]，A[i,j]=True 表示 j->i 存在边（j 属于 i 的邻居）
117 |         """
118 |         if adj_mask is not None:
119 |             A = adj_mask.to(device=device, dtype=torch.bool)
120 |             assert A.shape == (N, N), f"adj_mask 形状应为 [{N},{N}]"
121 |             return A
122 |         assert edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 2, \
123 |             "未提供 adj_mask 时必须提供 edge_index，形状 [2,E]"
124 |         return _edge_index_to_dense_adj(edge_index.to(device), N, device=device)
125 | 
126 |     def forward(
127 |         self,
128 |         H_tilde: torch.Tensor,            # [T, N, d_in]
129 |         S: torch.Tensor,                  # [T, N] in [0,1]
130 |         edge_index: Optional[torch.Tensor],  # [2, E] 或 None
131 |         time_idx: torch.Tensor,           # [T]
132 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool 或 None
133 |     ) -> torch.Tensor:
134 |         assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
135 |         assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
136 |         assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"
137 | 
138 |         T, N, Din = H_tilde.shape
139 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
140 | 
141 |         device = H_tilde.device
142 |         dtype = H_tilde.dtype
143 | 
144 |         # 邻接掩码（稠密回退）
145 |         A = self._build_adj_mask(N, edge_index, adj_mask, device)  # [N,N] Bool
146 | 
147 |         # 相对时间编码与（可选）偏置（仅构造 0..W）
148 |         pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
149 |         if not self.use_rel_bias:
150 |             rel_bias = torch.zeros_like(rel_bias)
151 | 
152 |         # 预计算所有时刻的 Q、V
153 |         Q_all = self.W_q(H_tilde)  # [T, N, d]
154 |         V_all = self.W_v(H_tilde)  # [T, N, d]
155 | 
156 |         # 输出容器
157 |         M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
158 | 
159 |         eps_gate = 1.0e-6
160 | 
161 |         for t in range(T):
162 |             W_eff = min(self.W, t)
163 | 
164 |             # 多头视图
165 |             # Q_t: [N, H, d_h] -> 转为 [H, N, d_h] 便于后续计算
166 |             Q_t = Q_all[t].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
167 | 
168 |             logits_chunks = []   # 每个块 [H,N,N]
169 |             mask_chunks = []     # 每个块 [H,N,N]
170 |             V_chunks = []        # 每个块 [H,N,d_h]
171 |             gate_chunks = []     # 每个块 [1,1,N]（用于 log-domain 加法；在拼接后按候选展开）
172 | 
173 |             for dt in range(W_eff + 1):
174 |                 t_prime = t - dt
175 | 
176 |                 # 相对时间编码拼接到 K 输入
177 |                 pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
178 |                 pe_expand = pe.view(1, 1, -1).expand(N, -1, -1)   # [N,1,d_pe] -> 与 H_tilde[t'] 拼接
179 |                 K_in = torch.cat([H_tilde[t_prime], pe_expand.squeeze(1)], dim=-1)  # [N, d_in + d_pe]
180 | 
181 |                 # 线性映射并切分多头
182 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
183 |                 V_tp = V_all[t_prime].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
184 | 
185 |                 # 注意力 logits（缩放点积）
186 |                 # scores[h,i,j] = <Q_t[h,i,:], K_tp[h,j,:]>
187 |                 scores = torch.einsum("hid,hjd->hij", Q_t, K_tp) * self.scale  # [H,N,N]
188 | 
189 |                 # 相对偏置共享到 (i,j)
190 |                 if rel_bias is not None:
191 |                     scores = scores + float(rel_bias[dt])
192 | 
193 |                 # 源端脉冲门控（log-domain 加法）
194 |                 gate_j = torch.clamp(S[t_prime], 0.0, 1.0).to(dtype=dtype)  # [N]
195 |                 gate_chunks.append(torch.log(0.5 + 0.5 * gate_j + eps_gate).view(1, 1, N))
196 | 
197 |                 # 邻接掩码广播到各头
198 |                 mask_hij = A.view(1, N, N).expand(self.heads, -1, -1)  # [H,N,N]
199 | 
200 |                 logits_chunks.append(scores)
201 |                 mask_chunks.append(mask_hij)
202 |                 V_chunks.append(V_tp)
203 | 
204 |             if not logits_chunks:
205 |                 # 该 t 无可用键（仅 t=0 且 W=0 时可能发生）
206 |                 continue
207 | 
208 |             # 拼接候选维（按 dt 依次拼接）
209 |             # logits_flat: [H,N,(W_eff+1)*N]
210 |             logits_flat = torch.cat(logits_chunks, dim=2)
211 |             mask_flat = torch.cat(mask_chunks, dim=2)  # [H,N,(W_eff+1)*N]
212 |             gate_log_flat = torch.cat(
213 |                 [g.expand(self.heads, N, -1) for g in gate_chunks], dim=2
214 |             )  # [H,N,(W_eff+1)*N]
215 |             logits_flat = logits_flat + gate_log_flat
216 | 
217 |             # Top-k + masked softmax（温度缩放在函数内部）
218 |             k_eff = min(self.topk, logits_flat.size(-1))
219 |             probs = masked_topk_softmax(
220 |                 logits_flat, mask_flat, k=k_eff, dim=-1, temperature=self.temp
221 |             )  # [H,N,(W_eff+1)*N]
222 |             probs = self.attn_drop(probs)
223 | 
224 |             # 构造对应的值向量拼接：V_cat: [H,(W_eff+1)*N,d_h]
225 |             V_cat = torch.cat(V_chunks, dim=1)  # [H, (W_eff+1)*N, d_h]
226 | 
227 |             # 聚合：msg_h = probs @ V_cat
228 |             msg_h = torch.einsum("hni,hid->hnd", probs, V_cat)  # [H,N,d_h]
229 | 
230 |             # 合并头并写入输出
231 |             M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
232 |             M_out[t] = M_t
233 | 
234 |         return M_out  # [T,N,d]
```

## File: F:\SomeProjects\CSGNN\spikenet_x\sta_sparse.py

- Extension: .py
- Language: python
- Size: 12853 bytes
- Created: 2025-08-22 23:21:28
- Modified: 2025-09-14 21:51:59

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | Sparse Spiking Temporal Attention（STA）——O(E) 稀疏实现（大图可用）
  4 | 
  5 | 设计要点
  6 | --------
  7 | - 不构造 [N,N] 稠密矩阵，完全基于 edge_index = (src,dst) 按边计算；
  8 | - 时间因果窗口 W：对每个 (t,i) 仅聚合 t' ∈ [t-W, t] 的消息；
  9 | - 源端脉冲门控：在 logit 上加 log(S[t',src] + eps)，等价于概率缩放；
 10 | - 两遍 segment-softmax（数值稳定）：
 11 |     Pass-1：按接收端 dst 计算各头的 segment-wise amax（log-sum-exp 的 max 项）；
 12 |     Pass-2：重新计算 exp(score - amax(dst))，用 scatter_add 聚合分母/分子得到消息；
 13 | - 相对时间编码与可学习偏置 b[Δt] 由 RelativeTimeEncoding 复用；
 14 | - 注意：本稀疏版本当前不实现 Top-K 截断（dense 版本支持），必要时后续可加入“每 dst 流式 Top-K”。
 15 | 
 16 | 复杂度
 17 | ------
 18 | O(T * H * W * E)，显存主要为按 E 规模的临时张量。适用于大图/子图批训练。
 19 | 
 20 | 接口
 21 | ----
 22 | forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E], time_idx:Long[T]) -> M:[T,N,d]
 23 | """
 24 | 
 25 | from __future__ import annotations
 26 | 
 27 | from typing import Optional, Tuple
 28 | 
 29 | import math
 30 | import torch
 31 | import torch.nn as nn
 32 | import torch.nn.functional as F
 33 | 
 34 | from .rel_time import RelativeTimeEncoding
 35 | 
 36 | 
 37 | def _has_scatter_reduce_tensor() -> bool:
 38 |     # PyTorch 1.12+ 提供 Tensor.scatter_reduce_
 39 |     return hasattr(torch.Tensor, "scatter_reduce_")
 40 | 
 41 | 
 42 | def _try_import_torch_scatter():
 43 |     try:
 44 |         import torch_scatter  # type: ignore
 45 |         return torch_scatter
 46 |     except Exception:
 47 |         return None
 48 | 
 49 | 
 50 | _TORCH_SCATTER = _try_import_torch_scatter()
 51 | _HAS_TSR = _has_scatter_reduce_tensor()
 52 | 
 53 | 
 54 | def _segment_amax_1d(x: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
 55 |     """
 56 |     计算 out[j] = max_{i: index[i]==j} x[i]，其中 j ∈ [0, K)
 57 |     优先使用 Tensor.scatter_reduce_('amax')；其次 torch_scatter.scatter_max；最终回退到排序段法。
 58 |     """
 59 |     device, dtype = x.device, x.dtype
 60 |     neg_inf = torch.tensor(-1e30, dtype=dtype, device=device)
 61 | 
 62 |     if _HAS_TSR:
 63 |         # 使用非 in-place 版本 scatter_reduce 修复梯度计算问题
 64 |         # include_self=False 时，空段结果未定义，需手动填充
 65 |         init_val = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
 66 |         out = init_val.scatter_reduce(0, index, x, reduce="amax", include_self=False)
 67 |         return out
 68 | 
 69 |     if _TORCH_SCATTER is not None:
 70 |         # torch_scatter.scatter_max 返回 (out, argmax)
 71 |         out, _ = _TORCH_SCATTER.scatter_max(x, index, dim=0, dim_size=K)
 72 |         # 对于空段，scatter_max 会给出 0；为一致性，将空段填为 -inf：
 73 |         # 通过统计计数判断空段
 74 |         cnt = torch.zeros(K, device=device, dtype=torch.long)
 75 |         cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
 76 |         out = torch.where(cnt > 0, out, neg_inf)
 77 |         return out
 78 | 
 79 |     # 回退：排序段法（可能较慢，但不依赖扩展）
 80 |     perm = torch.argsort(index)
 81 |     idx_s = index[perm]
 82 |     x_s = x[perm]
 83 |     out = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
 84 |     if idx_s.numel() == 0:
 85 |         return out
 86 |     # 找段边界
 87 |     boundary = torch.ones_like(idx_s, dtype=torch.bool)
 88 |     boundary[1:] = idx_s[1:] != idx_s[:-1]
 89 |     # 段起点位置
 90 |     starts = torch.nonzero(boundary, as_tuple=False).flatten()
 91 |     # 段终点（含）位置
 92 |     ends = torch.empty_like(starts)
 93 |     ends[:-1] = starts[1:] - 1
 94 |     ends[-1] = idx_s.numel() - 1
 95 |     # 逐段计算最大（Python 循环，仅在无高阶算子时作为兜底）
 96 |     for s, e in zip(starts.tolist(), ends.tolist()):
 97 |         j = int(idx_s[s].item())
 98 |         out[j] = torch.maximum(out[j], x_s[s : e + 1].max())
 99 |     return out
100 | 
101 | 
102 | class SparseSpikingTemporalAttention(nn.Module):
103 |     def __init__(
104 |         self,
105 |         d_in: int,
106 |         d: int,
107 |         heads: int = 4,
108 |         W: int = 64,
109 |         use_rel_bias: bool = True,
110 |         attn_drop: float = 0.1,
111 |         temp: float = 1.0,
112 |         # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8）
113 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
114 |         pe_n_freq: int = 3,
115 |         pe_num_buckets: int = 0,
116 |     ) -> None:
117 |         """
118 |         稀疏 STA，不支持 Top-K 截断（如需可后续加入流式 Top-K）。
119 |         """
120 |         super().__init__()
121 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
122 |         assert W >= 0, "W 必须 >= 0"
123 | 
124 |         self.d_in = int(d_in)
125 |         self.d = int(d)
126 |         self.heads = int(heads)
127 |         self.d_head = self.d // self.heads
128 |         self.W = int(W)
129 |         self.use_rel_bias = bool(use_rel_bias)
130 |         self.temp = float(temp)
131 | 
132 |         # 相对时间编码
133 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq, num_buckets=pe_num_buckets)
134 |         d_pe = self.rel_enc.d_pe
135 | 
136 |         # 线性投影（K 拼接相对时间编码）
137 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
138 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
139 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
140 | 
141 |         # 注意力 dropout：实现为“边贡献丢弃再归一化”的无缩放 Bernoulli mask（训练时）
142 |         self.p_drop = float(attn_drop)
143 |         self.scale = self.d_head ** -0.5
144 | 
145 |         self.reset_parameters()
146 | 
147 |     def reset_parameters(self) -> None:
148 |         nn.init.xavier_uniform_(self.W_q.weight)
149 |         nn.init.xavier_uniform_(self.W_k.weight)
150 |         nn.init.xavier_uniform_(self.W_v.weight)
151 | 
152 |     @staticmethod
153 |     def _check_edges(edge_index: torch.Tensor, N: int) -> None:
154 |         assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
155 |         E = edge_index.size(1)
156 |         if E == 0:
157 |             return
158 |         assert int(edge_index.min()) >= 0 and int(edge_index.max()) < N, "edge_index 越界"
159 | 
160 |     def _edge_arrays(self, edge_index: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
161 |         # 拆分 src, dst 为长度 E 的向量
162 |         src = edge_index[0].to(device=device, dtype=torch.long)
163 |         dst = edge_index[1].to(device=device, dtype=torch.long)
164 |         return src, dst
165 | 
166 |     @torch.no_grad()
167 |     def _drop_mask(self, shape: torch.Size, device: torch.device) -> Optional[torch.Tensor]:
168 |         if not self.training or self.p_drop <= 0.0:
169 |             return None
170 |         return (torch.rand(shape, device=device) > self.p_drop)
171 | 
172 |     def forward(
173 |         self,
174 |         H_tilde: torch.Tensor,            # [T, N, d_in]
175 |         S: torch.Tensor,                  # [T, N] in [0,1]
176 |         edge_index: torch.Tensor,         # [2, E]
177 |         time_idx: torch.Tensor,           # [T]
178 |         adj_mask: Optional[torch.Tensor] = None,  # 兼容签名；稀疏实现忽略
179 |     ) -> torch.Tensor:
180 |         assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
181 |         assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
182 |         assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"
183 | 
184 |         T, N, Din = H_tilde.shape
185 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
186 | 
187 |         device = H_tilde.device
188 |         dtype = H_tilde.dtype
189 | 
190 |         # 校验边并拆分
191 |         self._check_edges(edge_index, N)
192 |         src, dst = self._edge_arrays(edge_index, device)  # [E], [E]
193 |         E = src.numel()
194 | 
195 |         # 相对时间编码与偏置（仅构造 0..W）
196 |         pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
197 |         if not self.use_rel_bias:
198 |             rel_bias = torch.zeros_like(rel_bias)
199 | 
200 |         # 预计算 Q(t, ·) 与 V(t, ·)
201 |         Q_all = self.W_q(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()  # [T,H,N,d_h]
202 |         V_all = self.W_v(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()  # [T,H,N,d_h]
203 | 
204 |         # 输出
205 |         M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
206 | 
207 |         eps_gate = 1.0e-6
208 |         neg_inf = -1.0e30
209 | 
210 |         for t in range(T):
211 |             W_eff = min(self.W, t)
212 |             # Q_t: [H,N,d_h]
213 |             Q_t = Q_all[t]  # [H,N,d_h]
214 | 
215 |             # -------- Pass-1：按 dst 计算 segment-wise amax（各头独立） --------
216 |             max_dst_list = []
217 |             for dt in range(W_eff + 1):
218 |                 t_prime = t - dt
219 |                 # 构造 K_{t'}（拼接相对时间编码）
220 |                 pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
221 |                 K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)  # [N, d_in+d_pe]
222 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
223 | 
224 |                 # 门控（源端脉冲）
225 |                 gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)  # [N]
226 | 
227 |                 # 对每个头计算边打分：scores[h,e] = <Q_t[h,dst[e]], K_tp[h,src[e]]> * scale + b[dt] + log S
228 |                 # gather Q/K
229 |                 Q_d = Q_t[:, dst, :]              # [H,E,d_h]
230 |                 K_s = K_tp[:, src, :]             # [H,E,d_h]
231 |                 # 点积
232 |                 scores = (Q_d * K_s).sum(dim=-1) * self.scale  # [H,E]
233 |                 # 相对偏置
234 |                 scores = scores + float(rel_bias[dt])
235 |                 # 源脉冲门控
236 |                 scores = scores + gate_log[src]  # 广播到 [H,E]
237 | 
238 |                 # softmax 温度
239 |                 if self.temp != 1.0:
240 |                     scores = scores / float(self.temp)
241 | 
242 |                 # 对每头做 segment amax
243 |                 m_h = torch.stack([_segment_amax_1d(scores[h], dst, N) for h in range(self.heads)])
244 |                 max_dst_list.append(m_h)
245 | 
246 |             max_dst = torch.stack(max_dst_list, dim=0).max(dim=0)[0]
247 | 
248 |             # -------- Pass-2：exp(score - amax(dst)) 聚合分母/分子 --------
249 |             denom = torch.zeros((self.heads, N), device=device, dtype=dtype)          # [H,N]
250 |             numer = torch.zeros((self.heads, N, self.d_head), device=device, dtype=dtype)  # [H,N,d_h]
251 | 
252 |             for dt in range(W_eff + 1):
253 |                 t_prime = t - dt
254 |                 pe = pe_table[dt].to(dtype=dtype, device=device)
255 |                 K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)  # [N, d_in+d_pe]
256 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
257 |                 V_tp = V_all[t_prime]  # [H,N,d_h]
258 |                 gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)  # [N]
259 | 
260 |                 Q_d = Q_t[:, dst, :]          # [H,E,d_h]
261 |                 K_s = K_tp[:, src, :]         # [H,E,d_h]
262 |                 V_s = V_tp[:, src, :]         # [H,E,d_h]
263 | 
264 |                 scores = (Q_d * K_s).sum(dim=-1) * self.scale  # [H,E]
265 |                 scores = scores + float(rel_bias[dt])
266 |                 scores = scores + gate_log[src]
267 |                 if self.temp != 1.0:
268 |                     scores = scores / float(self.temp)
269 | 
270 |                 # exp(score - max_dst[dst])
271 |                 # broadcast: max_dst[:, dst] -> [H,E]
272 |                 max_g = max_dst[:, dst]
273 |                 ex = torch.exp(scores - max_g)  # [H,E]
274 | 
275 |                 # attention dropout：训练时对边贡献做伯努利丢弃（不做 1/(1-p) 缩放，随后自动归一化）
276 |                 mask = self._drop_mask(ex.shape, device=device)
277 |                 if mask is not None:
278 |                     ex = ex * mask.to(dtype=ex.dtype)
279 | 
280 |                 # 逐头 scatter_add 到 dst
281 |                 for h in range(self.heads):
282 |                     # 分母
283 |                     denom[h].index_add_(0, dst, ex[h])  # [N]
284 |                     # 分子：ex[h][:,None] * V_s[h] 累加到 dst
285 |                     contrib = ex[h].unsqueeze(-1) * V_s[h]  # [E,d_h]
286 |                     # 将 [E,d_h] 累加到 [N,d_h]：循环通道（d_h 小，循环成本可接受）
287 |                     # 向量化 index_add_ 仅支持 1D，这里按通道展开
288 |                     for c in range(self.d_head):
289 |                         numer[h, :, c].index_add_(0, dst, contrib[:, c])
290 | 
291 |             # 得到各头消息并合并
292 |             # 防零保护
293 |             denom = torch.clamp(denom, min=1e-12)
294 |             msg_h = numer / denom.unsqueeze(-1)  # [H,N,d_h]
295 |             M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
296 |             M_out[t] = M_t
297 | 
298 |         return M_out  # [T,N,d]
299 | 
300 | 
301 | __all__ = ["SparseSpikingTemporalAttention"]
```

## File: F:\SomeProjects\CSGNN\spikenet_x\surrogate_lif_cell.py

- Extension: .py
- Language: python
- Size: 3610 bytes
- Created: 2025-09-16 22:03:35
- Modified: 2025-09-16 22:03:41

### Code

```python
  1 | # F:\SomeProjects\CSGNN\spikenet_x\surrogate_lif_cell.py
  2 | import torch
  3 | import torch.nn as nn
  4 | from typing import Tuple
  5 | 
  6 | # --- 拷贝自 spikenet/neuron.py 的替代梯度函数 ---
  7 | class BaseSpike(torch.autograd.Function):
  8 |     @staticmethod
  9 |     def forward(ctx, x, alpha):
 10 |         ctx.save_for_backward(x, alpha)
 11 |         return x.gt(0).float()
 12 | 
 13 |     @staticmethod
 14 |     def backward(ctx, grad_output):
 15 |         raise NotImplementedError
 16 | 
 17 | class SigmoidSpike(BaseSpike):
 18 |     @staticmethod
 19 |     def backward(ctx, grad_output):
 20 |         x, alpha = ctx.saved_tensors
 21 |         grad_input = grad_output.clone()
 22 |         sgax = (x * alpha).sigmoid_()
 23 |         sg = (1. - sgax) * sgax * alpha
 24 |         return grad_input * sg, None
 25 | 
 26 | class TriangleSpike(BaseSpike):
 27 |     @staticmethod
 28 |     def backward(ctx, grad_output):
 29 |         x, alpha = ctx.saved_tensors
 30 |         grad_input = grad_output.clone()
 31 |         sg = torch.nn.functional.relu(1 - alpha * x.abs())
 32 |         return grad_input * sg, None
 33 | 
 34 | SURROGATE_MAP = {
 35 |     'sigmoid': SigmoidSpike.apply,
 36 |     'triangle': TriangleSpike.apply
 37 | }
 38 | # --- 替代梯度函数结束 ---
 39 | 
 40 | 
 41 | class SurrogateLIFCell(nn.Module):
 42 |     """
 43 |     一个支持高维特征并使用替代梯度进行训练的LIF神经元。
 44 |     它以 unrolled 方式处理 [T, N, D] 的输入。
 45 |     """
 46 |     def __init__(self, channels: int, v_threshold=1.0, v_reset=0.0, tau=0.95, alpha=1.0, surrogate='sigmoid'):
 47 |         super().__init__()
 48 |         self.channels = channels
 49 |         self.v_threshold = v_threshold
 50 |         self.v_reset = v_reset
 51 |         
 52 |         # 确保 tau 是一个可训练或固定的缓冲区
 53 |         self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
 54 |         self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
 55 |         
 56 |         if surrogate not in SURROGATE_MAP:
 57 |             raise ValueError(f"Surrogate function '{surrogate}' is not supported. Available: {list(SURROGATE_MAP.keys())}")
 58 |         self.surrogate_fn = SURROGATE_MAP[surrogate]
 59 | 
 60 |     def forward(self, I_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
 61 |         """
 62 |         LIF神经元的前向传播。
 63 | 
 64 |         Args:
 65 |             I_in (torch.Tensor): 输入电流/特征，形状为 [T, N, D]。
 66 | 
 67 |         Returns:
 68 |             Tuple[torch.Tensor, torch.Tensor]:
 69 |             - spikes (torch.Tensor): 输出脉冲序列，形状 [T, N, D]。
 70 |             - v_mem (torch.Tensor): 膜电位历史记录，形状 [T, N, D]。
 71 |         """
 72 |         T, N, D = I_in.shape
 73 |         device = I_in.device
 74 | 
 75 |         # 初始化膜电位
 76 |         v = torch.zeros(N, D, device=device)
 77 | 
 78 |         v_mem_list = []
 79 |         spike_list = []
 80 | 
 81 |         # 按时间步循环处理
 82 |         for t in range(T):
 83 |             # 1. 膜电位更新 (leaky integration)
 84 |             # v_new = v_old * tau (leak) + I_in (integrate)
 85 |             # 注意：原版Spike的LIF公式是 v = v + (dv - (v-v_reset))/tau
 86 |             # 这里简化为 v = v*tau + I_in，更常见
 87 |             v = v * self.tau + I_in[t]
 88 |             
 89 |             # 2. 发放脉冲 (使用替代梯度)
 90 |             # spike = surrogate(v - v_threshold)
 91 |             spike = self.surrogate_fn(v - self.v_threshold, self.alpha)
 92 |             
 93 |             # 3. 膜电位重置 (reset by subtraction)
 94 |             v = v - spike * self.v_threshold
 95 | 
 96 |             v_mem_list.append(v)
 97 |             spike_list.append(spike)
 98 | 
 99 |         v_mem_out = torch.stack(v_mem_list, dim=0)
100 |         spikes_out = torch.stack(spike_list, dim=0)
101 |         
102 |         return spikes_out, v_mem_out
103 | 
```

## File: F:\SomeProjects\CSGNN\spikenet_x\__init__.py

- Extension: .py
- Language: python
- Size: 648 bytes
- Created: 2025-08-22 13:06:01
- Modified: 2025-09-15 02:26:11

### Code

```python
 1 | # spikenet_x/__init__.py
 2 | from .masked_ops import (
 3 |     masked_softmax,
 4 |     masked_topk_softmax,
 5 |     topk_mask_logits,
 6 |     fill_masked_,
 7 |     NEG_INF,
 8 | )
 9 | from .rel_time import RelativeTimeEncoding
10 | from .lif_cell import LIFCell
11 | from .new_modules import *
12 | from .spiketdanet_layer import SpikeTDANetLayer
13 | from .model import SpikeTDANet
14 | 
15 | __all__ = [
16 |     # masked_ops
17 |     "masked_softmax", "masked_topk_softmax", "topk_mask_logits", "fill_masked_", "NEG_INF",
18 |     # Core components
19 |     "RelativeTimeEncoding", "LIFCell", "SpikeTDANetLayer", "SpikeTDANet",
20 |     # New modules
21 |     "SpatialGNNWrapper", "DelayLine", "STAGNNAggregator",
22 | ]
```

## File: F:\SomeProjects\CSGNN\spikenet_x\new_modules\delay_line.py

- Extension: .py
- Language: python
- Size: 1646 bytes
- Created: 2025-09-15 02:23:21
- Modified: 2025-09-15 02:23:26

### Code

```python
 1 | # spikenet_x/new_modules/delay_line.py
 2 | import torch
 3 | import torch.nn as nn
 4 | import torch.nn.functional as F
 5 | 
 6 | class DelayLine(nn.Module):
 7 |     """
 8 |     使用因果深度可分离1D卷积，低成本地建模多种时间延迟。
 9 |     输入格式: [T, N, d]
10 |     """
11 |     def __init__(self, channels: int, kernel_size: int = 5):
12 |         super().__init__()
13 |         self.channels = channels
14 |         self.kernel_size = kernel_size
15 |         
16 |         # 因果填充，只在左侧（过去）填充
17 |         self.padding = kernel_size - 1
18 |         
19 |         self.depthwise_conv = nn.Conv1d(
20 |             in_channels=channels, 
21 |             out_channels=channels, 
22 |             kernel_size=kernel_size, 
23 |             padding=self.padding, 
24 |             groups=channels
25 |         )
26 |         self.pointwise_conv = nn.Conv1d(
27 |             in_channels=channels, 
28 |             out_channels=channels, 
29 |             kernel_size=1
30 |         )
31 |         self.activation = nn.GELU()
32 | 
33 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
34 |         """
35 |         Args:
36 |             x (torch.Tensor): 输入特征, 形状 [T, N, d]
37 | 
38 |         Returns:
39 |             torch.Tensor: 经过延迟建模后的特征, 形状 [T, N, d]
40 |         """
41 |         # [T, N, d] -> [N, d, T]
42 |         x_permuted = x.permute(1, 2, 0)
43 |         
44 |         out = self.depthwise_conv(x_permuted)
45 |         out = self.pointwise_conv(out)
46 |         
47 |         # 切片以保持输出长度为T，实现因果性
48 |         out = out[:, :, :x_permuted.size(2)]
49 |         
50 |         out = self.activation(out)
51 |         
52 |         # [N, d, T] -> [T, N, d]
53 |         return out.permute(2, 0, 1)
```

## File: F:\SomeProjects\CSGNN\spikenet_x\new_modules\spatial_gnn_wrapper.py

- Extension: .py
- Language: python
- Size: 1504 bytes
- Created: 2025-09-15 02:23:07
- Modified: 2025-09-15 02:23:11

### Code

```python
 1 | # spikenet_x/new_modules/spatial_gnn_wrapper.py
 2 | import torch
 3 | import torch.nn as nn
 4 | from torch_geometric.nn import SAGEConv
 5 | 
 6 | class SpatialGNNWrapper(nn.Module):
 7 |     """
 8 |     在时序图的每个时间步上高效地应用SAGEConv。
 9 |     输入格式: [T, N, d]
10 |     """
11 |     def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean'):
12 |         super().__init__()
13 |         self.conv = SAGEConv(in_channels, out_channels, aggr=aggr)
14 |         self.activation = nn.GELU()
15 | 
16 |     def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
17 |         """
18 |         Args:
19 |             x (torch.Tensor): 节点特征, 形状 [T, N, d_in]
20 |             edge_index (torch.Tensor): 图的边索引, 形状 [2, E]
21 | 
22 |         Returns:
23 |             torch.Tensor: 空间聚合后的节点特征, 形状 [T, N, d_out]
24 |         """
25 |         T, N, d = x.shape
26 |         # 将 x 重塑为 [T*N, d] 以便进行批处理GNN卷积
27 |         x_reshaped = x.reshape(T * N, d)
28 | 
29 |         # 扩展 edge_index 以匹配 T 个图快照
30 |         # PyG的SAGEConv期望节点索引是全局的，所以我们需要为每个时间步的节点创建偏移
31 |         edge_indices = [edge_index + t * N for t in range(T)]
32 |         edge_index_expanded = torch.cat(edge_indices, dim=1)
33 | 
34 |         out_reshaped = self.conv(x_reshaped, edge_index_expanded)
35 |         out_reshaped = self.activation(out_reshaped)
36 | 
37 |         # 将输出恢复为 [T, N, d_out]
38 |         return out_reshaped.reshape(T, N, -1)
```

## File: F:\SomeProjects\CSGNN\spikenet_x\new_modules\sta_gnn_agg.py

- Extension: .py
- Language: python
- Size: 6051 bytes
- Created: 2025-09-15 02:23:37
- Modified: 2025-09-15 04:00:29

### Code

```python
  1 | # spikenet_x/new_modules/sta_gnn_agg.py
  2 | from __future__ import annotations
  3 | from typing import Optional, Tuple
  4 | import torch
  5 | import torch.nn as nn
  6 | from ..rel_time import RelativeTimeEncoding
  7 | 
  8 | # Helper functions for segment_amax 
  9 | def _has_scatter_reduce_tensor() -> bool:
 10 |     return hasattr(torch.Tensor, "scatter_reduce_")
 11 | 
 12 | def _try_import_torch_scatter():
 13 |     try:
 14 |         import torch_scatter
 15 |         return torch_scatter
 16 |     except Exception:
 17 |         return None
 18 | 
 19 | _TORCH_SCATTER = _try_import_torch_scatter()
 20 | _HAS_TSR = _has_scatter_reduce_tensor()
 21 | 
 22 | def _segment_amax_1d(x: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
 23 |     device, dtype = x.device, x.dtype
 24 |     neg_inf = torch.tensor(-1e30, dtype=dtype, device=device)
 25 |     if _HAS_TSR:
 26 |         init_val = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
 27 |         out = init_val.scatter_reduce(0, index, x, reduce="amax", include_self=False)
 28 |         return out
 29 |     if _TORCH_SCATTER is not None:
 30 |         out, _ = _TORCH_SCATTER.scatter_max(x, index, dim=0, dim_size=K)
 31 |         cnt = torch.zeros(K, device=device, dtype=torch.long)
 32 |         cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
 33 |         out = torch.where(cnt > 0, out, neg_inf)
 34 |         return out
 35 |     raise ImportError("STAGNNAggregator requires either PyTorch >= 1.12 or torch_scatter.")
 36 | 
 37 | 
 38 | class STAGNNAggregator(nn.Module):
 39 |     def __init__(
 40 |         self,
 41 |         d_in: int,
 42 |         d: int,
 43 |         heads: int = 4,
 44 |         W: int = 32,
 45 |         use_rel_bias: bool = True,
 46 |         attn_drop: float = 0.1,
 47 |         temp: float = 1.0,
 48 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
 49 |         pe_n_freq: int = 3,
 50 |     ) -> None:
 51 |         super().__init__()
 52 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
 53 |         self.d_in, self.d, self.heads, self.d_head, self.W = d_in, d, heads, d // heads, W
 54 |         self.use_rel_bias, self.temp, self.p_drop = use_rel_bias, temp, attn_drop
 55 |         
 56 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq)
 57 |         d_pe = self.rel_enc.d_pe
 58 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
 59 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
 60 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
 61 |         self.scale = self.d_head ** -0.5
 62 |         self.reset_parameters()
 63 | 
 64 |     def reset_parameters(self) -> None:
 65 |         nn.init.xavier_uniform_(self.W_q.weight)
 66 |         nn.init.xavier_uniform_(self.W_k.weight)
 67 |         nn.init.xavier_uniform_(self.W_v.weight)
 68 | 
 69 |     def forward(
 70 |         self,
 71 |         H_tilde: torch.Tensor,
 72 |         S: torch.Tensor,
 73 |         edge_index: torch.Tensor,
 74 |         time_idx: torch.Tensor,
 75 |     ) -> torch.Tensor:
 76 |         T, N, _ = H_tilde.shape
 77 |         device, dtype = H_tilde.device, H_tilde.dtype
 78 |         src, dst = edge_index[0].to(device), edge_index[1].to(device)
 79 |         E = src.numel()
 80 | 
 81 |         pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)
 82 |         if not self.use_rel_bias:
 83 |             rel_bias = torch.zeros_like(rel_bias)
 84 | 
 85 |         Q_all = self.W_q(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3)
 86 |         
 87 |         M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
 88 |         eps_gate = 1.0e-6
 89 | 
 90 |         for t in range(T):
 91 |             if E == 0: continue
 92 |             
 93 |             W_eff = min(self.W, t)
 94 |             dt_range = torch.arange(W_eff + 1, device=device)
 95 |             t_prime_range = t - dt_range
 96 | 
 97 |             Q_d_t = Q_all[t, :, dst, :]
 98 |             
 99 |             H_src_window = H_tilde[t_prime_range][:, src, :]
100 |             pe_window = pe_table[:W_eff+1].unsqueeze(1).expand(-1, E, -1)
101 |             
102 |             K_in_flat = torch.cat([H_src_window, pe_window], dim=-1).view(-1, self.d_in + self.rel_enc.d_pe)
103 |             K_s_window = self.W_k(K_in_flat).view(W_eff + 1, E, self.heads, self.d_head).permute(2, 0, 1, 3)
104 |             
105 |             V_s_window = self.W_v(H_src_window).view(W_eff + 1, E, self.heads, self.d_head).permute(2, 0, 1, 3)
106 | 
107 |             gate_log_window = torch.log(S[t_prime_range][:, src] + eps_gate)
108 |             rel_bias_window = rel_bias[:W_eff+1]
109 | 
110 |             scores = torch.einsum('hed,hwed->hwe', Q_d_t, K_s_window) * self.scale
111 |             
112 |             ### --- MODIFICATION START: Replaced in-place additions --- ###
113 |             scores = scores + rel_bias_window.view(1, -1, 1)
114 |             scores = scores + gate_log_window.view(1, -1, E)
115 |             ### --- MODIFICATION END --- ###
116 |             
117 |             if self.temp != 1.0: scores /= self.temp
118 |             
119 |             scores_flat = scores.reshape(self.heads, -1)
120 |             dst_expanded = dst.repeat(W_eff + 1)
121 | 
122 |             max_scores_per_dst = torch.stack([_segment_amax_1d(s, dst_expanded, N) for s in scores_flat])
123 |             max_g = max_scores_per_dst[:, dst_expanded]
124 |             ex = torch.exp(scores_flat - max_g)
125 | 
126 |             if self.training and self.p_drop > 0:
127 |                 drop_mask = (torch.rand_like(ex) > self.p_drop).to(dtype)
128 |                 ### --- MODIFICATION START: Replaced in-place multiplication --- ###
129 |                 ex = ex * drop_mask
130 |                 ### --- MODIFICATION END --- ###
131 | 
132 |             V_flat = V_s_window.reshape(self.heads, -1, self.d_head)
133 |             
134 |             numer_flat = ex.unsqueeze(-1) * V_flat
135 |             numer = torch.zeros((self.heads, N, self.d_head), device=device, dtype=dtype)
136 |             for h in range(self.heads):
137 |                 for c in range(self.d_head):
138 |                     numer[h, :, c].index_add_(0, dst_expanded, numer_flat[h, :, c])
139 | 
140 |             denom = torch.zeros((self.heads, N), device=device, dtype=dtype)
141 |             for h in range(self.heads):
142 |                 denom[h].index_add_(0, dst_expanded, ex[h])
143 |             
144 |             denom = torch.clamp(denom, min=1e-12).unsqueeze(-1)
145 |             msg_h = numer / denom
146 |             M_out[t] = msg_h.permute(1, 0, 2).reshape(N, self.d)
147 | 
148 |         return M_out
```

## File: F:\SomeProjects\CSGNN\spikenet_x\new_modules\sta_gnn_agg_optimized.py

- Extension: .py
- Language: python
- Size: 5589 bytes
- Created: 2025-09-16 22:11:43
- Modified: 2025-09-16 22:33:10

### Code

```python
  1 | # spikenet_x/new_modules/sta_gnn_agg_optimized.py
  2 | from __future__ import annotations
  3 | from typing import Tuple
  4 | import torch
  5 | import torch.nn as nn
  6 | from ..rel_time import RelativeTimeEncoding
  7 | 
  8 | # Helper function to handle vectorized scatter_add for multi-dimensional tensors
  9 | def _segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
 10 |     """Vectorized segment sum using scatter_add_."""
 11 |     result_shape = (num_segments,) + data.shape[1:]
 12 |     result = torch.zeros(result_shape, dtype=data.dtype, device=data.device)
 13 |     # expand segment_ids to match data's dimensions for scatter_add_
 14 |     view_shape = (segment_ids.shape[0],) + (1,) * (data.dim() - 1)
 15 |     segment_ids = segment_ids.view(view_shape).expand_as(data)
 16 |     result.scatter_add_(0, segment_ids, data)
 17 |     return result
 18 | 
 19 | class STAGNNAggregator_Optimized(nn.Module):
 20 |     def __init__(
 21 |         self,
 22 |         d_in: int,
 23 |         d: int,
 24 |         heads: int = 4,
 25 |         W: int = 32,
 26 |         use_rel_bias: bool = True,
 27 |         attn_drop: float = 0.1,
 28 |         temp: float = 1.0,
 29 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
 30 |         pe_n_freq: int = 3,
 31 |     ) -> None:
 32 |         super().__init__()
 33 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
 34 |         self.d_in, self.d, self.heads, self.d_head, self.W = d_in, d, heads, d // heads, W
 35 |         self.use_rel_bias, self.temp, self.p_drop = use_rel_bias, temp, attn_drop
 36 |         
 37 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq)
 38 |         d_pe = self.rel_enc.d_pe
 39 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
 40 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
 41 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
 42 |         self.scale = self.d_head ** -0.5
 43 |         self.reset_parameters()
 44 | 
 45 |     def reset_parameters(self) -> None:
 46 |         nn.init.xavier_uniform_(self.W_q.weight)
 47 |         nn.init.xavier_uniform_(self.W_k.weight)
 48 |         nn.init.xavier_uniform_(self.W_v.weight)
 49 | 
 50 |     def forward(
 51 |         self,
 52 |         H_tilde: torch.Tensor,
 53 |         S: torch.Tensor,
 54 |         edge_index: torch.Tensor,
 55 |         time_idx: torch.Tensor,
 56 |     ) -> torch.Tensor:
 57 |         T, N, _ = H_tilde.shape
 58 |         device, dtype = H_tilde.device, H_tilde.dtype
 59 |         src, dst = edge_index[0].to(device), edge_index[1].to(device)
 60 |         E = src.numel()
 61 |         if E == 0:
 62 |             return torch.zeros((T, N, self.d), device=device, dtype=dtype)
 63 | 
 64 |         pe_table, rel_bias = self.rel_enc(time_idx, W=self.W)
 65 |         if not self.use_rel_bias:
 66 |             rel_bias = torch.zeros_like(rel_bias)
 67 | 
 68 |         # --- Vectorized Index Generation ---
 69 |         t_coords = torch.arange(T, device=device).view(-1, 1)
 70 |         dt_coords = torch.arange(self.W + 1, device=device).view(1, -1)
 71 |         
 72 |         t_prime_matrix = t_coords - dt_coords
 73 |         valid_mask = t_prime_matrix >= 0
 74 |         
 75 |         t_indices, dt_indices = valid_mask.nonzero(as_tuple=True)
 76 |         t_prime_indices = t_prime_matrix[t_indices, dt_indices]
 77 |         
 78 |         num_interactions = len(t_indices)
 79 |         
 80 |         t_indices_exp = t_indices.repeat_interleave(E)
 81 |         dt_indices_exp = dt_indices.repeat_interleave(E)
 82 |         t_prime_indices_exp = t_prime_indices.repeat_interleave(E)
 83 |         
 84 |         src_exp = src.repeat(num_interactions)
 85 |         dst_exp = dst.repeat(num_interactions)
 86 | 
 87 |         # --- Vectorized Attention Calculation ---
 88 |         Q = self.W_q(H_tilde).view(T, N, self.heads, self.d_head)
 89 |         V = self.W_v(H_tilde).view(T, N, self.heads, self.d_head)
 90 | 
 91 |         Q_gathered = Q[t_indices_exp, dst_exp]
 92 |         
 93 |         H_k_gathered = H_tilde[t_prime_indices_exp, src_exp]
 94 |         pe_gathered = pe_table[dt_indices_exp]
 95 |         K_in = torch.cat([H_k_gathered, pe_gathered], dim=-1)
 96 |         K_gathered = self.W_k(K_in).view(-1, self.heads, self.d_head)
 97 | 
 98 |         V_gathered = V[t_prime_indices_exp, src_exp]
 99 | 
100 |         scores = torch.einsum('ehd,ehd->eh', Q_gathered, K_gathered) * self.scale
101 |         
102 |         # 使用非原地操作 (out-of-place)
103 |         scores = scores + rel_bias[dt_indices_exp].unsqueeze(-1)
104 |         
105 |         eps_gate = 1e-6
106 |         spike_gate = S[t_prime_indices_exp, src_exp]
107 |         
108 |         spike_gate_per_head = spike_gate.view(-1, self.heads, self.d_head)
109 |         scalar_gate_per_head = spike_gate_per_head.mean(dim=-1)
110 |         scores = scores + torch.log(scalar_gate_per_head + eps_gate)
111 |         
112 |         if self.temp != 1.0:
113 |             scores = scores / self.temp
114 |             
115 |         # 5. Numerically stable softmax (segment-wise)
116 |         segment_ids = t_indices_exp * N + dst_exp
117 |         num_segments = T * N
118 |         
119 |         max_scores = torch.full((num_segments, self.heads), -1e30, device=device, dtype=dtype)
120 |         max_scores.scatter_reduce_(0, segment_ids.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=False)
121 | 
122 |         scores_normalized = torch.exp(scores - max_scores[segment_ids])
123 |         
124 |         if self.training and self.p_drop > 0:
125 | 
126 |             scores_normalized = scores_normalized * (torch.rand_like(scores_normalized) > self.p_drop)
127 | 
128 |         denom = _segment_sum(scores_normalized, segment_ids, num_segments)
129 |         
130 |         numer_contrib = scores_normalized.unsqueeze(-1) * V_gathered
131 |         numer = _segment_sum(numer_contrib, segment_ids, num_segments)
132 |         
133 |         M_flat = numer / torch.clamp(denom, min=1e-12).unsqueeze(-1)
134 |         M_out = M_flat.reshape(T, N, self.d)
135 |         
136 |         return M_out
```

## File: F:\SomeProjects\CSGNN\spikenet_x\new_modules\__init__.py

- Extension: .py
- Language: python
- Size: 355 bytes
- Created: 2025-09-15 02:22:47
- Modified: 2025-09-16 22:12:53

### Code

```python
 1 | # spikenet_x/new_modules/__init__.py
 2 | from .spatial_gnn_wrapper import SpatialGNNWrapper
 3 | from .delay_line import DelayLine
 4 | from .sta_gnn_agg import STAGNNAggregator
 5 | from .sta_gnn_agg_optimized import STAGNNAggregator_Optimized
 6 | 
 7 | __all__ = [
 8 |     "SpatialGNNWrapper",
 9 |     "DelayLine",
10 |     "STAGNNAggregator", 
11 |     "STAGNNAggregator_Optimized", 
12 | ]
```


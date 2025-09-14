# Table of Contents
- F:\SomeProjects\CSGNN\.gitignore
- F:\SomeProjects\CSGNN\generate_feature.py
- F:\SomeProjects\CSGNN\LICENSE
- F:\SomeProjects\CSGNN\main.py
- F:\SomeProjects\CSGNN\main_static.py
- F:\SomeProjects\CSGNN\README.md
- F:\SomeProjects\CSGNN\setup.py
- F:\SomeProjects\CSGNN\提示词.md
- F:\SomeProjects\CSGNN\cline_docs\activeContext.md
- F:\SomeProjects\CSGNN\cline_docs\productContext.md
- F:\SomeProjects\CSGNN\cline_docs\progress.md
- F:\SomeProjects\CSGNN\cline_docs\systemPatterns.md
- F:\SomeProjects\CSGNN\cline_docs\techContext.md
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
- F:\SomeProjects\CSGNN\spikenet_x\sta.py
- F:\SomeProjects\CSGNN\spikenet_x\sta_sparse.py
- F:\SomeProjects\CSGNN\spikenet_x\__init__.py

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
- Size: 19761 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2025-09-14 19:45:03

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
 13 | from spikenet_x.model import SpikeNetX
 14 | from texttable import Texttable # Added for tab_printer
 15 | import numpy as np # Added for set_seed
 16 | 
 17 | 
 18 | def sample_subgraph(nodes: torch.Tensor, edge_index_full: torch.Tensor, num_neighbors: int = -1):
 19 |     """
 20 |     返回：
 21 |       subgraph_nodes: 子图包含的全局节点 id，形状 [N_sub]（升序、无重复）
 22 |       subgraph_edge_index: 子图边（局部 id），形状 [2, E_sub]，值域 ∈ [0, N_sub-1]
 23 |       nodes_local_index: 种子在子图中的局部索引，形状 [B]
 24 |     """
 25 |     row, col = edge_index_full
 26 |     device = row.device
 27 |     nodes = nodes.to(device)
 28 | 
 29 |     # 1) 收集邻居（全收或按预算随机下采样）
 30 |     mask = torch.isin(row, nodes)
 31 |     neigh_all = col[mask]
 32 |     if num_neighbors == -1 or neigh_all.numel() <= nodes.numel() * max(num_neighbors, 0):
 33 |         neighbors = neigh_all
 34 |     else:
 35 |         target = nodes.numel() * int(num_neighbors)
 36 |         perm = torch.randperm(neigh_all.numel(), device=device)[:target]
 37 |         neighbors = neigh_all[perm]
 38 | 
 39 |     # 2) 子图节点集（仍是全局 id）
 40 |     subgraph_nodes = torch.unique(torch.cat([nodes, neighbors], dim=0))
 41 |     subgraph_nodes_sorted, _ = torch.sort(subgraph_nodes)
 42 | 
 43 |     # 3) **关键**：仅保留“子图内部”的边，然后把 (全局 id) → (局部 id)
 44 |     mask_src = torch.isin(row, subgraph_nodes_sorted)
 45 |     mask_dst = torch.isin(col, subgraph_nodes_sorted)
 46 |     edge_mask = mask_src & mask_dst
 47 |     e_global = edge_index_full[:, edge_mask]  # [2, E_sub_global]
 48 | 
 49 |     # 全局→局部：利用 searchsorted（要求 subgraph_nodes_sorted 升序）
 50 |     src_local = torch.searchsorted(subgraph_nodes_sorted, e_global[0])
 51 |     dst_local = torch.searchsorted(subgraph_nodes_sorted, e_global[1])
 52 |     subgraph_edge_index = torch.stack([src_local, dst_local], dim=0)
 53 | 
 54 |     # 4) 种子在子图中的局部位置
 55 |     nodes_local_index = torch.searchsorted(subgraph_nodes_sorted, nodes)
 56 | 
 57 |     return subgraph_nodes_sorted, subgraph_edge_index, nodes_local_index
 58 | 
 59 | 
 60 | def set_seed(seed):
 61 |     np.random.seed(seed)
 62 |     torch.manual_seed(seed)
 63 |     torch.cuda.manual_seed(seed)
 64 | 
 65 | def tab_printer(args):
 66 |     """Function to print the logs in a nice tabular format."""
 67 |     args = vars(args)
 68 |     keys = sorted(args.keys())
 69 |     t = Texttable() 
 70 |     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
 71 |     print(t.draw())
 72 | 
 73 | 
 74 | class SpikeNet(nn.Module):
 75 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
 76 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 77 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):
 78 | 
 79 |         super().__init__()
 80 |         
 81 |         from spikenet.utils import RandomWalkSampler, Sampler, add_selfloops
 82 |         
 83 |         tau = 1.0
 84 |         if sampler == 'rw':
 85 |             self.sampler = [RandomWalkSampler(
 86 |                 add_selfloops(adj_matrix)) for adj_matrix in data.adj]
 87 |             self.sampler_t = [RandomWalkSampler(add_selfloops(
 88 |                 adj_matrix)) for adj_matrix in data.adj_evolve]
 89 |         elif sampler == 'sage':
 90 |             self.sampler = [Sampler(add_selfloops(adj_matrix))
 91 |                             for adj_matrix in data.adj]
 92 |             self.sampler_t = [Sampler(add_selfloops(adj_matrix))
 93 |                               for adj_matrix in data.adj_evolve]
 94 |         else:
 95 |             raise ValueError(sampler)
 96 | 
 97 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 98 | 
 99 |         for hid in hids:
100 |             aggregators.append(SAGEAggregator(in_features, hid,
101 |                                               concat=concat, bias=bias,
102 |                                               aggr=aggr))
103 | 
104 |             if act == "IF":
105 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
106 |             elif act == 'LIF':
107 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
108 |             elif act == 'PLIF':
109 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
110 |             else:
111 |                 raise ValueError(act)
112 | 
113 |             in_features = hid * 2 if concat else hid
114 | 
115 |         self.aggregators = aggregators
116 |         self.dropout = nn.Dropout(dropout)
117 |         self.snn = snn
118 |         self.sizes = sizes
119 |         self.p = p
120 |         self.pooling = nn.Linear(len(data) * in_features, out_features)
121 | 
122 |     def encode(self, nodes):
123 |         spikes = []
124 |         sizes = self.sizes
125 |         for time_step in range(len(data)):
126 | 
127 |             snapshot = data[time_step]
128 |             sampler = self.sampler[time_step]
129 |             sampler_t = self.sampler_t[time_step]
130 | 
131 |             x = snapshot.x
132 |             h = [x[nodes].to(device)]
133 |             num_nodes = [nodes.size(0)]
134 |             nbr = nodes
135 |             for size in sizes:
136 |                 size_1 = max(int(size * self.p), 1)
137 |                 size_2 = size - size_1
138 | 
139 |                 if size_2 > 0:
140 |                     nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
141 |                     nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
142 |                     nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
143 |                 else:
144 |                     nbr = sampler(nbr, size_1).view(-1)
145 | 
146 |                 num_nodes.append(nbr.size(0))
147 |                 h.append(x[nbr].to(device))
148 | 
149 |             for i, aggregator in enumerate(self.aggregators):
150 |                 self_x = h[:-1]
151 |                 neigh_x = []
152 |                 for j, n_x in enumerate(h[1:]):
153 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
154 | 
155 |                 out = self.snn[i](aggregator(self_x, neigh_x))
156 |                 if i != len(sizes) - 1:
157 |                     out = self.dropout(out)
158 |                     h = torch.split(out, num_nodes[:-(i + 1)])
159 | 
160 |             spikes.append(out)
161 |         spikes = torch.cat(spikes, dim=1)
162 |         neuron.reset_net(self)
163 |         return spikes
164 | 
165 |     def forward(self, nodes):
166 |         spikes = self.encode(nodes)
167 |         return self.pooling(spikes)
168 | 
169 | 
170 | parser = argparse.ArgumentParser()
171 | parser.add_argument("--model", nargs="?", default="spikenet",
172 |                     help="Model to use ('spikenet', 'spikenetx'). (default: spikenet)")
173 | parser.add_argument("--dataset", nargs="?", default="DBLP",
174 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
175 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
176 | parser.add_argument('--hids', type=int, nargs='+',
177 |                     default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
178 | parser.add_argument("--aggr", nargs="?", default="mean",
179 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
180 | parser.add_argument("--sampler", nargs="?", default="sage",
181 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
182 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
183 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
184 | parser.add_argument("--neuron", nargs="?", default="LIF",
185 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
186 | parser.add_argument('--batch_size', type=int, default=1024,
187 |                     help='Batch size for training. (default: 1024)')
188 | parser.add_argument('--lr', type=float, default=5e-3,
189 |                     help='Learning rate for training. (default: 5e-3)')
190 | parser.add_argument('--train_size', type=float, default=0.4,
191 |                     help='Ratio of nodes for training. (default: 0.4)')
192 | parser.add_argument('--alpha', type=float, default=1.0,
193 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
194 | parser.add_argument('--p', type=float, default=0.5,
195 |                     help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
196 | parser.add_argument('--dropout', type=float, default=0.7,
197 |                     help='Dropout probability. (default: 0.7)')
198 | parser.add_argument('--epochs', type=int, default=100,
199 |                     help='Number of training epochs. (default: 100)')
200 | parser.add_argument('--concat', action='store_true',
201 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
202 | parser.add_argument('--seed', type=int, default=2022,
203 |                     help='Random seed for model. (default: 2022)')
204 | parser.add_argument('--datapath', type=str, default='./data',
205 |                     help='Wheres your data?, Default is ./data')
206 | 
207 | # SpikeNet-X specific args
208 | parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for SpikeNet-X. (default: 4)')
209 | parser.add_argument('--topk', type=int, default=8, help='Top-k neighbors for SpikeNet-X attention. (default: 8)')
210 | parser.add_argument('--W', type=int, default=8, help='Time window size for SpikeNet-X. (default: 8)')
211 | parser.add_argument('--attn_impl', type=str, default='dense', choices=['dense','sparse'],
212 |                     help='Attention kernel for SpikeNet-X: "dense" (fallback, supports top-k) or "sparse". (default: "dense")')
213 | 
214 | 
215 | # 新增：模型保存、加载与测试参数
216 | parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
217 |                     help='Directory to save model checkpoints. (default: checkpoints)')
218 | parser.add_argument('--resume_path', type=str, default=None,
219 |                     help='Path to a checkpoint file to resume training from. (default: None)')
220 | parser.add_argument('--test_model_path', type=str, default=None,
221 |                     help='Path to a model file to load for testing only. (default: None)')
222 | 
223 | 
224 | try:
225 |     args = parser.parse_args()
226 |     args.test_size = 1 - args.train_size
227 |     args.train_size = args.train_size - 0.05
228 |     args.val_size = 0.05
229 |     args.split_seed = 42
230 |     tab_printer(args)
231 | except:
232 |     parser.print_help()
233 |     exit(0)
234 | 
235 | assert len(args.hids) == len(args.sizes), "must be equal!"
236 | 
237 | if args.dataset.lower() == "dblp":
238 |     data = dataset.DBLP(root = args.datapath)
239 | elif args.dataset.lower() == "tmall":
240 |     data = dataset.Tmall(root = args.datapath)
241 | elif args.dataset.lower() == "patent":
242 |     data = dataset.Patent(root = args.datapath)
243 | else:
244 |     raise ValueError(
245 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
246 | 
247 | # train:val:test
248 | data.split_nodes(train_size=args.train_size, val_size=args.val_size,
249 |                  test_size=args.test_size, random_state=args.split_seed)
250 | 
251 | set_seed(args.seed)
252 | 
253 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
254 | 
255 | y = data.y.to(device)
256 | 
257 | train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
258 | val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
259 |                         pin_memory=False, batch_size=200000, shuffle=False)
260 | test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
261 | 
262 | if args.model == 'spikenetx':
263 |     
264 |     def train_spikenetx():
265 |         model.train()
266 |         total_loss = 0
267 |         # Let's use a fixed number of neighbors for now to control memory
268 |         # 现为25， -1 为全邻居
269 |         num_neighbors_to_sample = 25 
270 |         for nodes in tqdm(train_loader, desc='Training'):
271 |             nodes = nodes.to(device)
272 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
273 |             
274 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
275 |             # --- 强校验：子图边必须是局部 id，且与特征 N 一致 ---
276 |             
277 |             N_sub = subgraph_nodes.numel()
278 |             assert subgraph_edge_index.dtype == torch.long, "subgraph_edge_index 必须是 torch.long（int64）"
279 |             assert subgraph_edge_index.numel() > 0, "子图没有边（可能邻居采样太小或图太稀疏）"
280 |             assert int(subgraph_edge_index.max().item()) < N_sub and int(subgraph_edge_index.min().item()) >= 0, \
281 |                 f"边索引越界：[{int(subgraph_edge_index.min())}, {int(subgraph_edge_index.max())}]，但 N_sub={N_sub}"
282 |             # H0_subgraph 形状应为 [T, N_sub, d_in]
283 |             assert H0_subgraph.size(1) == N_sub, f"H0_subgraph 第二维应等于 N_sub，但拿到 {H0_subgraph.size()} vs N_sub={N_sub}"
284 | 
285 |             optimizer.zero_grad()
286 |             
287 |             # The model's output `repr` and `logits` are for all nodes in the subgraph
288 |             output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
289 |             subgraph_logits = output['logits']
290 | 
291 |             # We only compute the loss on the seed nodes of the batch
292 |             loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])
293 |             
294 |             loss.backward()
295 |             optimizer.step()
296 |             total_loss += loss.item()
297 |         return total_loss / len(train_loader)
298 | 
299 |     @torch.no_grad()
300 |     def test_spikenetx(loader):
301 |         model.eval()
302 |         logits_list = []
303 |         labels_list = []
304 |         num_neighbors_to_sample = 10 # Use the same for testing
305 |         for nodes in tqdm(loader, desc='Testing'):
306 |             nodes = nodes.to(device)
307 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
308 |             
309 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
310 |             
311 |             output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
312 |             subgraph_logits = output['logits']
313 |             
314 |             logits_list.append(subgraph_logits[nodes_local_index].cpu())
315 |             labels_list.append(y[nodes].cpu())
316 |         
317 |         logits = torch.cat(logits_list, dim=0).argmax(1)
318 |         labels = torch.cat(labels_list, dim=0)
319 | 
320 |         micro = metrics.f1_score(labels, logits, average='micro', zero_division=0)
321 |         macro = metrics.f1_score(labels, logits, average='macro', zero_division=0)
322 |         return macro, micro
323 |     
324 |     # --- SpikeNet-X Training and Evaluation (with batching) ---
325 | 
326 |     # 1. Data Preparation (Full graph data)
327 |     print("Preparing data for SpikeNet-X...")
328 |     T = len(data)
329 |     N = data.num_nodes
330 |     d_in = data.num_features
331 |     
332 |     edge_list = [snapshot.edge_index for snapshot in data]
333 |     edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(device)
334 |     H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(device)
335 |     time_idx_full = torch.arange(T, device=device)
336 | 
337 |     # 2. Model, Optimizer, Loss
338 |     model = SpikeNetX(
339 |         d_in=d_in,
340 |         d=args.hids[0],
341 |         layers=len(args.sizes),
342 |         heads=args.heads,
343 |         out_dim=data.num_classes,
344 |         topk=args.topk,
345 |         W=args.W,
346 |         attn_impl=args.attn_impl,
347 |         readout="mean",  # ← 新增：用时间平均读出
348 |     ).to(device)
349 | 
350 |     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
351 |     loss_fn = nn.CrossEntropyLoss()
352 |     
353 |     
354 |     # --- Test-only mode ---
355 |     if args.test_model_path:
356 |         print(f"Loading model from {args.test_model_path} for testing...")
357 |         checkpoint = torch.load(args.test_model_path, map_location=device)
358 |         model.load_state_dict(checkpoint['model_state_dict'])
359 |         test_macro, test_micro = test_spikenetx(test_loader)
360 |         print(f"Test Results: Macro-F1={test_macro:.4f}, Micro-F1={test_micro:.4f}")
361 |         exit(0)
362 | 
363 | 
364 | 
365 |     # 3. Training Loop
366 |     start_epoch = 1
367 |     best_val_metric = 0
368 |     best_test_metric = (0, 0)
369 | 
370 |     # --- Resume from checkpoint ---
371 |     if args.resume_path:
372 |         if os.path.exists(args.resume_path):
373 |             print(f"Resuming training from {args.resume_path}...")
374 |             checkpoint = torch.load(args.resume_path, map_location=device)
375 |             model.load_state_dict(checkpoint['model_state_dict'])
376 |             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
377 |             start_epoch = checkpoint['epoch'] + 1
378 |             best_val_metric = checkpoint.get('best_val_metric', 0) # Use .get for backward compatibility
379 |             print(f"Resumed from epoch {start_epoch-1}. Best val metric so far: {best_val_metric:.4f}")
380 |         else:
381 |             print(f"Warning: Checkpoint path {args.resume_path} not found. Starting from scratch.")
382 | 
383 |     print("Starting SpikeNet-X training...")
384 |     start = time.time()
385 |     for epoch in range(start_epoch, args.epochs + 1):
386 |         train_spikenetx()
387 |         val_metric = test_spikenetx(val_loader)
388 |         test_metric = test_spikenetx(test_loader)
389 |         
390 |         is_best = val_metric[1] > best_val_metric
391 |         if is_best:
392 |             best_val_metric = val_metric[1]
393 |             best_test_metric = test_metric
394 | 
395 |             # --- Save checkpoint ---
396 |             os.makedirs(args.checkpoint_dir, exist_ok=True)
397 |             checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.dataset}.pth')
398 |             torch.save({
399 |                 'epoch': epoch,
400 |                 'model_state_dict': model.state_dict(),
401 |                 'optimizer_state_dict': optimizer.state_dict(),
402 |                 'best_val_metric': best_val_metric,
403 |                 'test_metric_at_best_val': test_metric,
404 |             }, checkpoint_path)
405 |             print(f"Epoch {epoch:03d}: New best model saved to {checkpoint_path} with Val Micro: {best_val_metric:.4f}")
406 | 
407 |         end = time.time()
408 |         print(
409 |             f'Epoch: {epoch:03d}, Val Micro: {val_metric[1]:.4f}, Test Micro: {test_metric[1]:.4f}, '
410 |             f'Best Test: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time: {end-start:.2f}s'
411 |         )
412 | 
413 | else:
414 |     # --- Original SpikeNet Training and Evaluation ---
415 |     model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
416 |                      dropout=args.dropout, sampler=args.sampler, p=args.p,
417 |                      aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
418 |                      hids=args.hids, act=args.neuron, bias=True).to(device)
419 | 
420 |     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
421 |     loss_fn = nn.CrossEntropyLoss()
422 | 
423 |     def train():
424 |         model.train()
425 |         for nodes in tqdm(train_loader, desc='Training'):
426 |             optimizer.zero_grad()
427 |             loss_fn(model(nodes), y[nodes]).backward()
428 |             optimizer.step()
429 | 
430 |     @torch.no_grad()
431 |     def test(loader):
432 |         model.eval()
433 |         logits = []
434 |         labels = []
435 |         for nodes in loader:
436 |             logits.append(model(nodes))
437 |             labels.append(y[nodes])
438 |         logits = torch.cat(logits, dim=0).cpu()
439 |         labels = torch.cat(labels, dim=0).cpu()
440 |         logits = logits.argmax(1)
441 |         metric_macro = metrics.f1_score(labels, logits, average='macro')
442 |         metric_micro = metrics.f1_score(labels, logits, average='micro')
443 |         return metric_macro, metric_micro
444 | 
445 |     best_val_metric = test_metric = 0
446 |     start = time.time()
447 |     for epoch in range(1, args.epochs + 1):
448 |         train()
449 |         val_metric, test_metric = test(val_loader), test(test_loader)
450 |         if val_metric[1] > best_val_metric:
451 |             best_val_metric = val_metric[1]
452 |             best_test_metric = test_metric
453 |         end = time.time()
454 |         print(
455 |             f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
456 | 
457 |     # save bianry node embeddings (spikes)
458 |     # emb = model.encode(torch.arange(data.num_nodes)).cpu()
459 |     # torch.save(emb, 'emb.pth')
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

## File: F:\SomeProjects\CSGNN\README.md

- Extension: .md
- Language: markdown
- Size: 8506 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2025-09-14 15:06:45

### Code

```markdown
  1 | # Abstract
  2 | 
  3 | Recent years have seen a surge in research on dynamic graph representation learning, which aims to model temporal graphs that are dynamic and evolving constantly over time. However, current work typically models graph dynamics with recurrent neural networks (RNNs), making them suffer seriously from computation and memory overheads on large temporal graphs. So far, scalability of dynamic graph representation learning on large temporal graphs remains one of the major challenges. In this paper, we present a scalable framework, namely SpikeNet, to efficiently capture the temporal and structural patterns of temporal graphs. We explore a new direction in that we can capture the evolving dynamics of temporal graphs with spiking neural networks (SNNs) instead of RNNs. As a low-power alternative to RNNs, SNNs explicitly model graph dynamics as spike trains of neuron populations and enable spike-based propagation in an efficient way. Experiments on three large real-world temporal graph datasets demonstrate that SpikeNet outperforms strong baselines on the temporal node classification task with lower computational costs. Particularly, SpikeNet generalizes to a large temporal graph (2M nodes and 13M edges) with significantly fewer parameters and computation overheads.
  4 | 
  5 | # Dataset
  6 | 
  7 | ## Overview
  8 | 
  9 | |             | DBLP    | Tmall     | Patent     |
 10 | | ----------- | ------- | --------- | ---------- |
 11 | | #nodes      | 28,085  | 577,314   | 2,738,012  |
 12 | | #edges      | 236,894 | 4,807,545 | 13,960,811 |
 13 | | #time steps | 27      | 186       | 25         |
 14 | | #classes    | 10      | 5         | 6          |
 15 | 
 16 | ## Download datasets
 17 | 
 18 | + DBLP
 19 | + Tmall
 20 | + Patent
 21 | 
 22 | All dataset can be found at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0).
 23 | You can download the datasets and put them in the folder `data/`, e.g., `data/dblp`.
 24 | 
 25 | ## (Optional) Re-generate node features via DeepWalk
 26 | 
 27 | Since these datasets have no associated node features, we have generated node features via unsupervised DeepWalk method (saved as `.npy` format).
 28 | You can find them at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0) as well.
 29 | Only `dblp.npy` is uploaded due to size limit of Dropbox.
 30 | 
 31 | (Update) The generated node features for Tmall and Patent datasets have been shared through Aliyun Drive, and the link is as follows: https://www.aliyundrive.com/s/LH9qa9XZmXa.
 32 | 
 33 | Note: Since Aliyun Drive does not support direct sharing of npy files, you will need to manually change the file extension `.txt` to `.npy` after downloading.
 34 | 
 35 | We also provide the script to generate the node features. Alternatively, you can generate them on your end (this will take about minutes to hours):
 36 | 
 37 | ```bash
 38 | python generate_feature.py --dataset dblp
 39 | python generate_feature.py --dataset tmall --normalize
 40 | python generate_feature.py --dataset patent --normalize
 41 | ```
 42 | 
 43 | ## Overall file structure
 44 | 
 45 | ```bash
 46 | SpikeNet
 47 | ├── data
 48 | │   ├── dblp
 49 | │   │   ├── dblp.npy
 50 | │   │   ├── dblp.txt
 51 | │   │   └── node2label.txt
 52 | │   ├── tmall
 53 | │   │   ├── tmall.npy
 54 | │   │   └── tmall.txt
 55 | │   │   ├── node2label.txt
 56 | │   ├── patent
 57 | │   │   ├── patent_edges.json
 58 | │   │   ├── patent_nodes.json
 59 | │   │   └── patent.npy
 60 | ├── figs
 61 | │   └── spikenet.png
 62 | ├── spikenet
 63 | │   ├── ...
 64 | ├── spikenet_x
 65 | │   ├── __init__.py
 66 | │   ├── delayline.py
 67 | │   ├── lif_cell.py
 68 | │   ├── masked_ops.py
 69 | │   ├── minimal_example.py
 70 | │   ├── model.py
 71 | │   ├── rel_time.py
 72 | │   ├── spikenetx_layer.py
 73 | │   └── sta.py
 74 | ├── generate_feature.py
 75 | ├── main.py
 76 | ├── main_static.py
 77 | ├── README.md
 78 | ├── setup.py
 79 | ```
 80 | 
 81 | # Requirements
 82 | 
 83 | ```
 84 | gensim==4.2.0
 85 | numba==0.61.2
 86 | numpy==1.25.2
 87 | scikit_learn==1.1.3
 88 | scipy==1.16.2
 89 | setuptools==68.2.2
 90 | texttable==1.7.0
 91 | torch==1.13.0+cu117
 92 | torch_cluster==1.6.3
 93 | torch_geometric==2.6.1
 94 | torch_scatter==2.1.0+pt113cu117
 95 | tqdm==4.67.1
 96 | ```
 97 | 
 98 | In fact, the version of these packages does not have to be consistent to ours. For example, Pytorch 1.6~-1.12 should also work.
 99 | 
100 | # Usage
101 | 
102 | ## Build neighborhood sampler
103 | 
104 | ```bash
105 | python setup.py install
106 | ```
107 | 
108 | ## Run SpikeNet
109 | 
110 | ```bash
111 | # DBLP
112 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4
113 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.6
114 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.8
115 | 
116 | # Tmall
117 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.4
118 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.6
119 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.8
120 | 
121 | # Patent
122 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.4
123 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.6
124 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 0.5 --train_size 0.8
125 | ```
126 | 
127 | ## Run SpikeNet-X
128 | 
129 | SpikeNet-X is a new model variant that uses temporal attention. You can run it by specifying `--model spikenetx`.
130 | 
131 | ### Basic Training
132 | 
133 | This command starts a standard training run from scratch
134 | 
135 | ```bash
136 | # DBLP Example
137 | python main.py --model spikenetx --dataset dblp --hids 64 --sizes 2 --epochs 100 --lr 0.005 --heads 4 --topk 8 --W 8
138 | ```
139 | 
140 | ### Training Workflow (with Checkpoints)
141 | 
142 | The training script now supports saving, resuming, and testing.
143 | 
144 | **1. Standard Training & Saving**
145 | 
146 | When you run a standard training, the script will automatically save the model with the best validation performance to the `checkpoints/` directory (or the directory specified by `--checkpoint_dir`).
147 | 
148 | ```bash
149 | # The best model will be saved as checkpoints/best_model_DBLP.pth
150 | python main.py --model spikenetx --dataset DBLP --epochs 100
151 | ```
152 | 
153 | **2. Resuming Training**
154 | 
155 | If your training was interrupted, you can resume from the last saved checkpoint using the `--resume_path` argument.
156 | 
157 | ```bash
158 | # This will load the model, optimizer, and epoch number from the checkpoint and continue training.
159 | python main.py --model spikenetx --dataset DBLP --epochs 100 --resume_path checkpoints/best_model_DBLP.pth
160 | ```
161 | 
162 | **3. Testing a Model**
163 | 
164 | To evaluate a trained model on the test set without running the full training loop, use the `--test_model_path` argument.
165 | 
166 | ```bash
167 | # This will load the model, run evaluation on the test set, and print the results.
168 | python main.py --model spikenetx --dataset DBLP --test_model_path checkpoints/best_model_DBLP.pth
169 | ```
170 | 
171 | # On the extention to stastic graphs
172 | 
173 | Actually, SpikeNet is not only applicaple for temporal graphs, it is also straightforward to extend to stastic graphs by defining a time step hyperparameter $T$ manually.
174 | In this way, the sampled subgraph at each time step naturally form graph snapshot. We can use SpikeNet to capture the *evolving* dynamics of sampled subgraphs.
175 | Due to space limit, we did not discuss this part in our paper. However, we believe this is indeed necessary to show the effectiveness of our work.
176 | 
177 | We provide a simple example for the usage on stastic graphs datasets `Flickr` and `Reddit` (be sure you have PyTorch Geometric installed):
178 | 
179 | ```bash
180 | # Flickr
181 | python main_static.py --dataset flickr --surrogate super
182 | 
183 | # Reddit
184 | python main_static.py --dataset reddit --surrogate super
185 | ```
186 | 
187 | We report Micro-F1 score and the results are as follows:
188 | 
189 | | Method     | Flickr       | Reddit       |
190 | | ---------- | ------------ | ------------ |
191 | | GCN        | 0.492±0.003 | 0.933±0.000 |
192 | | GraphSAGE  | 0.501±0.013 | 0.953±0.001 |
193 | | FastGCN    | 0.504±0.001 | 0.924±0.001 |
194 | | S-GCN      | 0.482±0.003 | 0.964±0.001 |
195 | | AS-GCN     | 0.504±0.002 | 0.958±0.001 |
196 | | ClusterGCN | 0.481±0.005 | 0.954±0.001 |
197 | | GraphSAINT | 0.511±0.001 | 0.966±0.001 |
198 | | SpikeNet   | 0.515±0.003 | 0.953±0.001 |
199 | 
200 | # Reference
201 | 
202 | ```bibtex
203 | @inproceedings{li2023scaling,
204 |   author    = {Jintang Li and
205 |                Zhouxin Yu and
206 |                Zulun Zhu and
207 |                Liang Chen and
208 |                Qi Yu and
209 |                Zibin Zheng and
210 |                Sheng Tian and
211 |                Ruofan Wu and
212 |                Changhua Meng},
213 |   title     = {Scaling Up Dynamic Graph Representation Learning via Spiking Neural
214 |                Networks},
215 |   booktitle = {{AAAI}},
216 |   pages     = {8588--8596},
217 |   publisher = {{AAAI} Press},
218 |   year      = {2023}
219 | }
220 | ```
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

## File: F:\SomeProjects\CSGNN\提示词.md

- Extension: .md
- Language: markdown
- Size: 14446 bytes
- Created: 2025-08-22 12:38:17
- Modified: 2025-08-22 12:38:19

### Code

```markdown
  1 | # SpikeNet-X 技术规格（实现提示词）
  2 | 
  3 | ## 0. 目标与总览
  4 | **目标**：在动态图/时序图上，用事件驱动的**脉冲时序注意力聚合（STA）**替换传统时间池化，并在其前端串联**可学习多延迟通路（DelayLine）**，显式建模关系传播的**时间错位与传播时延**，保持 SNN 的稀疏事件驱动与可解释性。
  5 | 
  6 | **层级结构（自上而下）**  
  7 | 1) **DelayLine**：沿时间维的**因果深度可分离 1D 卷积**（K 个离散延迟 tap），学习不同延迟的响应；  
  8 | 2) **STA-GNN Aggregator**：对邻居在过去时间的事件序列进行**因果多头注意力**，以**相对时间编码**与**脉冲门控**（surrogate）做软选择，并用 **Top-k 稀疏化**控制成本；  
  9 | 3) **脉冲单元（LIF/GLIF）**：将上一步的聚合消息注入膜电位，阈值比较产生脉冲，使用代理梯度反传。
 10 | 
 11 | ---
 12 | 
 13 | ## 1. 记号与输入输出约定
 14 | 
 15 | - 时长 \(T\)、节点数 \(N\)、特征维 \(d_\text{in}\)、隐藏维 \(d\)、注意力头数 \(H\)、延迟 tap 数 \(K\)、相对时间编码维 \(d_\text{pe}\)。  
 16 | - 默认采用**时间优先格式**：`H_tn` 表示 `[T, N, d]`；脉冲张量 `S` 为 `[T, N]`（0/1 或 {0,1} 的浮点）。  
 17 | - 图结构用 `edge_index: LongTensor[2, E]`（PyG 风格），或邻接稀疏矩阵。若使用稠密掩码 `adj_mask: Bool[N, N]` 亦可。  
 18 | - **Batch**：建议将多图拼成大图，用 `batch: LongTensor[N_total]` 指示归属；若必须 `B×N×T×d`，可在外层再套一维 batch 并展开为大图。  
 19 | - **模块 API（核心）**：
 20 |   - `LearnableDelayLine.forward(H: [T, N, d_in]) -> H̃: [T, N, d_in]`
 21 |   - `SpikingTemporalAttention.forward(H̃: [T,N,d_qkv], S: [T,N], edge_index or adj_mask, time_idx: [T]) -> M: [T,N,d]`
 22 |   - `LIFCell.forward(M: [T,N,d], state0) -> (S: [T,N], V: [T,N], aux)`  
 23 |   其中 `d_qkv` 与 `d` 可相同或通过线性层投影。
 24 | 
 25 | ---
 26 | 
 27 | ## 2. DelayLine（可学习多延迟通路）
 28 | 
 29 | ### 原理
 30 | 对每个节点、每个通道沿时间做**因果卷积**，学习对不同**传播时延**的权重偏好。  
 31 | \[
 32 | \tilde{h}_{t}^{(c)}=\sum_{k=0}^{K-1} w_k^{(c)} \cdot h_{t-k}^{(c)},\quad 
 33 | w_k^{(c)}=\frac{\text{softplus}(u_k^{(c)})}{\sum_r \text{softplus}(u_r^{(c)})}\cdot \rho^k,\ \rho\in(0,1)
 34 | \]
 35 | - `softplus`+**归一化**保证稳定、可解释（权重非负，且随延迟指数折扣 \(\rho^k\)）。  
 36 | - 支持**通道共享**（每层统一一组 \(\{w_k\}\)）或**逐通道**（推荐：逐通道但分组实现）。
 37 | 
 38 | ### 接口与形状
 39 | - 入：`H: Float[T, N, d_in]`  
 40 | - 出：`H_tilde: Float[T, N, d_in]`（与输入同形）  
 41 | - 可选参数：`K=3~7`，`rho=0.85`，`per_channel=True`，`causal_pad='left'`。
 42 | 
 43 | ### 计算与实现要点
 44 | - **实现**：等价 `groups=d_in` 的 `Conv1d`（深度可分离），输入先转置为 `[N*d_in, 1, T]` 做分组 1D 卷积后还原；或用自定义 `causal_depthwise_conv1d`。  
 45 | - **复杂度**：\(O(T\cdot N\cdot d_\text{in}\cdot K)\)。  
 46 | - **正则**（可选）：`L1(w)` 或 `Entropy(w)` 促进稀疏/清晰峰值。  
 47 | - **边界**：对 `t<0` 采用零填充。
 48 | 
 49 | ---
 50 | 
 51 | ## 3. STA-GNN（脉冲时序注意力聚合）
 52 | 
 53 | ### 核心思想
 54 | 在**因果掩码**与**邻接掩码**下，使用**相对时间编码**的多头注意力，让每个节点在时刻 \(t\) 选择来自邻居在**过去各时刻 \(t' \le t\)** 的关键事件，并用**脉冲发放**对注意力进行**门控**，保持事件驱动与可导性。
 55 | 
 56 | ### 相对时间编码 \(\phi(\Delta t)\)
 57 | - 组合基函数（推荐维度 \(d_\text{pe}=8\)）：  
 58 |   - 指数衰减：\(\exp(-\Delta t/\tau_m)\), \(\exp(-\Delta t/\tau_s)\)  
 59 |   - 正弦基：\(\sin(\omega_r \Delta t),\ \cos(\omega_r \Delta t)\)（对数均匀频率）  
 60 |   - 分桶 one-hot：\(\text{bucket}(\Delta t)\)（对数间隔）  
 61 | - 预计算 `PE: Float[T, T, d_pe]`，仅在 \(t' \le t\) 使用。
 62 | 
 63 | ### 注意力计算（单头，后接 H 头拼接）
 64 | 给定 `H̃`（DelayLine 之后）：
 65 | \[
 66 | \begin{aligned}
 67 | q_{i,t}&=W_Q h_{i,t},\\
 68 | k_{j,t'}&=W_K [h_{j,t'} \,\|\, \phi(t-t')],\\
 69 | v_{j,t'}&=W_V h_{j,t'}.
 70 | \end{aligned}
 71 | \]
 72 | **门控**：将源端脉冲 \(s_{j,t'}\in\{0,1\}\) 通过 surrogate \(\sigma_{\text{sur}}(\cdot)\) 映射到 \([0,1]\)，作为注意力值的可导缩放因子。  
 73 | \[
 74 | e_{i,t,j,t'}=\frac{q_{i,t}\cdot k_{j,t'}}{\sqrt{d/H}} + b_{\Delta t} \quad (\text{可选相对偏置})
 75 | \]
 76 | \[
 77 | a_{i,t,j,t'}=\text{softmax}_{(j,t')\in\mathcal{N}(i),t'\le t}\big(e_{i,t,j,t'}\big)\cdot \sigma_{\text{sur}}(s_{j,t'})
 78 | \]
 79 | \[
 80 | m_{i,t}=\sum_{t'\le t}\sum_{j\in\mathcal{N}(i)} a_{i,t,j,t'} \, v_{j,t'}
 81 | \]
 82 | 
 83 | ### 稀疏化与掩码
 84 | - **因果掩码**：仅允许 \(t'\le t\)。  
 85 | - **邻接掩码**：仅允许 `j ∈ N(i)`。  
 86 | - **时间窗**：限制 \(t-t'\le W\)（建议 `W∈[16,128]` 根据任务/显存选择）。  
 87 | - **Top-k**：对每个 `(i,t)` 仅保留前 `k` 个 key（在 `(j,t')` 维度上），其余置零并重新归一化（`k=8~32`）。
 88 | 
 89 | ### 形状与接口
 90 | - 入：  
 91 |   - `H_tilde: Float[T, N, d_in]`  
 92 |   - `S: Float[T, N]`（0/1）  
 93 |   - `edge_index: Long[2, E]` 或 `adj_mask: Bool[N,N]`  
 94 |   - `time_idx: Long[T]`（通常为 `torch.arange(T)`）  
 95 | - 出：`M: Float[T, N, d]`  
 96 | - 可选：`dropout_attn`, `attn_temperature`, `relative_bias=True/False`。
 97 | 
 98 | ### 复杂度与两种实现模式
 99 | - **Dense 模式（小图/短序列）**：构建 `[T,T,N,N]` 掩码后矩阵化，配合 Top-k；实现简单，峰值显存较高。  
100 | - **Sparse-edge 模式（推荐）**：以 `edge_index` 为骨架，仅对边上的 `(i,j)` 计算注意力；按时间窗为每条边滚动收集 \(t'\in[t-W,t]\) 的 key，使用分块与 `segment_softmax`。复杂度近似 \(O(H\cdot E\cdot W)\)。
101 | 
102 | ---
103 | 
104 | ## 4. 脉冲神经元（LIF/GLIF）与集成
105 | 
106 | ### LIF 更新（可换成你已有实现）
107 | \[
108 | \begin{aligned}
109 | V_{i,t} &= \lambda V_{i,t-1}+ U m_{i,t} - \theta_{i,t-1} R_{i,t-1} \\
110 | S_{i,t} &= \mathbb{1}[V_{i,t} > \theta_{i,t}] \\
111 | V_{i,t} &\leftarrow V_{i,t} - S_{i,t}\cdot \theta_{i,t} \quad (\text{重置}) \\
112 | \theta_{i,t} &= \tau_\theta \theta_{i,t-1} + \gamma S_{i,t-1} \quad (\text{自适应阈值，可选})
113 | \end{aligned}
114 | \]
115 | - **反传**：使用 surrogate \(\sigma_{\text{sur}}'(V-\theta)\)（fast-tanh 或 piecewise-linear STE）。  
116 | - **接口**：`LIFCell(M: [T,N,d]) -> S: [T,N], V: [T,N], (theta/R 等可选)`
117 | 
118 | ---
119 | 
120 | ## 5. SpikeNet-X 层与前向流程
121 | 
122 | ### SpikeNet-X 层伪接口
123 | ```python
124 | class SpikeNetXLayer(nn.Module):
125 |     def __init__(self, d_in, d, heads=4, topk=16, W=64, K=5, rho=0.85,
126 |                  use_rel_bias=True, attn_drop=0.1, temp=1.0, per_channel=True):
127 |         self.delay = LearnableDelayLine(d_in, K, rho, per_channel=per_channel)
128 |         self.sta   = SpikingTemporalAttention(d_in, d, heads, topk, W,
129 |                                               use_rel_bias, attn_drop, temp)
130 |         self.neuron = LIFCell(d, adaptive=True)  # 或接入你现有的 SNN 单元
131 |         self.norm = LayerNorm(d)                 # 可选：Pre/LN
132 |         self.ffn  = MLP(d, d)                    # 可选：残差前馈
133 |     def forward(self, H, S_prev, edge_index, time_idx, adj_mask=None, batch=None):
134 |         H̃ = self.delay(H)                                    # [T,N,d_in]
135 |         M  = self.sta(H̃, S_prev, edge_index, time_idx, adj_mask)  # [T,N,d]
136 |         S, V, aux = self.neuron(M)                            # [T,N], [T,N]
137 |         Y = self.norm(M)                                      # 或对 M+FFN 做残差
138 |         return S, V, Y, {"M": M, **aux}
139 | ```
140 | 
141 | ### 整体网络（L 层堆叠）
142 | - 时间维在外层循环或用并行张量化均可。推荐**张量化时间**（形状 `[T,N,·]`）以便 DelayLine 与 STA 使用缓存的 `PE`。  
143 | - 层与层之间传递：`H_{l+1,t} = proj([H_{l,t} || Y_{l,t} || onehot(S_{l,t})])`（可选拼接上一层输出与脉冲 one-hot）。  
144 | - 读出：  
145 |   - 节点分类：`readout_t` 可择 `t=T` 或 `temporal_attention_pool`（轻量单头）  
146 |   - 图级任务：按 batch 聚合（mean/max/attention）
147 | 
148 | ---
149 | 
150 | ## 6. 训练配方（默认值可直接用）
151 | 
152 | - **优化**：AdamW，`lr=2e-3`，`weight_decay=0.01`；线性 warmup 5% 步数；`grad_clip=1.0`。  
153 | - **surrogate**：`fast_tanh`: \(\sigma'(x)=\beta(1-\tanh^2(\beta x))\)，`β=2.0`（前 10% epoch 用 `β=1.0` 软化）。  
154 | - **正则**：  
155 |   - 脉冲率 L1：\(\lambda_\text{spk}\in[1e-5,5e-5]\) 约束平均发放率；  
156 |   - 注意力熵惩罚（温和）：\(\lambda_\text{ent}=1e-4\)；  
157 |   - 延迟权重 L1/熵：\(\lambda_\text{delay}=1e-4\)。  
158 | - **时间窗/稀疏**：`W=64`，`topk=16`（大图任务可改 `W=32, topk=8`）。  
159 | - **混合精度**：AMP O2；**梯度检查点**：在 STA 内按 `(time block)` 分段。  
160 | - **数据增强（可选）**：时间戳抖动（±1~2 tick），随机时间伸缩（0.9~1.1）。
161 | 
162 | ---
163 | 
164 | ## 7. 掩码与数值稳定性（务必实现）
165 | 
166 | 1) **softmax 掩码**：对被掩蔽位置赋 `-inf`（或非常负的数），再 softmax。  
167 | 2) **Top-k**：在 logits 上选 k 大，再将非选中项 logits 置 `-inf`，避免“零后再归一”。  
168 | 3) **温度**：`logits /= temp`（`temp∈[0.7,1.4]`），可缓解早期梯度噪声。  
169 | 4) **归一**：DelayLine 权重用 `softplus`+`normalize`，数值安全加 `eps=1e-8`。  
170 | 5) **空邻居/空窗口**：若 `(i,t)` 无可用 key，返回零向量（或残差直通 `h_{i,t}`）。
171 | 
172 | ---
173 | 
174 | ## 8. 复杂度与内存控制
175 | 
176 | - **理论**：STA 稀疏实现复杂度 \(O(H\cdot E\cdot W)\)，内存近似同量级；  
177 | - **工程手段**：  
178 |   - 分块时间 `T = sum(T_b)`，逐块缓存 `PE[t−t']`；  
179 |   - 将 `edge_index` 排序（`coalesce`）以提升 `segment_softmax` 命中率；  
180 |   - 对高入度节点可设**邻居 top-k**上限（先按入度采样邻居，再做时序 top-k）。
181 | 
182 | ---
183 | 
184 | ## 9. 相对时间编码实现建议
185 | ```python
186 | def rel_time_enc(time_idx, d_pe=8, taus=(4,16), n_freq=3):
187 |     # time_idx: [T], return PE: [T, T, d_pe] for Δt>=0 else zero
188 |     # channels: [exp(-Δt/τ_m), exp(-Δt/τ_s), sin/cos with log-spaced freq, log-bucket onehot]
189 |     ...
190 | ```
191 | - 预计算仅对 \(\Delta t \in [0, W]\) 的子矩阵；其余赋零以节省显存。  
192 | - 可选**相对偏置** \(b_{\Delta t}\)（标量表）：长度 `W+1` 的可学习向量。
193 | 
194 | ---
195 | 
196 | ## 10. 模块签名与断言（供代码生成器遵循）
197 | 
198 | ### `LearnableDelayLine`
199 | - `__init__(d_in:int, K:int=5, rho:float=0.85, per_channel:bool=True)`.  
200 | - `forward(H:[T,N,d_in])->[T,N,d_in]`.  
201 | - **断言**：`K>=1`, `0<rho<1`, `H.dim()==3`.
202 | 
203 | ### `SpikingTemporalAttention`
204 | - `__init__(d_in:int, d:int, heads:int=4, topk:int=16, W:int=64, use_rel_bias:bool=True, attn_drop:float=0.1, temp:float=1.0)`.  
205 | - `forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E], time_idx:Long[T], adj_mask:Optional[Bool[N,N]]=None) -> [T,N,d]`.  
206 | - **断言**：`topk>=1`, `W>=1`, `heads*d_head==d`。  
207 | - **稀疏实现关键步骤**：  
208 |   1) 对每条边 `(j->i)` 构造过去窗口 `t'∈[t-W,t]` 的键集合；  
209 |   2) 计算 `q(i,t)` 与 `k(j,t')`，加上相对编码后做点积；  
210 |   3) 在每个 `(i,t)` 的候选集合上做 Top-k，再 masked-softmax；  
211 |   4) 加 `attn_drop`，与 `v(j,t')` 加权求和。
212 | 
213 | ### `LIFCell`
214 | - `__init__(d:int, lambda_mem:float=0.95, tau_theta:float=0.99, gamma:float=0.1, adaptive:bool=True, surrogate:str='fast_tanh', beta:float=2.0)`.  
215 | - `forward(M:[T,N,d])->Tuple[S:[T,N], V:[T,N], aux:Dict]`.
216 | 
217 | ---
218 | 
219 | ## 11. 训练与日志（必须记录的指标）
220 | 
221 | - 任务指标：Micro/Macro-F1 或 AUC。  
222 | - SNN 指标：平均发放率（全局/分层）、失活率（持续 0 发放）、爆发率（>50% 发放）。  
223 | - STA 指标：平均注意力熵、Top-k 选择比例、相对时间分布（\(\Delta t\) 直方图）。  
224 | - DelayLine 指标：`w_k` 的分布热图；`argmax k` 的频率。  
225 | - 资源指标：每 step 时间、峰值显存。  
226 | - **可视化**：`r_t` 与注意力重心的时间轨迹，`w_k` 热图，`Δt` 权重柱状图。
227 | 
228 | ---
229 | 
230 | ## 12. 消融与开关（实现为 config flags）
231 | 
232 | - `use_delayline: bool`（False = 仅 STA）  
233 | - `use_sta: bool`（False = 回退到原时间池化）  
234 | - `topk: int in {0->不裁剪, 8, 16, 32}`  
235 | - `W: int`（时间窗）  
236 | - `use_rel_bias: bool`（相对偏置）  
237 | - `per_channel_delay: bool`  
238 | - `surrogate_beta_warmup: bool`（早期软梯度）
239 | 
240 | ---
241 | 
242 | ## 13. 失败模式与守护
243 | 
244 | - **注意力过密/显存爆**：启用/减小 `topk` 与 `W`；`d_head` 降低；开启分块。  
245 | - **延迟学成平滑**：对 `w_k` 加熵惩罚或“中心惩罚”鼓励峰化；  
246 | - **梯度震荡**：`attn_temperature ↑`、`grad_clip=1.0`、`AdamW β2=0.99`；  
247 | - **空邻居**：返回零向量并走残差；  
248 | - **发放塌陷**：提高 `λ_spk` 下限、软化 surrogate（小 `β`）、降低阈值上调 `γ`。
249 | 
250 | ---
251 | 
252 | ## 14. 参考默认配置（YAML 片段）
253 | 
254 | ```yaml
255 | model:
256 |   d_in: 128
257 |   d: 256
258 |   layers: 3
259 |   heads: 4
260 |   topk: 16
261 |   W: 64
262 |   delayline:
263 |     use: true
264 |     K: 5
265 |     rho: 0.85
266 |     per_channel: true
267 |   sta:
268 |     use_rel_bias: true
269 |     attn_drop: 0.1
270 |     temp: 1.0
271 |   lif:
272 |     lambda_mem: 0.95
273 |     tau_theta: 0.99
274 |     gamma: 0.10
275 |     surrogate: fast_tanh
276 |     beta: 2.0
277 | train:
278 |   lr: 0.002
279 |   weight_decay: 0.01
280 |   grad_clip: 1.0
281 |   amp: true
282 |   seed: 42
283 | regularization:
284 |   l1_spike: 2.0e-5
285 |   attn_entropy: 1.0e-4
286 |   delay_reg: 1.0e-4
287 | ```
288 | 
289 | ---
290 | 
291 | ## 15. 最小工作示例（形状检查伪代码）
292 | ```python
293 | T, N, d_in, d, H = 64, 1024, 128, 256, 4
294 | H0 = torch.randn(T, N, d_in)
295 | S0 = torch.zeros(T, N)  # 若首层无前序脉冲，可用全 1 门控或上层脉冲
296 | edge_index = ...        # [2,E]
297 | time_idx = torch.arange(T)
298 | 
299 | layer = SpikeNetXLayer(d_in, d, heads=H, topk=16, W=64, K=5, rho=0.85)
300 | S, V, Y, aux = layer(H0, S0, edge_index, time_idx)
301 | assert S.shape == (T, N) and Y.shape == (T, N, d)
302 | ```
303 | 
304 | ---
305 | 
306 | ## 16. 写作要点（供注释/文档使用）
307 | - **创新点**：将**邻居选择（空间）× 时间对齐（时序）**统一到**事件驱动注意力**，并通过 DelayLine 显式建模**传播时延**；  
308 | - **可解释性**：输出 `w_k`、`Δt` 权重与注意力热区；  
309 | - **可扩展性**：STA 与 DelayLine 均为**即插即用**，可替换到任意脉冲/非脉冲时序图骨干。
310 | 
311 | ---
```

## File: F:\SomeProjects\CSGNN\cline_docs\activeContext.md

- Extension: .md
- Language: markdown
- Size: 1639 bytes
- Created: 2025-08-22 11:30:37
- Modified: 2025-09-14 18:36:42

### Code

```markdown
 1 | 
 2 | # 当前工作（2025-09-14）
 3 | 
 4 | SpikeNet-X 在 DBLP 上出现 **训练很慢 & 指标仅 ~0.3** 的异常，根因已排查并形成补丁：
 5 | 
 6 | - **子图边集构造错误**：之前只保留了“种子 → 邻居”的边，导致多层传播被掐断；已修复为**保留子图内所有边（src/dst ∈ 子图节点）**。
 7 | - **读出策略**：SpikeNet-X 默认 `readout="last"`，对抖动敏感；已统一为 **`readout="mean"`**。
 8 | - **注意力实现选择**：稀疏 STA 当前实现不支持 Top-k，且在小子图上速度欠佳；训练侧改为 **`attn_impl=dense`**，以获得矩阵并行和可用的 Top-k。
 9 | - **邻居采样**：将 `num_neighbors_to_sample` 提升至 **25**（或 `-1` 全邻居）以稳定覆盖。
10 | 
11 | ## 最近的变更
12 | 
13 | 1. **`sample_subgraph` 修复**：保留子图内部所有边，并使用纯 Torch 做“全局 → 局部”映射，移除 Python 字典热点。
14 | 2. **模型构造**：`SpikeNetX(..., readout="mean", attn_impl=...)`；命令行默认推荐 `--attn_impl dense`。
15 | 3. **训练脚本**：将 `num_neighbors_to_sample` 从 10 → 25；建议 `--W 32 --topk 8` 起步。
16 | 
17 | ## 当前状态
18 | 
19 | - SpikeNet（SAGE+SNN）F1≈0.75（基线稳）。
20 | - SpikeNet-X 预期：修复后应恢复到“与 SpikeNet 同量级”的区间；具体以本周全量训练为准。
21 | 
22 | ## 下一步计划
23 | 
24 | - 以 DBLP 为基线，跑满 100 epoch，记录 Macro/Micro-F1、训练时长与显存曲线。
25 | - 做 `W∈{16,32,64}` × `topk∈{8,16}` × `heads∈{2,4}` 的小网格。
26 | - 若需大图扩展，再切回 `attn_impl=sparse` 并开发 CUDA/高阶算子版本的 Top-k。
```

## File: F:\SomeProjects\CSGNN\cline_docs\productContext.md

- Extension: .md
- Language: markdown
- Size: 1639 bytes
- Created: 2025-08-22 11:30:18
- Modified: 2025-08-22 11:30:21

### Code

```markdown
 1 | # 产品背景 (Product Context)
 2 | 
 3 | ## 为什么需要这个项目？
 4 | 该项目旨在解决动态图表示学习中，现有方法在处理大型时间图时，由于通常使用循环神经网络 (RNNs) 而导致的计算和内存开销严重的问题。随着时间图的规模不断扩大，可伸缩性成为一个主要挑战。
 5 | 
 6 | ## 解决什么问题？
 7 | SpikeNet 提出了一种可伸缩的框架，旨在高效地捕获时间图的时序和结构模式，同时显著降低计算成本。它通过使用脉冲神经网络 (SNNs) 替代 RNNs 来解决传统方法在大型动态图上的效率问题，SNNs 作为 RNNs 的低功耗替代方案，能够以高效的方式将图动态建模为神经元群的脉冲序列，并实现基于脉冲的传播。
 8 | 
 9 | ## 应该如何工作？
10 | SpikeNet 框架通过 SNNs 建模时间图的演化动态。它通过实验证明，在时间节点分类任务上，相比现有基线，SpikeNet 具有更低的计算成本和更优的性能。特别地，它能够以显著更少的参数和计算开销扩展到大型时间图（2M 节点和 13M 边）。该项目还提供了扩展到静态图的示例。
11 | 
12 | 该项目包含以下主要部分：
13 | - **数据处理**：支持 DBLP、Tmall、Patent 等大型时间图数据集，并提供节点特征生成脚本 (`generate_feature.py`)。
14 | - **SpikeNet 模型实现**：核心模型逻辑可能位于 `spikenet/layers.py` 和 `spikenet/neuron.py` 等文件中。
15 | - **邻居采样器**：通过 `setup.py` 进行构建以实现高效的邻居采样。
16 | - **主训练脚本**：`main.py` 用于动态图，`main_static.py` 用于静态图。
```

## File: F:\SomeProjects\CSGNN\cline_docs\progress.md

- Extension: .md
- Language: markdown
- Size: 2604 bytes
- Created: 2025-08-22 12:19:46
- Modified: 2025-09-14 18:37:33

### Code

```markdown
 1 | # 项目进度
 2 | 
 3 | ## 已完成功能
 4 | 
 5 | - **`SpikeNet-X` 原型实现**:
 6 |   - `spikenet_x` 目录下的所有核心模块已完成。
 7 |   - 模型可以通过 `spikenet_x/minimal_example.py` 进行验证。
 8 | - **`main.py` 集成与稀疏 STA**:
 9 |   - `main.py` 支持通过 `--model spikenetx` 调用模型。
10 |   - 实现了 O(E) 复杂度的稀疏 STA 注意力机制，解决了大型图上的内存溢出问题。
11 | - **`SpikeNet-X` 训练流程修复**:
12 |   - **状态**: **已完成**
13 |   - **描述**: 成功定位并修复了稀疏 STA 实现中的 `RuntimeError` (in-place 操作错误)。模型现在可以在大型数据集上稳定运行。
14 | - **训练框架功能增强**:
15 |   - **状态**: **已完成**
16 |   - **描述**: 为 `main.py` 增加了关键的工程能力，包括：
17 |     - 基于验证集性能的模型自动保存。
18 |     - 从检查点文件恢复训练的断点续训功能。
19 |     - 用于快速评估已保存模型的独立测试模式。
20 | 
21 | ## 需要构建的内容
22 | 
23 | - **基线模型性能评估**:
24 | 
25 |   - **状态**: **未开始**
26 |   - **描述**: 在 DBLP 数据集上运行一次完整的端到端训练，获得基线性能指标（如 Macro/Micro-F1），为后续优化提供参考。
27 | - **超参数调优**:
28 | 
29 |   - **状态**: **未开始**
30 |   - **描述**: 在获得基线性能后，系统性地对模型的关键超参数（如学习率, 时间窗口, 注意力头数等）进行调优，以最大化模型性能。
31 | 
32 | ## 进度状态
33 | 
34 | - **`SpikeNet-X` 功能**: **已就绪 (READY FOR EXPERIMENTS)**
35 | - **原因**: 核心的 `RuntimeError` 已修复，训练流程稳定。同时，模型保存、断点续训等关键功能的加入，使得进行系统性的实验和调优成为可能。项目已从“功能解锁”阶段推进到“实验与优化”阶段。
36 | 
37 | ## 2025-09-14 更新
38 | 
39 | - ✅ **修复子图边集错误**：`sample_subgraph` 现在保留子图内部所有边（src/dst ∈ 子图）。
40 | - ✅ **读出策略统一**：SpikeNet-X 改为 `readout="mean"`。
41 | - ✅ **训练实现切换**：默认用 `attn_impl=dense`，启用 `--topk`；`W` 推荐 32。
42 | - ✅ **邻居采样放宽**：`num_neighbors_to_sample=25`（可按显存调节或 `-1`）。
43 | - 🔄 **DBLP 基线复现（SpikeNet-X）**：100 epoch，记录 Macro/Micro-F1（目标：≥ SpikeNet 同量级）。
44 | - 📌 **超参网格**：`W∈{16,32,64}` × `topk∈{8,16}` × `heads∈{2,4}`。
45 | 
46 | ## 待办清单
47 | 
48 | - [ ] 跑满 DBLP 100 epoch（dense, W=32, topk=8）。
49 | - [ ] 对比 `readout=last` vs `mean` 的消融。
50 | - [ ] sparse 实现的 Top-k 与 CUDA 化评估。
```

## File: F:\SomeProjects\CSGNN\cline_docs\systemPatterns.md

- Extension: .md
- Language: markdown
- Size: 4998 bytes
- Created: 2025-08-22 11:32:15
- Modified: 2025-09-14 18:37:56

### Code

```markdown
 1 | # 系统模式 (System Patterns)
 2 | 
 3 | ## 系统如何构建？
 4 | 
 5 | SpikeNet 项目是一个基于 PyTorch 实现的动态图表示学习框架，核心在于利用脉冲神经网络 (SNNs) 处理时间图数据。其主要组件和构建方式如下：
 6 | 
 7 | 1. **数据层 (`spikenet/dataset.py`)**：
 8 | 
 9 |    * 提供 `Dataset` 基类，以及 DBLP、Tmall、Patent 等具体数据集的实现。
10 |    * 负责从文件中读取节点特征（`.npy`）、边（`.txt` 或 `.json`）和标签（`node2label.txt` 或 `.json`）。
11 |    * 支持对节点特征进行标准化。
12 |    * 将边列表转换为稀疏邻接矩阵 (`scipy.sparse.csr_matrix`)。
13 |    * 实现节点和边的时间切片与划分，以模拟图的动态演化。
14 |    * 数据集迭代器允许按时间步访问图快照。
15 | 2. **核心模型组件 (`spikenet/neuron.py`, `spikenet/layers.py`)**：
16 | 
17 |    * **神经元模型 (`spikenet/neuron.py`)**：定义了基本的脉冲神经元（如 IF, LIF, PLIF）。这些神经元模型负责电压积分、发放脉冲和重置。
18 |    * **替代梯度 (`spikenet/neuron.py`)**：由于 SNN 的脉冲函数不可导，使用了多种替代梯度技术（如 SuperSpike, MultiGaussSpike, TriangleSpike, ArctanSpike, SigmoidSpike）来实现反向传播训练。
19 |    * **图聚合器 (`spikenet/layers.py`)**：包含了 `SAGEAggregator`，表明网络层可能采用了 GraphSAGE 风格的邻居特征聚合机制。它将中心节点特征与聚合后的邻居特征进行组合。
20 | 3. **图采样器 (`spikenet/utils.py`, `spikenet/sample_neighber.cpp`)**：
21 | 
22 |    * `spikenet/utils.py` 中定义了 `Sampler` 和 `RandomWalkSampler` 类，用于从邻接矩阵中采样邻居。
23 |    * `Sampler` 类利用了外部 C++ 实现 `sample_neighber_cpu` 进行高效的邻居采样，这可能是为了性能优化。
24 |    * `RandomWalkSampler` 在可选依赖 `torch_cluster` 存在时提供随机游走采样功能。
25 | 4. **特征生成 (`generate_feature.py`, `spikenet/deepwalk.py`)**：
26 | 
27 |    * `generate_feature.py` 脚本用于为不带原始特征的数据集生成节点特征，通过无监督的 DeepWalk 方法实现，其核心逻辑可能在 `spikenet/deepwalk.py` 中。
28 | 5. **训练入口 (`main.py`, `main_static.py`)**：
29 | 
30 |    * `main.py` 是用于动态图训练的主脚本，配置数据集、模型参数和训练过程。
31 |    * `main_static.py` 是用于静态图训练的脚本，可能适配了不同的数据集和训练流程。
32 | 
33 | ## 关键技术决策
34 | 
35 | * **SNNs 用于动态图**：核心创新是将 SNNs 应用于动态图表示学习，以解决传统 RNNs 在大规模图上的计算和内存效率问题。
36 | * **替代梯度**：采用替代梯度方法来训练 SNNs，使其能够通过反向传播进行优化。
37 | * **GraphSAGE 风格聚合**：使用聚合器从邻居节点收集信息，这是图神经网络中的常见模式。
38 | * **C++ 优化采样**：通过 `sample_neighber.cpp` 提供的 C++ 实现进行邻居采样，以提高性能和处理大规模图的能力。
39 | * **模块化设计**：将神经元模型、网络层、数据处理和采样器等功能分别封装在不同的模块中，提高了代码的可维护性和可扩展性。
40 | * **数据集支持**：设计了通用的 `Dataset` 接口，并为多个真实世界大型时间图数据集提供了具体实现。
41 | 
42 | ## 架构模式
43 | 
44 | * **时间序列图处理**：通过迭代时间步来处理图快照，捕获图的动态演化。
45 | * **消息传递范式**：聚合器（如 `SAGEAggregator`）遵循图神经网络的消息传递范式，其中节点通过聚合邻居信息来更新其表示。
46 | * **分离的数据加载与模型逻辑**：`dataset.py` 负责数据管理，而 `neuron.py` 和 `layers.py` 负责模型核心逻辑，实现了关注点分离。
47 | * **参数化神经元行为**：神经元模型（如 LIF）通过可配置的参数（如 `tau`, `v_threshold`, `alpha`）和可选择的替代梯度类型，提供了灵活性。
48 | * **命令行参数配置**：`main.py` 和 `main_static.py` 通过命令行参数 (`argparse`) 配置训练过程，方便实验和调优。
49 | 
50 | ## 子图构建（必遵循）
51 | 
52 | - **节点集合**：`subgraph_nodes = seeds ∪ sampled_neighbors`。
53 | - **边集合**：**仅保留 `src ∈ subgraph_nodes` 且 `dst ∈ subgraph_nodes` 的边**（解锁多层传播）。
54 | - **索引映射**：用 `torch.sort + searchsorted` 完成“全局 → 局部”的矢量化映射，避免 Python 字典循环。
55 | 
56 | ## 注意力实现选择
57 | 
58 | - **小至中规模子图（当前训练配置）**：`attn_impl=dense` + `topk`，速度/可复现性更优。
59 | - **大图扩展**：`attn_impl=sparse`（O(E·W)）— 当前版本 **不支持 Top-k**，需配合较小 `W` 或后续 CUDA 算子。
60 | 
61 | ## 读出策略
62 | 
63 | - SpikeNet-X 读出采用 **`mean`**（时间平均）比 `last` 更鲁棒（对时序抖动与对齐误差不敏感）。
64 | 
65 | ## 推荐默认超参
66 | 
67 | - `W=32`，`topk=8`，`heads=4`，`num_neighbors_to_sample=25`。
```

## File: F:\SomeProjects\CSGNN\cline_docs\techContext.md

- Extension: .md
- Language: markdown
- Size: 2097 bytes
- Created: 2025-08-22 11:33:19
- Modified: 2025-08-22 11:33:22

### Code

```markdown
 1 | # 技术背景 (Tech Context)
 2 | 
 3 | ## 使用的技术
 4 | *   **Python**：主要的编程语言。
 5 | *   **PyTorch**：核心深度学习框架，用于构建和训练神经网络。
 6 | *   **NumPy**：用于数值计算和数组操作。
 7 | *   **SciPy**：用于科学计算，特别是稀疏矩阵操作 (`scipy.sparse`)。
 8 | *   **Scikit-learn**：用于数据预处理（如 `LabelEncoder`）和模型评估。
 9 | *   **tqdm**：用于显示进度条。
10 | *   **texttable**：用于命令行参数的表格化输出。
11 | *   **Numba**：一个 JIT 编译器，可能用于加速某些 Python 代码。
12 | *   **C++**：用于高性能的邻居采样模块 (`sample_neighber.cpp`)，通过 `setup.py` 进行编译和集成。
13 | *   **torch_cluster (可选)**：如果安装，用于更高级的图采样操作，如随机游走。
14 | 
15 | ## 开发设置
16 | *   **环境**：项目支持在 PyTorch 环境下运行。
17 | *   **依赖**：`requirements` 部分列出了具体的包及其版本，包括 `tqdm`, `scipy`, `texttable`, `torch`, `numpy`, `numba`, `scikit_learn` 和可选的 `torch_cluster`。
18 | *   **邻居采样器构建**：需要通过运行 `python setup.py install` 来编译和安装 C++ 实现的邻居采样器。
19 | *   **数据准备**：数据集需要下载并放置在 `data/` 目录下。对于没有原始节点特征的数据集，可以通过 `generate_feature.py` 脚本使用 DeepWalk 生成特征。
20 | 
21 | ## 技术约束
22 | *   **大规模图处理**：设计目标是处理包含数百万节点和数千万边的大型时间图，对计算和内存效率有较高要求。
23 | *   **SNN 训练挑战**：脉冲函数不可导，需要依赖替代梯度方法进行训练。
24 | *   **数据格式**：需要适配不同数据集的特定文件格式（`.txt`, `.json`, `.npy`）。
25 | *   **PyTorch 版本兼容性**：代码应兼容 PyTorch 1.6-1.12 版本。
26 | *   **C++ 依赖**：邻居采样器依赖 C++ 编译，可能需要相应的编译环境。
27 | *   **`torch_cluster` 依赖 (可选)**：随机游走采样功能依赖于 `torch_cluster` 库，如果未安装则无法使用该功能。
```

## File: F:\SomeProjects\CSGNN\spikenet\dataset.py

- Extension: .py
- Language: python
- Size: 11862 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2023-09-27 17:42:24

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
- Size: 4914 bytes
- Created: 2025-08-22 12:57:19
- Modified: 2025-08-22 12:57:41

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | LIFCell: 脉冲神经元单元（支持自适应阈值与 fast-tanh 代理梯度）
  4 | 
  5 | 接口
  6 | ----
  7 | forward(M: Float[T, N, d]) -> Tuple[S: Float[T, N], V: Float[T, N], aux: Dict]
  8 | - M 为从聚合器得到的消息（电流输入）
  9 | - 先用线性投影 U: R^d -> R 将通道聚合为标量电流 I_tn
 10 | - 递推更新膜电位与阈值，产生脉冲
 11 | 
 12 | 参考公式（提示词）
 13 | ----------------
 14 | V_{i,t} = λ V_{i,t-1} + U m_{i,t} - θ_{i,t-1} R_{i,t-1}
 15 | S_{i,t} = 𝟙[V_{i,t} > θ_{i,t}]
 16 | V_{i,t} ← V_{i,t} - S_{i,t} · θ_{i,t}          (重置)
 17 | θ_{i,t} = τ_θ θ_{i,t-1} + γ S_{i,t-1}          (自适应阈值，可选)
 18 | 
 19 | 训练
 20 | ----
 21 | - 使用 fast-tanh 代理梯度:
 22 |   y = H(x) + (tanh(βx) - tanh(βx).detach())
 23 |   其中 H(x) 为硬阶跃 (x>0)
 24 | """
 25 | 
 26 | from __future__ import annotations
 27 | 
 28 | from typing import Dict, Optional, Tuple
 29 | 
 30 | import torch
 31 | import torch.nn as nn
 32 | 
 33 | 
 34 | def _fast_tanh_surrogate(x: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
 35 |     """
 36 |     硬触发 + 平滑梯度的 STE 实现:
 37 |       forward: step(x)
 38 |       backward: tanh(βx) 的导数 (≈ β * (1 - tanh^2(βx)))
 39 |     """
 40 |     hard = (x > 0).to(x.dtype)
 41 |     soft = torch.tanh(beta * x)
 42 |     return hard + (soft - soft.detach())
 43 | 
 44 | 
 45 | class LIFCell(nn.Module):
 46 |     def __init__(
 47 |         self,
 48 |         d: int,
 49 |         lambda_mem: float = 0.95,
 50 |         tau_theta: float = 0.99,
 51 |         gamma: float = 0.10,
 52 |         adaptive: bool = True,
 53 |         surrogate: str = "fast_tanh",
 54 |         beta: float = 2.0,
 55 |     ) -> None:
 56 |         super().__init__()
 57 |         assert 0.0 <= lambda_mem <= 1.0
 58 |         assert 0.0 <= tau_theta <= 1.0
 59 |         assert gamma >= 0.0
 60 | 
 61 |         self.d = int(d)
 62 |         self.adaptive = bool(adaptive)
 63 |         self.surrogate = str(surrogate)
 64 |         self.beta = float(beta)
 65 | 
 66 |         # U: R^d -> R（共享于所有节点），无偏置避免电流漂移
 67 |         self.proj = nn.Linear(d, 1, bias=False)
 68 | 
 69 |         # 将标量参数注册为 buffer，便于脚本化与移动设备
 70 |         self.register_buffer("lambda_mem", torch.as_tensor(lambda_mem, dtype=torch.float32))
 71 |         self.register_buffer("tau_theta", torch.as_tensor(tau_theta, dtype=torch.float32))
 72 |         self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float32))
 73 | 
 74 |     def _spike(self, x: torch.Tensor) -> torch.Tensor:
 75 |         if self.surrogate == "fast_tanh":
 76 |             return _fast_tanh_surrogate(x, beta=self.beta)
 77 |         # 兜底：纯硬阈值（无代理梯度）
 78 |         return (x > 0).to(x.dtype)
 79 | 
 80 |     @torch.no_grad()
 81 |     def reset_parameters(self) -> None:
 82 |         nn.init.xavier_uniform_(self.proj.weight)
 83 | 
 84 |     def forward(
 85 |         self,
 86 |         M: torch.Tensor,                # [T, N, d]
 87 |         state0: Optional[Dict] = None,  # 可选: {"V": [N], "theta": [N], "S": [N]}
 88 |     ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
 89 |         assert M.dim() == 3, "M 形状应为 [T, N, d]"
 90 |         T, N, d = M.shape
 91 |         assert d == self.d, f"d 不匹配: 期望 {self.d}, 实得 {d}"
 92 | 
 93 |         device = M.device
 94 |         dtype = M.dtype
 95 | 
 96 |         # 初始状态
 97 |         if state0 is None:
 98 |             V = torch.zeros(N, device=device, dtype=dtype)
 99 |             theta = torch.ones(N, device=device, dtype=dtype)  # 初始阈值 1.0
100 |             S_prev = torch.zeros(N, device=device, dtype=dtype)
101 |         else:
102 |             V = state0.get("V", torch.zeros(N, device=device, dtype=dtype)).to(dtype)
103 |             theta = state0.get("theta", torch.ones(N, device=device, dtype=dtype)).to(dtype)
104 |             S_prev = state0.get("S", torch.zeros(N, device=device, dtype=dtype)).to(dtype)
105 | 
106 |         S_seq = []
107 |         V_seq = []
108 |         theta_seq = []
109 | 
110 |         lam = self.lambda_mem
111 |         tau = self.tau_theta
112 |         gam = self.gamma
113 | 
114 |         for t in range(T):
115 |             # 投影到标量电流 I_tn: [N]
116 |             I = self.proj(M[t]).squeeze(-1)  # [N]
117 | 
118 |             # 记忆衰减 + 输入累积
119 |             V = lam * V + I - (theta * S_prev)  # 包含上一步的 Refractory 抑制项
120 | 
121 |             # 触发条件与代理梯度
122 |             x = V - theta
123 |             S = self._spike(x)  # [N] in [0,1]
124 | 
125 |             # 重置：发放处扣除阈值
126 |             V = V - S * theta
127 | 
128 |             # 自适应阈值
129 |             if self.adaptive:
130 |                 theta = tau * theta + gam * S_prev
131 | 
132 |             # 记录
133 |             S_seq.append(S)
134 |             V_seq.append(V)
135 |             theta_seq.append(theta)
136 | 
137 |             # 更新上一时刻的发放
138 |             S_prev = S
139 | 
140 |         S_out = torch.stack(S_seq, dim=0)  # [T, N]
141 |         V_out = torch.stack(V_seq, dim=0)  # [T, N]
142 | 
143 |         aux = {
144 |             "theta": torch.stack(theta_seq, dim=0),   # [T, N]
145 |             "spike_rate": S_out.mean().detach(),      # 标量，便于监控
146 |         }
147 |         return S_out, V_out, aux
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
- Size: 2572 bytes
- Created: 2025-08-22 13:11:54
- Modified: 2025-08-22 23:52:28

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
54 |         attn_impl="sparse",
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
- Size: 7649 bytes
- Created: 2025-08-22 13:07:03
- Modified: 2025-08-22 23:32:42

### Code

```python
  1 | # -*- coding: utf-8 -*-
  2 | """
  3 | SpikeNet-X: multi-layer network composed of (DelayLine -> STA -> LIF) blocks.
  4 | 
  5 | This module provides a thin, task-agnostic backbone `SpikeNetX` that stacks
  6 | `SpikeNetXLayer` L times and (optionally) a lightweight readout head.
  7 | 
  8 | Key ideas follow `提示词.md`:
  9 | - Time-first tensors: H: [T, N, d_in], S: [T, N]
 10 | - Event-driven STA with causal window W and Top-k sparsification
 11 | - Learnable DelayLine in front of STA to model propagation delay
 12 | - LIF cell generates spikes that can gate attention in the next layer
 13 | 
 14 | Typical usage
 15 | -------------
 16 | >>> import torch
 17 | >>> from spikenet_x import SpikeNetX
 18 | >>> T, N, d_in, d, Hs, L = 32, 128, 64, 128, 4, 2
 19 | >>> H0 = torch.randn(T, N, d_in)
 20 | >>> edge_index = torch.randint(0, N, (2, 4*N))  # toy edges
 21 | >>> time_idx = torch.arange(T)
 22 | >>> model = SpikeNetX(d_in=d_in, d=d, layers=L, heads=Hs, topk=16, W=32, out_dim=10)
 23 | >>> out = model(H0, edge_index=edge_index, time_idx=time_idx)
 24 | >>> out["logits"].shape  # [N, out_dim] by default (last-time readout)
 25 | torch.Size([128, 10])
 26 | """
 27 | 
 28 | from __future__ import annotations
 29 | 
 30 | from typing import Dict, List, Optional, Tuple
 31 | 
 32 | import torch
 33 | import torch.nn as nn
 34 | 
 35 | from .spikenetx_layer import SpikeNetXLayer
 36 | 
 37 | 
 38 | class SpikeNetX(nn.Module):
 39 |     """
 40 |     A stack of SpikeNetXLayer blocks with optional classifier head.
 41 | 
 42 |     Args
 43 |     ----
 44 |     d_in: int
 45 |         Input feature dimension.
 46 |     d: int
 47 |         Hidden/STA output dimension per layer.
 48 |     layers: int
 49 |         Number of stacked layers.
 50 |     heads: int
 51 |         Number of attention heads per layer.
 52 |     topk: int
 53 |         Top-k candidates kept per (i,t) in STA.
 54 |     W: int
 55 |         Causal attention time window.
 56 |     K: int
 57 |         DelayLine taps.
 58 |     rho: float
 59 |         DelayLine exponential discount.
 60 |     use_rel_bias: bool
 61 |         Whether to use learnable relative bias b[Δt].
 62 |     attn_drop: float
 63 |         Attention dropout prob.
 64 |     temp: float
 65 |         Softmax temperature for attention logits.
 66 |     per_channel: bool
 67 |         Per-channel DelayLine weights if True (recommended).
 68 |     ffn_hidden_mult: int
 69 |         Multiplier of FFN hidden width inside each layer.
 70 |     ffn_drop: float
 71 |         Dropout inside layer FFN.
 72 |     lif_*: see LIFCell.
 73 |     out_dim: Optional[int]
 74 |         If set, attach a linear head to produce logits for node-level tasks.
 75 |     readout: str
 76 |         'last' (default): use last time-step T-1 for logits,
 77 |         'mean': temporal mean pooling over T.
 78 |     """
 79 | 
 80 |     def __init__(
 81 |         self,
 82 |         d_in: int,
 83 |         d: int,
 84 |         layers: int = 2,
 85 |         heads: int = 4,
 86 |         topk: int = 16,
 87 |         W: int = 64,
 88 |         K: int = 5,
 89 |         rho: float = 0.85,
 90 |         use_rel_bias: bool = True,
 91 |         attn_drop: float = 0.1,
 92 |         temp: float = 1.0,
 93 |         attn_impl: str = "dense",
 94 |         per_channel: bool = True,
 95 |         ffn_hidden_mult: int = 4,
 96 |         ffn_drop: float = 0.1,
 97 |         lif_lambda_mem: float = 0.95,
 98 |         lif_tau_theta: float = 0.99,
 99 |         lif_gamma: float = 0.10,
100 |         lif_adaptive: bool = True,
101 |         lif_surrogate: str = "fast_tanh",
102 |         lif_beta: float = 2.0,
103 |         out_dim: Optional[int] = None,
104 |         readout: str = "last",
105 |     ) -> None:
106 |         super().__init__()
107 |         assert layers >= 1, "layers must be >= 1"
108 |         assert readout in ("last", "mean"), "readout must be 'last' or 'mean'"
109 | 
110 |         self.layers = int(layers)
111 |         self.readout = readout
112 |         self.out_dim = out_dim
113 |         self.attn_impl = attn_impl
114 |         assert self.attn_impl in ("dense", "sparse"), "attn_impl must be 'dense' or 'sparse'"
115 | 
116 |         mods: List[SpikeNetXLayer] = []
117 |         for l in range(layers):
118 |             in_dim = d_in if l == 0 else d
119 |             mods.append(
120 |                 SpikeNetXLayer(
121 |                     d_in=in_dim,
122 |                     d=d,
123 |                     heads=heads,
124 |                     topk=topk,
125 |                     W=W,
126 |                     K=K,
127 |                     rho=rho,
128 |                     use_rel_bias=use_rel_bias,
129 |                     attn_drop=attn_drop,
130 |                     temp=temp,
131 |                     attn_impl=attn_impl,
132 |                     per_channel=per_channel,
133 |                     ffn_hidden_mult=ffn_hidden_mult,
134 |                     ffn_drop=ffn_drop,
135 |                     lif_lambda_mem=lif_lambda_mem,
136 |                     lif_tau_theta=lif_tau_theta,
137 |                     lif_gamma=lif_gamma,
138 |                     lif_adaptive=lif_adaptive,
139 |                     lif_surrogate=lif_surrogate,
140 |                     lif_beta=lif_beta,
141 |                 )
142 |             )
143 |         self.blocks = nn.ModuleList(mods)
144 | 
145 |         self.head = nn.Linear(d, out_dim, bias=True) if out_dim is not None else None
146 |         if self.head is not None:
147 |             nn.init.xavier_uniform_(self.head.weight)
148 |             nn.init.zeros_(self.head.bias)
149 | 
150 |     def forward(
151 |         self,
152 |         H: torch.Tensor,                        # [T, N, d_in]
153 |         edge_index: Optional[torch.Tensor],     # [2, E] or None (if adj_mask provided)
154 |         time_idx: torch.Tensor,                 # [T]
155 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool or None
156 |         S0: Optional[torch.Tensor] = None,        # initial spikes for layer-0 gating [T, N] (optional)
157 |     ) -> Dict[str, torch.Tensor]:
158 |         assert H.dim() == 3, "H should be [T, N, d_in]"
159 |         T, N, _ = H.shape
160 |         assert time_idx.dim() == 1 and time_idx.numel() == T, "time_idx must be [T]"
161 | 
162 |         S_prev = S0  # first layer gating; None -> all-ones gating inside block
163 |         Y = None
164 |         S_list: List[torch.Tensor] = []
165 |         V_list: List[torch.Tensor] = []
166 | 
167 |         X = H
168 |         aux_last: Dict[str, torch.Tensor] = {}
169 |         for blk in self.blocks:
170 |             S, V, Y, aux = blk(
171 |                 H=X,
172 |                 S_prev=S_prev,
173 |                 edge_index=edge_index,
174 |                 time_idx=time_idx,
175 |                 adj_mask=adj_mask,
176 |             )
177 |             S_list.append(S)   # each: [T, N]
178 |             V_list.append(V)
179 |             S_prev = S         # spikes feed-forward as gate for next layer
180 |             X = Y              # features for next layer
181 |             aux_last = aux
182 | 
183 |         # Readout
184 |         if self.readout == "last":
185 |             z = Y[-1]  # [N, d]
186 |         else:  # "mean"
187 |             z = Y.mean(dim=0)  # [N, d]
188 | 
189 |         logits = self.head(z) if self.head is not None else None
190 | 
191 |         out: Dict[str, torch.Tensor] = {
192 |             "repr": z,                       # [N, d]
193 |             "Y_last": Y,                     # [T, N, d]
194 |             "S_list": torch.stack(S_list),   # [L, T, N]
195 |             "V_list": torch.stack(V_list),   # [L, T, N]
196 |         }
197 |         if logits is not None:
198 |             out["logits"] = logits           # [N, out_dim]
199 | 
200 |         # for convenience: expose a few internals when available
201 |         if "M" in aux_last:
202 |             out["M_last"] = aux_last["M"]    # [T, N, d]
203 | 
204 |         return out
205 | 
206 | 
207 | def shape_check_demo() -> Tuple[torch.Size, Optional[torch.Size]]:
208 |     """
209 |     Minimal shape check (no training). Returns (Y_last_shape, logits_shape).
210 |     """
211 |     T, N, d_in, d, Hs, L = 16, 64, 32, 64, 4, 2
212 |     E = N * 4
213 |     H0 = torch.randn(T, N, d_in)
214 |     edge_index = torch.randint(0, N, (2, E))
215 |     time_idx = torch.arange(T)
216 | 
217 |     model = SpikeNetX(d_in=d_in, d=d, layers=L, heads=Hs, topk=8, W=8, out_dim=5)
218 |     out = model(H0, edge_index=edge_index, time_idx=time_idx)
219 |     return out["Y_last"].shape, out.get("logits", None).shape if "logits" in out else None
```

## File: F:\SomeProjects\CSGNN\spikenet_x\rel_time.py

- Extension: .py
- Language: python
- Size: 6544 bytes
- Created: 2025-08-22 12:52:55
- Modified: 2025-08-22 12:53:38

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

## File: F:\SomeProjects\CSGNN\spikenet_x\sta.py

- Extension: .py
- Language: python
- Size: 9673 bytes
- Created: 2025-08-22 12:59:28
- Modified: 2025-09-14 19:28:20

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
195 |                 gate_chunks.append(torch.log(gate_j + eps_gate).view(1, 1, N))  # [1,1,N]
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
- Modified: 2025-08-23 00:17:47

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

## File: F:\SomeProjects\CSGNN\spikenet_x\__init__.py

- Extension: .py
- Language: python
- Size: 973 bytes
- Created: 2025-08-22 13:06:01
- Modified: 2025-08-22 23:27:05

### Code

```python
 1 | # -*- coding: utf-8 -*-
 2 | """
 3 | SpikeNet-X package
 4 | 
 5 | Exports the core building blocks specified in `提示词.md`:
 6 | - LearnableDelayLine
 7 | - SpikingTemporalAttention (dense fallback implementation)
 8 | - LIFCell
 9 | - SpikeNetXLayer
10 | - Masked ops helpers and RelativeTimeEncoding
11 | """
12 | 
13 | from .masked_ops import (
14 |     masked_softmax,
15 |     masked_topk_softmax,
16 |     topk_mask_logits,
17 |     fill_masked_,
18 |     NEG_INF,
19 | )
20 | from .rel_time import RelativeTimeEncoding
21 | from .delayline import LearnableDelayLine
22 | from .sta import SpikingTemporalAttention
23 | from .sta_sparse import SparseSpikingTemporalAttention
24 | from .lif_cell import LIFCell
25 | from .spikenetx_layer import SpikeNetXLayer
26 | 
27 | __all__ = [
28 |     "masked_softmax",
29 |     "masked_topk_softmax",
30 |     "topk_mask_logits",
31 |     "fill_masked_",
32 |     "NEG_INF",
33 |     "RelativeTimeEncoding",
34 |     "LearnableDelayLine",
35 |     "SpikingTemporalAttention",
36 |     "SparseSpikingTemporalAttention",
37 |     "LIFCell",
38 |     "SpikeNetXLayer",
39 | ]
```


# Table of Contents
- F:\SomeProjects\CSGNN\.gitignore
- F:\SomeProjects\CSGNN\CSGNN_prompt.md
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

## File: F:\SomeProjects\CSGNN\CSGNN_prompt.md

- Extension: .md
- Language: markdown
- Size: 194215 bytes
- Created: 2025-09-14 17:37:24
- Modified: 2025-09-14 17:37:24

### Code

```markdown
   1 | # Table of Contents
   2 | - F:\SomeProjects\CSGNN\.gitignore
   3 | - F:\SomeProjects\CSGNN\generate_feature.py
   4 | - F:\SomeProjects\CSGNN\LICENSE
   5 | - F:\SomeProjects\CSGNN\main.py
   6 | - F:\SomeProjects\CSGNN\main_static.py
   7 | - F:\SomeProjects\CSGNN\README.md
   8 | - F:\SomeProjects\CSGNN\setup.py
   9 | - F:\SomeProjects\CSGNN\提示词.md
  10 | - F:\SomeProjects\CSGNN\cline_docs\activeContext.md
  11 | - F:\SomeProjects\CSGNN\cline_docs\productContext.md
  12 | - F:\SomeProjects\CSGNN\cline_docs\progress.md
  13 | - F:\SomeProjects\CSGNN\cline_docs\systemPatterns.md
  14 | - F:\SomeProjects\CSGNN\cline_docs\techContext.md
  15 | - F:\SomeProjects\CSGNN\spikenet\dataset.py
  16 | - F:\SomeProjects\CSGNN\spikenet\deepwalk.py
  17 | - F:\SomeProjects\CSGNN\spikenet\layers.py
  18 | - F:\SomeProjects\CSGNN\spikenet\neuron.py
  19 | - F:\SomeProjects\CSGNN\spikenet\sample_neighber.cpp
  20 | - F:\SomeProjects\CSGNN\spikenet\utils.py
  21 | - F:\SomeProjects\CSGNN\spikenet_x\delayline.py
  22 | - F:\SomeProjects\CSGNN\spikenet_x\lif_cell.py
  23 | - F:\SomeProjects\CSGNN\spikenet_x\masked_ops.py
  24 | - F:\SomeProjects\CSGNN\spikenet_x\minimal_example.py
  25 | - F:\SomeProjects\CSGNN\spikenet_x\model.py
  26 | - F:\SomeProjects\CSGNN\spikenet_x\rel_time.py
  27 | - F:\SomeProjects\CSGNN\spikenet_x\spikenetx_layer.py
  28 | - F:\SomeProjects\CSGNN\spikenet_x\sta.py
  29 | - F:\SomeProjects\CSGNN\spikenet_x\sta_sparse.py
  30 | - F:\SomeProjects\CSGNN\spikenet_x\__init__.py
  31 | 
  32 | ## File: F:\SomeProjects\CSGNN\.gitignore
  33 | 
  34 | - Extension: 
  35 | - Language: unknown
  36 | - Size: 1281 bytes
  37 | - Created: 2025-08-21 17:29:04
  38 | - Modified: 2023-09-27 17:42:24
  39 | 
  40 | ### Code
  41 | 
  42 | ```unknown
  43 |   1 | # Custom
  44 |   2 | *.idea
  45 |   3 | *.pdf
  46 |   4 | *.txt
  47 |   5 | *.npy
  48 |   6 | !requirements.txt
  49 |   7 | data/
  50 |   8 | # Byte-compiled / optimized / DLL files
  51 |   9 | __pycache__/
  52 |  10 | *.py[cod]
  53 |  11 | *$py.class
  54 |  12 | 
  55 |  13 | # C extensions
  56 |  14 | *.so
  57 |  15 | 
  58 |  16 | # Distribution / packaging
  59 |  17 | .Python
  60 |  18 | env/
  61 |  19 | build/
  62 |  20 | develop-eggs/
  63 |  21 | dist/
  64 |  22 | downloads/
  65 |  23 | eggs/
  66 |  24 | .eggs/
  67 |  25 | lib/
  68 |  26 | lib64/
  69 |  27 | parts/
  70 |  28 | sdist/
  71 |  29 | var/
  72 |  30 | *.egg-info/
  73 |  31 | .installed.cfg
  74 |  32 | *.egg
  75 |  33 | 
  76 |  34 | # PyInstaller
  77 |  35 | #  Usually these files are written by a python script from a template
  78 |  36 | #  before PyInstaller builds the exe, so as to inject date/other infos into it.
  79 |  37 | *.manifest
  80 |  38 | *.spec
  81 |  39 | 
  82 |  40 | # Installer logs
  83 |  41 | pip-log.txt
  84 |  42 | pip-delete-this-directory.txt
  85 |  43 | 
  86 |  44 | # Unit test / coverage reports
  87 |  45 | htmlcov/
  88 |  46 | .tox/
  89 |  47 | .coverage
  90 |  48 | .coverage.*
  91 |  49 | .cache
  92 |  50 | nosetests.xml
  93 |  51 | coverage.xml
  94 |  52 | *,cover
  95 |  53 | .hypothesis/
  96 |  54 | 
  97 |  55 | # Translations
  98 |  56 | *.mo
  99 |  57 | *.pot
 100 |  58 | 
 101 |  59 | # Django stuff:
 102 |  60 | *.log
 103 |  61 | local_settings.py
 104 |  62 | 
 105 |  63 | # Flask stuff:
 106 |  64 | instance/
 107 |  65 | .webassets-cache
 108 |  66 | 
 109 |  67 | # Scrapy stuff:
 110 |  68 | .scrapy
 111 |  69 | 
 112 |  70 | # Sphinx documentation
 113 |  71 | docs/build/
 114 |  72 | 
 115 |  73 | # PyBuilder
 116 |  74 | target/
 117 |  75 | 
 118 |  76 | # IPython Notebook
 119 |  77 | .ipynb_checkpoints
 120 |  78 | 
 121 |  79 | # pyenv
 122 |  80 | .python-version
 123 |  81 | 
 124 |  82 | # celery beat schedule file
 125 |  83 | celerybeat-schedule
 126 |  84 | 
 127 |  85 | # dotenv
 128 |  86 | .env
 129 |  87 | 
 130 |  88 | # virtualenv
 131 |  89 | venv/
 132 |  90 | ENV/
 133 |  91 | 
 134 |  92 | # Spyder project settings
 135 |  93 | .spyderproject
 136 |  94 | 
 137 |  95 | # Rope project settings
 138 |  96 | .ropeproject
 139 |  97 | 
 140 |  98 | *.pickle
 141 |  99 | .vscode
 142 | 100 | 
 143 | 101 | # checkpoint
 144 | 102 | *.h5
 145 | 103 | *.pkl
 146 | 104 | *.pth
 147 | 105 | 
 148 | 106 | # Mac files
 149 | 107 | .DS_Store
 150 | ```
 151 | 
 152 | ## File: F:\SomeProjects\CSGNN\generate_feature.py
 153 | 
 154 | - Extension: .py
 155 | - Language: python
 156 | - Size: 1176 bytes
 157 | - Created: 2025-08-21 17:29:04
 158 | - Modified: 2023-09-27 17:42:24
 159 | 
 160 | ### Code
 161 | 
 162 | ```python
 163 |  1 | import argparse
 164 |  2 | 
 165 |  3 | import numpy as np
 166 |  4 | from tqdm import tqdm
 167 |  5 | 
 168 |  6 | from spikenet import dataset
 169 |  7 | from spikenet.deepwalk import DeepWalk
 170 |  8 | 
 171 |  9 | parser = argparse.ArgumentParser()
 172 | 10 | parser.add_argument("--dataset", nargs="?", default="DBLP",
 173 | 11 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
 174 | 12 | parser.add_argument('--normalize', action='store_true',
 175 | 13 |                     help='Whether to normalize output embedding. (default: False)')
 176 | 14 | 
 177 | 15 | 
 178 | 16 | args = parser.parse_args()
 179 | 17 | if args.dataset.lower() == "dblp":
 180 | 18 |     data = dataset.DBLP()
 181 | 19 | elif args.dataset.lower() == "tmall":
 182 | 20 |     data = dataset.Tmall()
 183 | 21 | elif args.dataset.lower() == "patent":
 184 | 22 |     data = dataset.Patent()
 185 | 23 | else:
 186 | 24 |     raise ValueError(
 187 | 25 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
 188 | 26 | 
 189 | 27 | 
 190 | 28 | model = DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16)
 191 | 29 | xs = []
 192 | 30 | for g in tqdm(data.adj):
 193 | 31 |     model.fit(g)
 194 | 32 |     x = model.get_embedding(normalize=args.normalize)
 195 | 33 |     xs.append(x)
 196 | 34 | 
 197 | 35 | 
 198 | 36 | file_path = f'{data.root}/{data.name}/{data.name}.npy'
 199 | 37 | np.save(file_path, np.stack(xs, axis=0)) # [T, N, F]
 200 | 38 | print(f"Generated node feautures saved at {file_path}")
 201 | ```
 202 | 
 203 | ## File: F:\SomeProjects\CSGNN\LICENSE
 204 | 
 205 | - Extension: 
 206 | - Language: unknown
 207 | - Size: 1112 bytes
 208 | - Created: 2025-08-21 17:29:04
 209 | - Modified: 2023-09-27 17:42:24
 210 | 
 211 | ### Code
 212 | 
 213 | ```unknown
 214 |  1 | MIT License
 215 |  2 | 
 216 |  3 | Copyright (c) 2022 Jintang Li, Sun Yat-sen University
 217 |  4 | 
 218 |  5 | Permission is hereby granted, free of charge, to any person obtaining a copy
 219 |  6 | of this software and associated documentation files (the "Software"), to deal
 220 |  7 | in the Software without restriction, including without limitation the rights
 221 |  8 | to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 222 |  9 | copies of the Software, and to permit persons to whom the Software is
 223 | 10 | furnished to do so, subject to the following conditions:
 224 | 11 | 
 225 | 12 | The above copyright notice and this permission notice shall be included in all
 226 | 13 | copies or substantial portions of the Software.
 227 | 14 | 
 228 | 15 | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 229 | 16 | IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 230 | 17 | FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 231 | 18 | AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 232 | 19 | LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 233 | 20 | OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 234 | 21 | SOFTWARE.
 235 | ```
 236 | 
 237 | ## File: F:\SomeProjects\CSGNN\main.py
 238 | 
 239 | - Extension: .py
 240 | - Language: python
 241 | - Size: 19617 bytes
 242 | - Created: 2025-08-21 17:29:04
 243 | - Modified: 2025-09-14 16:54:48
 244 | 
 245 | ### Code
 246 | 
 247 | ```python
 248 |   1 | import argparse
 249 |   2 | import time
 250 |   3 | import os
 251 |   4 | 
 252 |   5 | import torch
 253 |   6 | import torch.nn as nn
 254 |   7 | from sklearn import metrics
 255 |   8 | from torch.utils.data import DataLoader
 256 |   9 | from tqdm import tqdm
 257 |  10 | 
 258 |  11 | from spikenet import dataset, neuron
 259 |  12 | from spikenet.layers import SAGEAggregator
 260 |  13 | from spikenet_x.model import SpikeNetX
 261 |  14 | from texttable import Texttable # Added for tab_printer
 262 |  15 | import numpy as np # Added for set_seed
 263 |  16 | 
 264 |  17 | 
 265 |  18 | def sample_subgraph(nodes, edge_index_full, num_neighbors=-1):
 266 |  19 |     """
 267 |  20 |     Samples a 1-hop subgraph for the given seed nodes, with an optional
 268 |  21 |     limit on the number of neighbors per node.
 269 |  22 |     
 270 |  23 |     Args:
 271 |  24 |         nodes (Tensor): Seed nodes of shape [B].
 272 |  25 |         edge_index_full (Tensor): Full graph edge index of shape [2, E_full].
 273 |  26 |         num_neighbors (int): Number of neighbors to sample for each node.
 274 |  27 |                              -1 means all neighbors.
 275 |  28 | 
 276 |  29 |     Returns:
 277 |  30 |         subgraph_nodes (Tensor): Unique nodes in the subgraph, shape [N_sub].
 278 |  31 |         subgraph_edge_index (Tensor): Edge index of the subgraph, shape [2, E_sub].
 279 |  32 |         nodes_local_index (Tensor): Indices of the seed nodes within subgraph_nodes.
 280 |  33 |     """
 281 |  34 |     row, col = edge_index_full
 282 |  35 |     
 283 |  36 |     # Efficiently find neighbors for each node in the batch
 284 |  37 |     sampled_neighbors = []
 285 |  38 |     if num_neighbors == -1:
 286 |  39 |         # Get all neighbors
 287 |  40 |         node_mask = torch.isin(row, nodes)
 288 |  41 |         neighbors = edge_index_full[1, node_mask]
 289 |  42 |         sampled_neighbors.append(neighbors)
 290 |  43 |     else:
 291 |  44 |         # Sample a fixed number of neighbors for each node
 292 |  45 |         for node_id in nodes:
 293 |  46 |             node_mask = (row == node_id)
 294 |  47 |             node_neighbors = col[node_mask]
 295 |  48 |             if node_neighbors.numel() > num_neighbors:
 296 |  49 |                 # Randomly sample neighbors
 297 |  50 |                 perm = torch.randperm(node_neighbors.numel(), device=nodes.device)[:num_neighbors]
 298 |  51 |                 node_neighbors = node_neighbors[perm]
 299 |  52 |             sampled_neighbors.append(node_neighbors)
 300 |  53 | 
 301 |  54 |     if sampled_neighbors:
 302 |  55 |         neighbors = torch.cat(sampled_neighbors)
 303 |  56 |         subgraph_nodes = torch.cat([nodes, neighbors]).unique()
 304 |  57 |     else:
 305 |  58 |         subgraph_nodes = nodes.unique()
 306 |  59 |     
 307 |  60 |     # This is the correct set of edges for the 1-hop subgraph
 308 |  61 |     edge_mask = torch.isin(row, nodes) & torch.isin(col, subgraph_nodes)
 309 |  62 |     subgraph_edge_index_global = edge_index_full[:, edge_mask]
 310 |  63 | 
 311 |  64 |     # Map global node indices to local indices in the subgraph
 312 |  65 |     node_map = {global_id.item(): local_id for local_id, global_id in enumerate(subgraph_nodes)}
 313 |  66 |     
 314 |  67 |     # Remap edge_index to local subgraph indices
 315 |  68 |     subgraph_edge_index = torch.tensor(
 316 |  69 |         [[node_map[src.item()] for src in subgraph_edge_index_global[0]],
 317 |  70 |          [node_map[dst.item()] for dst in subgraph_edge_index_global[1]]],
 318 |  71 |         dtype=torch.long, device=nodes.device
 319 |  72 |     )
 320 |  73 | 
 321 |  74 |     # Get local indices of the seed nodes
 322 |  75 |     nodes_local_index = torch.tensor([node_map[n.item()] for n in nodes], dtype=torch.long, device=nodes.device)
 323 |  76 | 
 324 |  77 |     return subgraph_nodes, subgraph_edge_index, nodes_local_index
 325 |  78 | 
 326 |  79 | 
 327 |  80 | def set_seed(seed):
 328 |  81 |     np.random.seed(seed)
 329 |  82 |     torch.manual_seed(seed)
 330 |  83 |     torch.cuda.manual_seed(seed)
 331 |  84 | 
 332 |  85 | def tab_printer(args):
 333 |  86 |     """Function to print the logs in a nice tabular format."""
 334 |  87 |     args = vars(args)
 335 |  88 |     keys = sorted(args.keys())
 336 |  89 |     t = Texttable() 
 337 |  90 |     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
 338 |  91 |     print(t.draw())
 339 |  92 | 
 340 |  93 | 
 341 |  94 | class SpikeNet(nn.Module):
 342 |  95 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
 343 |  96 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 344 |  97 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):
 345 |  98 | 
 346 |  99 |         super().__init__()
 347 | 100 |         
 348 | 101 |         from spikenet.utils import RandomWalkSampler, Sampler, add_selfloops
 349 | 102 |         
 350 | 103 |         tau = 1.0
 351 | 104 |         if sampler == 'rw':
 352 | 105 |             self.sampler = [RandomWalkSampler(
 353 | 106 |                 add_selfloops(adj_matrix)) for adj_matrix in data.adj]
 354 | 107 |             self.sampler_t = [RandomWalkSampler(add_selfloops(
 355 | 108 |                 adj_matrix)) for adj_matrix in data.adj_evolve]
 356 | 109 |         elif sampler == 'sage':
 357 | 110 |             self.sampler = [Sampler(add_selfloops(adj_matrix))
 358 | 111 |                             for adj_matrix in data.adj]
 359 | 112 |             self.sampler_t = [Sampler(add_selfloops(adj_matrix))
 360 | 113 |                               for adj_matrix in data.adj_evolve]
 361 | 114 |         else:
 362 | 115 |             raise ValueError(sampler)
 363 | 116 | 
 364 | 117 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 365 | 118 | 
 366 | 119 |         for hid in hids:
 367 | 120 |             aggregators.append(SAGEAggregator(in_features, hid,
 368 | 121 |                                               concat=concat, bias=bias,
 369 | 122 |                                               aggr=aggr))
 370 | 123 | 
 371 | 124 |             if act == "IF":
 372 | 125 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
 373 | 126 |             elif act == 'LIF':
 374 | 127 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
 375 | 128 |             elif act == 'PLIF':
 376 | 129 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
 377 | 130 |             else:
 378 | 131 |                 raise ValueError(act)
 379 | 132 | 
 380 | 133 |             in_features = hid * 2 if concat else hid
 381 | 134 | 
 382 | 135 |         self.aggregators = aggregators
 383 | 136 |         self.dropout = nn.Dropout(dropout)
 384 | 137 |         self.snn = snn
 385 | 138 |         self.sizes = sizes
 386 | 139 |         self.p = p
 387 | 140 |         self.pooling = nn.Linear(len(data) * in_features, out_features)
 388 | 141 | 
 389 | 142 |     def encode(self, nodes):
 390 | 143 |         spikes = []
 391 | 144 |         sizes = self.sizes
 392 | 145 |         for time_step in range(len(data)):
 393 | 146 | 
 394 | 147 |             snapshot = data[time_step]
 395 | 148 |             sampler = self.sampler[time_step]
 396 | 149 |             sampler_t = self.sampler_t[time_step]
 397 | 150 | 
 398 | 151 |             x = snapshot.x
 399 | 152 |             h = [x[nodes].to(device)]
 400 | 153 |             num_nodes = [nodes.size(0)]
 401 | 154 |             nbr = nodes
 402 | 155 |             for size in sizes:
 403 | 156 |                 size_1 = max(int(size * self.p), 1)
 404 | 157 |                 size_2 = size - size_1
 405 | 158 | 
 406 | 159 |                 if size_2 > 0:
 407 | 160 |                     nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
 408 | 161 |                     nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
 409 | 162 |                     nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
 410 | 163 |                 else:
 411 | 164 |                     nbr = sampler(nbr, size_1).view(-1)
 412 | 165 | 
 413 | 166 |                 num_nodes.append(nbr.size(0))
 414 | 167 |                 h.append(x[nbr].to(device))
 415 | 168 | 
 416 | 169 |             for i, aggregator in enumerate(self.aggregators):
 417 | 170 |                 self_x = h[:-1]
 418 | 171 |                 neigh_x = []
 419 | 172 |                 for j, n_x in enumerate(h[1:]):
 420 | 173 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
 421 | 174 | 
 422 | 175 |                 out = self.snn[i](aggregator(self_x, neigh_x))
 423 | 176 |                 if i != len(sizes) - 1:
 424 | 177 |                     out = self.dropout(out)
 425 | 178 |                     h = torch.split(out, num_nodes[:-(i + 1)])
 426 | 179 | 
 427 | 180 |             spikes.append(out)
 428 | 181 |         spikes = torch.cat(spikes, dim=1)
 429 | 182 |         neuron.reset_net(self)
 430 | 183 |         return spikes
 431 | 184 | 
 432 | 185 |     def forward(self, nodes):
 433 | 186 |         spikes = self.encode(nodes)
 434 | 187 |         return self.pooling(spikes)
 435 | 188 | 
 436 | 189 | 
 437 | 190 | parser = argparse.ArgumentParser()
 438 | 191 | parser.add_argument("--model", nargs="?", default="spikenet",
 439 | 192 |                     help="Model to use ('spikenet', 'spikenetx'). (default: spikenet)")
 440 | 193 | parser.add_argument("--dataset", nargs="?", default="DBLP",
 441 | 194 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
 442 | 195 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
 443 | 196 | parser.add_argument('--hids', type=int, nargs='+',
 444 | 197 |                     default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
 445 | 198 | parser.add_argument("--aggr", nargs="?", default="mean",
 446 | 199 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
 447 | 200 | parser.add_argument("--sampler", nargs="?", default="sage",
 448 | 201 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
 449 | 202 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
 450 | 203 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
 451 | 204 | parser.add_argument("--neuron", nargs="?", default="LIF",
 452 | 205 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
 453 | 206 | parser.add_argument('--batch_size', type=int, default=1024,
 454 | 207 |                     help='Batch size for training. (default: 1024)')
 455 | 208 | parser.add_argument('--lr', type=float, default=5e-3,
 456 | 209 |                     help='Learning rate for training. (default: 5e-3)')
 457 | 210 | parser.add_argument('--train_size', type=float, default=0.4,
 458 | 211 |                     help='Ratio of nodes for training. (default: 0.4)')
 459 | 212 | parser.add_argument('--alpha', type=float, default=1.0,
 460 | 213 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
 461 | 214 | parser.add_argument('--p', type=float, default=0.5,
 462 | 215 |                     help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
 463 | 216 | parser.add_argument('--dropout', type=float, default=0.7,
 464 | 217 |                     help='Dropout probability. (default: 0.7)')
 465 | 218 | parser.add_argument('--epochs', type=int, default=100,
 466 | 219 |                     help='Number of training epochs. (default: 100)')
 467 | 220 | parser.add_argument('--concat', action='store_true',
 468 | 221 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
 469 | 222 | parser.add_argument('--seed', type=int, default=2022,
 470 | 223 |                     help='Random seed for model. (default: 2022)')
 471 | 224 | parser.add_argument('--datapath', type=str, default='./data',
 472 | 225 |                     help='Wheres your data?, Default is ./data')
 473 | 226 | 
 474 | 227 | # SpikeNet-X specific args
 475 | 228 | parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for SpikeNet-X. (default: 4)')
 476 | 229 | parser.add_argument('--topk', type=int, default=8, help='Top-k neighbors for SpikeNet-X attention. (default: 8)')
 477 | 230 | parser.add_argument('--W', type=int, default=8, help='Time window size for SpikeNet-X. (default: 8)')
 478 | 231 | parser.add_argument('--attn_impl', type=str, default='sparse', choices=['dense','sparse'],
 479 | 232 |                     help='Attention kernel for SpikeNet-X: "dense" (fallback) or "sparse" (scales to big graphs). (default: "sparse")')
 480 | 233 | 
 481 | 234 | # 新增：模型保存、加载与测试参数
 482 | 235 | parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
 483 | 236 |                     help='Directory to save model checkpoints. (default: checkpoints)')
 484 | 237 | parser.add_argument('--resume_path', type=str, default=None,
 485 | 238 |                     help='Path to a checkpoint file to resume training from. (default: None)')
 486 | 239 | parser.add_argument('--test_model_path', type=str, default=None,
 487 | 240 |                     help='Path to a model file to load for testing only. (default: None)')
 488 | 241 | 
 489 | 242 | 
 490 | 243 | try:
 491 | 244 |     args = parser.parse_args()
 492 | 245 |     args.test_size = 1 - args.train_size
 493 | 246 |     args.train_size = args.train_size - 0.05
 494 | 247 |     args.val_size = 0.05
 495 | 248 |     args.split_seed = 42
 496 | 249 |     tab_printer(args)
 497 | 250 | except:
 498 | 251 |     parser.print_help()
 499 | 252 |     exit(0)
 500 | 253 | 
 501 | 254 | assert len(args.hids) == len(args.sizes), "must be equal!"
 502 | 255 | 
 503 | 256 | if args.dataset.lower() == "dblp":
 504 | 257 |     data = dataset.DBLP(root = args.datapath)
 505 | 258 | elif args.dataset.lower() == "tmall":
 506 | 259 |     data = dataset.Tmall(root = args.datapath)
 507 | 260 | elif args.dataset.lower() == "patent":
 508 | 261 |     data = dataset.Patent(root = args.datapath)
 509 | 262 | else:
 510 | 263 |     raise ValueError(
 511 | 264 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
 512 | 265 | 
 513 | 266 | # train:val:test
 514 | 267 | data.split_nodes(train_size=args.train_size, val_size=args.val_size,
 515 | 268 |                  test_size=args.test_size, random_state=args.split_seed)
 516 | 269 | 
 517 | 270 | set_seed(args.seed)
 518 | 271 | 
 519 | 272 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 520 | 273 | 
 521 | 274 | y = data.y.to(device)
 522 | 275 | 
 523 | 276 | train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
 524 | 277 | val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
 525 | 278 |                         pin_memory=False, batch_size=200000, shuffle=False)
 526 | 279 | test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
 527 | 280 | 
 528 | 281 | if args.model == 'spikenetx':
 529 | 282 |     
 530 | 283 |     def train_spikenetx():
 531 | 284 |         model.train()
 532 | 285 |         total_loss = 0
 533 | 286 |         # Let's use a fixed number of neighbors for now to control memory
 534 | 287 |         num_neighbors_to_sample = 10 
 535 | 288 |         for nodes in tqdm(train_loader, desc='Training'):
 536 | 289 |             nodes = nodes.to(device)
 537 | 290 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
 538 | 291 |             
 539 | 292 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
 540 | 293 |             
 541 | 294 |             optimizer.zero_grad()
 542 | 295 |             
 543 | 296 |             # The model's output `repr` and `logits` are for all nodes in the subgraph
 544 | 297 |             output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
 545 | 298 |             subgraph_logits = output['logits']
 546 | 299 | 
 547 | 300 |             # We only compute the loss on the seed nodes of the batch
 548 | 301 |             loss = loss_fn(subgraph_logits[nodes_local_index], y[nodes])
 549 | 302 |             
 550 | 303 |             loss.backward()
 551 | 304 |             optimizer.step()
 552 | 305 |             total_loss += loss.item()
 553 | 306 |         return total_loss / len(train_loader)
 554 | 307 | 
 555 | 308 |     @torch.no_grad()
 556 | 309 |     def test_spikenetx(loader):
 557 | 310 |         model.eval()
 558 | 311 |         logits_list = []
 559 | 312 |         labels_list = []
 560 | 313 |         num_neighbors_to_sample = 10 # Use the same for testing
 561 | 314 |         for nodes in tqdm(loader, desc='Testing'):
 562 | 315 |             nodes = nodes.to(device)
 563 | 316 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
 564 | 317 |             
 565 | 318 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
 566 | 319 |             
 567 | 320 |             output = model(H0_subgraph, subgraph_edge_index, time_idx_full)
 568 | 321 |             subgraph_logits = output['logits']
 569 | 322 |             
 570 | 323 |             logits_list.append(subgraph_logits[nodes_local_index].cpu())
 571 | 324 |             labels_list.append(y[nodes].cpu())
 572 | 325 |         
 573 | 326 |         logits = torch.cat(logits_list, dim=0).argmax(1)
 574 | 327 |         labels = torch.cat(labels_list, dim=0)
 575 | 328 | 
 576 | 329 |         micro = metrics.f1_score(labels, logits, average='micro', zero_division=0)
 577 | 330 |         macro = metrics.f1_score(labels, logits, average='macro', zero_division=0)
 578 | 331 |         return macro, micro
 579 | 332 |     
 580 | 333 |     # --- SpikeNet-X Training and Evaluation (with batching) ---
 581 | 334 | 
 582 | 335 |     # 1. Data Preparation (Full graph data)
 583 | 336 |     print("Preparing data for SpikeNet-X...")
 584 | 337 |     T = len(data)
 585 | 338 |     N = data.num_nodes
 586 | 339 |     d_in = data.num_features
 587 | 340 |     
 588 | 341 |     edge_list = [snapshot.edge_index for snapshot in data]
 589 | 342 |     edge_index_full = torch.unique(torch.cat(edge_list, dim=1), dim=1).to(device)
 590 | 343 |     H0_full = torch.stack([snapshot.x for snapshot in data], dim=0).to(device)
 591 | 344 |     time_idx_full = torch.arange(T, device=device)
 592 | 345 | 
 593 | 346 |     # 2. Model, Optimizer, Loss
 594 | 347 |     model = SpikeNetX(
 595 | 348 |         d_in=d_in,
 596 | 349 |         d=args.hids[0],
 597 | 350 |         layers=len(args.sizes),
 598 | 351 |         heads=args.heads,
 599 | 352 |         out_dim=data.num_classes,
 600 | 353 |         topk=args.topk,
 601 | 354 |         W=args.W,
 602 | 355 |         attn_impl=args.attn_impl
 603 | 356 |     ).to(device)
 604 | 357 | 
 605 | 358 |     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
 606 | 359 |     loss_fn = nn.CrossEntropyLoss()
 607 | 360 |     
 608 | 361 |     
 609 | 362 |     # --- Test-only mode ---
 610 | 363 |     if args.test_model_path:
 611 | 364 |         print(f"Loading model from {args.test_model_path} for testing...")
 612 | 365 |         checkpoint = torch.load(args.test_model_path, map_location=device)
 613 | 366 |         model.load_state_dict(checkpoint['model_state_dict'])
 614 | 367 |         test_macro, test_micro = test_spikenetx(test_loader)
 615 | 368 |         print(f"Test Results: Macro-F1={test_macro:.4f}, Micro-F1={test_micro:.4f}")
 616 | 369 |         exit(0)
 617 | 370 | 
 618 | 371 | 
 619 | 372 | 
 620 | 373 |     # 3. Training Loop
 621 | 374 |     start_epoch = 1
 622 | 375 |     best_val_metric = 0
 623 | 376 |     best_test_metric = (0, 0)
 624 | 377 | 
 625 | 378 |     # --- Resume from checkpoint ---
 626 | 379 |     if args.resume_path:
 627 | 380 |         if os.path.exists(args.resume_path):
 628 | 381 |             print(f"Resuming training from {args.resume_path}...")
 629 | 382 |             checkpoint = torch.load(args.resume_path, map_location=device)
 630 | 383 |             model.load_state_dict(checkpoint['model_state_dict'])
 631 | 384 |             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 632 | 385 |             start_epoch = checkpoint['epoch'] + 1
 633 | 386 |             best_val_metric = checkpoint.get('best_val_metric', 0) # Use .get for backward compatibility
 634 | 387 |             print(f"Resumed from epoch {start_epoch-1}. Best val metric so far: {best_val_metric:.4f}")
 635 | 388 |         else:
 636 | 389 |             print(f"Warning: Checkpoint path {args.resume_path} not found. Starting from scratch.")
 637 | 390 | 
 638 | 391 |     print("Starting SpikeNet-X training...")
 639 | 392 |     start = time.time()
 640 | 393 |     for epoch in range(start_epoch, args.epochs + 1):
 641 | 394 |         train_spikenetx()
 642 | 395 |         val_metric = test_spikenetx(val_loader)
 643 | 396 |         test_metric = test_spikenetx(test_loader)
 644 | 397 |         
 645 | 398 |         is_best = val_metric[1] > best_val_metric
 646 | 399 |         if is_best:
 647 | 400 |             best_val_metric = val_metric[1]
 648 | 401 |             best_test_metric = test_metric
 649 | 402 | 
 650 | 403 |             # --- Save checkpoint ---
 651 | 404 |             os.makedirs(args.checkpoint_dir, exist_ok=True)
 652 | 405 |             checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.dataset}.pth')
 653 | 406 |             torch.save({
 654 | 407 |                 'epoch': epoch,
 655 | 408 |                 'model_state_dict': model.state_dict(),
 656 | 409 |                 'optimizer_state_dict': optimizer.state_dict(),
 657 | 410 |                 'best_val_metric': best_val_metric,
 658 | 411 |                 'test_metric_at_best_val': test_metric,
 659 | 412 |             }, checkpoint_path)
 660 | 413 |             print(f"Epoch {epoch:03d}: New best model saved to {checkpoint_path} with Val Micro: {best_val_metric:.4f}")
 661 | 414 | 
 662 | 415 |         end = time.time()
 663 | 416 |         print(
 664 | 417 |             f'Epoch: {epoch:03d}, Val Micro: {val_metric[1]:.4f}, Test Micro: {test_metric[1]:.4f}, '
 665 | 418 |             f'Best Test: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time: {end-start:.2f}s'
 666 | 419 |         )
 667 | 420 | 
 668 | 421 | else:
 669 | 422 |     # --- Original SpikeNet Training and Evaluation ---
 670 | 423 |     model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
 671 | 424 |                      dropout=args.dropout, sampler=args.sampler, p=args.p,
 672 | 425 |                      aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
 673 | 426 |                      hids=args.hids, act=args.neuron, bias=True).to(device)
 674 | 427 | 
 675 | 428 |     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
 676 | 429 |     loss_fn = nn.CrossEntropyLoss()
 677 | 430 | 
 678 | 431 |     def train():
 679 | 432 |         model.train()
 680 | 433 |         for nodes in tqdm(train_loader, desc='Training'):
 681 | 434 |             optimizer.zero_grad()
 682 | 435 |             loss_fn(model(nodes), y[nodes]).backward()
 683 | 436 |             optimizer.step()
 684 | 437 | 
 685 | 438 |     @torch.no_grad()
 686 | 439 |     def test(loader):
 687 | 440 |         model.eval()
 688 | 441 |         logits = []
 689 | 442 |         labels = []
 690 | 443 |         for nodes in loader:
 691 | 444 |             logits.append(model(nodes))
 692 | 445 |             labels.append(y[nodes])
 693 | 446 |         logits = torch.cat(logits, dim=0).cpu()
 694 | 447 |         labels = torch.cat(labels, dim=0).cpu()
 695 | 448 |         logits = logits.argmax(1)
 696 | 449 |         metric_macro = metrics.f1_score(labels, logits, average='macro')
 697 | 450 |         metric_micro = metrics.f1_score(labels, logits, average='micro')
 698 | 451 |         return metric_macro, metric_micro
 699 | 452 | 
 700 | 453 |     best_val_metric = test_metric = 0
 701 | 454 |     start = time.time()
 702 | 455 |     for epoch in range(1, args.epochs + 1):
 703 | 456 |         train()
 704 | 457 |         val_metric, test_metric = test(val_loader), test(test_loader)
 705 | 458 |         if val_metric[1] > best_val_metric:
 706 | 459 |             best_val_metric = val_metric[1]
 707 | 460 |             best_test_metric = test_metric
 708 | 461 |         end = time.time()
 709 | 462 |         print(
 710 | 463 |             f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
 711 | 464 | 
 712 | 465 |     # save bianry node embeddings (spikes)
 713 | 466 |     # emb = model.encode(torch.arange(data.num_nodes)).cpu()
 714 | 467 |     # torch.save(emb, 'emb.pth')
 715 | ```
 716 | 
 717 | ## File: F:\SomeProjects\CSGNN\main_static.py
 718 | 
 719 | - Extension: .py
 720 | - Language: python
 721 | - Size: 8038 bytes
 722 | - Created: 2025-08-21 17:29:04
 723 | - Modified: 2023-09-27 17:42:24
 724 | 
 725 | ### Code
 726 | 
 727 | ```python
 728 |   1 | import argparse
 729 |   2 | import os.path as osp
 730 |   3 | import time
 731 |   4 | 
 732 |   5 | import torch
 733 |   6 | import torch.nn as nn
 734 |   7 | from sklearn import metrics
 735 |   8 | from spikenet import dataset, neuron
 736 |   9 | from spikenet.layers import SAGEAggregator
 737 |  10 | from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
 738 |  11 |                             set_seed, tab_printer)
 739 |  12 | from torch.utils.data import DataLoader
 740 |  13 | from torch_geometric.datasets import Flickr, Reddit
 741 |  14 | from torch_geometric.utils import to_scipy_sparse_matrix
 742 |  15 | from tqdm import tqdm
 743 |  16 | 
 744 |  17 | 
 745 |  18 | class SpikeNet(nn.Module):
 746 |  19 |     def __init__(self, in_features, out_features, hids=[32], alpha=1.0, T=5,
 747 |  20 |                  dropout=0.7, bias=True, aggr='mean', sampler='sage',
 748 |  21 |                  surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):
 749 |  22 | 
 750 |  23 |         super().__init__()
 751 |  24 | 
 752 |  25 |         tau = 1.0
 753 |  26 |         if sampler == 'rw':
 754 |  27 |             self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
 755 |  28 |         elif sampler == 'sage':
 756 |  29 |             self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
 757 |  30 |         else:
 758 |  31 |             raise ValueError(sampler)
 759 |  32 | 
 760 |  33 |         del data.edge_index
 761 |  34 | 
 762 |  35 |         aggregators, snn = nn.ModuleList(), nn.ModuleList()
 763 |  36 | 
 764 |  37 |         for hid in hids:
 765 |  38 |             aggregators.append(SAGEAggregator(in_features, hid,
 766 |  39 |                                               concat=concat, bias=bias,
 767 |  40 |                                               aggr=aggr))
 768 |  41 | 
 769 |  42 |             if act == "IF":
 770 |  43 |                 snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
 771 |  44 |             elif act == 'LIF':
 772 |  45 |                 snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
 773 |  46 |             elif act == 'PLIF':
 774 |  47 |                 snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
 775 |  48 |             else:
 776 |  49 |                 raise ValueError(act)
 777 |  50 | 
 778 |  51 |             in_features = hid * 2 if concat else hid
 779 |  52 | 
 780 |  53 |         self.aggregators = aggregators
 781 |  54 |         self.dropout = nn.Dropout(dropout)
 782 |  55 |         self.snn = snn
 783 |  56 |         self.sizes = sizes
 784 |  57 |         self.T = T
 785 |  58 |         self.pooling = nn.Linear(T * in_features, out_features)
 786 |  59 | 
 787 |  60 |     def encode(self, nodes):
 788 |  61 |         spikes = []
 789 |  62 |         sizes = self.sizes
 790 |  63 |         x = data.x
 791 |  64 | 
 792 |  65 |         for time_step in range(self.T):
 793 |  66 |             h = [x[nodes].to(device)]
 794 |  67 |             num_nodes = [nodes.size(0)]
 795 |  68 |             nbr = nodes
 796 |  69 |             for size in sizes:
 797 |  70 |                 nbr = self.sampler(nbr, size)
 798 |  71 |                 num_nodes.append(nbr.size(0))
 799 |  72 |                 h.append(x[nbr].to(device))
 800 |  73 | 
 801 |  74 |             for i, aggregator in enumerate(self.aggregators):
 802 |  75 |                 self_x = h[:-1]
 803 |  76 |                 neigh_x = []
 804 |  77 |                 for j, n_x in enumerate(h[1:]):
 805 |  78 |                     neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
 806 |  79 | 
 807 |  80 |                 out = self.snn[i](aggregator(self_x, neigh_x))
 808 |  81 |                 if i != len(sizes) - 1:
 809 |  82 |                     out = self.dropout(out)
 810 |  83 |                     h = torch.split(out, num_nodes[:-(i + 1)])
 811 |  84 | 
 812 |  85 |             spikes.append(out)
 813 |  86 |         spikes = torch.cat(spikes, dim=1)
 814 |  87 |         neuron.reset_net(self)
 815 |  88 |         return spikes
 816 |  89 | 
 817 |  90 |     def forward(self, nodes):
 818 |  91 |         spikes = self.encode(nodes)
 819 |  92 |         return self.pooling(spikes)
 820 |  93 | 
 821 |  94 | 
 822 |  95 | parser = argparse.ArgumentParser()
 823 |  96 | parser.add_argument("--dataset", nargs="?", default="flickr",
 824 |  97 |                     help="Datasets (Reddit and Flickr only). (default: Flickr)")
 825 |  98 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2],
 826 |  99 |                     help='Neighborhood sampling size for each layer. (default: [5, 2])')
 827 | 100 | parser.add_argument('--hids', type=int, nargs='+',
 828 | 101 |                     default=[512, 10], help='Hidden units for each layer. (default: [128, 10])')
 829 | 102 | parser.add_argument("--aggr", nargs="?", default="mean",
 830 | 103 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
 831 | 104 | parser.add_argument("--sampler", nargs="?", default="sage",
 832 | 105 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
 833 | 106 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
 834 | 107 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
 835 | 108 | parser.add_argument("--neuron", nargs="?", default="LIF",
 836 | 109 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
 837 | 110 | parser.add_argument('--batch_size', type=int, default=2048,
 838 | 111 |                     help='Batch size for training. (default: 1024)')
 839 | 112 | parser.add_argument('--lr', type=float, default=5e-3,
 840 | 113 |                     help='Learning rate for training. (default: 5e-3)')
 841 | 114 | parser.add_argument('--alpha', type=float, default=1.0,
 842 | 115 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
 843 | 116 | parser.add_argument('--T', type=int, default=15,
 844 | 117 |                     help='Number of time steps. (default: 15)')
 845 | 118 | parser.add_argument('--dropout', type=float, default=0.5,
 846 | 119 |                     help='Dropout probability. (default: 0.5)')
 847 | 120 | parser.add_argument('--epochs', type=int, default=100,
 848 | 121 |                     help='Number of training epochs. (default: 100)')
 849 | 122 | parser.add_argument('--concat', action='store_true',
 850 | 123 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
 851 | 124 | parser.add_argument('--seed', type=int, default=2022,
 852 | 125 |                     help='Random seed for model. (default: 2022)')
 853 | 126 | 
 854 | 127 | 
 855 | 128 | try:
 856 | 129 |     args = parser.parse_args()
 857 | 130 |     args.split_seed = 42
 858 | 131 |     tab_printer(args)
 859 | 132 | except:
 860 | 133 |     parser.print_help()
 861 | 134 |     exit(0)
 862 | 135 | 
 863 | 136 | assert len(args.hids) == len(args.sizes), "must be equal!"
 864 | 137 | 
 865 | 138 | root = "data/"  # Specify your root path
 866 | 139 | 
 867 | 140 | if args.dataset.lower() == "reddit":
 868 | 141 |     dataset = Reddit(osp.join(root, 'Reddit'))
 869 | 142 |     data = dataset[0]
 870 | 143 | elif args.dataset.lower() == "flickr":
 871 | 144 |     dataset = Flickr(osp.join(root, 'Flickr'))
 872 | 145 |     data = dataset[0]
 873 | 146 |     
 874 | 147 | data.x = torch.nn.functional.normalize(data.x, dim=1)
 875 | 148 | 
 876 | 149 | set_seed(args.seed)
 877 | 150 | 
 878 | 151 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 879 | 152 | 
 880 | 153 | y = data.y.to(device)
 881 | 154 | 
 882 | 155 | train_loader = DataLoader(data.train_mask.nonzero().view(-1), pin_memory=False, batch_size=args.batch_size, shuffle=True)
 883 | 156 | val_loader = DataLoader(data.val_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
 884 | 157 | test_loader = DataLoader(data.test_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
 885 | 158 | 
 886 | 159 | 
 887 | 160 | model = SpikeNet(dataset.num_features, dataset.num_classes, alpha=args.alpha,
 888 | 161 |                  dropout=args.dropout, sampler=args.sampler, T=args.T,
 889 | 162 |                  aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
 890 | 163 |                  hids=args.hids, act=args.neuron, bias=True).to(device)
 891 | 164 | 
 892 | 165 | optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
 893 | 166 | loss_fn = nn.CrossEntropyLoss()
 894 | 167 | 
 895 | 168 | 
 896 | 169 | def train():
 897 | 170 |     model.train()
 898 | 171 |     for nodes in tqdm(train_loader, desc='Training'):
 899 | 172 |         optimizer.zero_grad()
 900 | 173 |         loss_fn(model(nodes), y[nodes]).backward()
 901 | 174 |         optimizer.step()
 902 | 175 | 
 903 | 176 | 
 904 | 177 | @torch.no_grad()
 905 | 178 | def test(loader):
 906 | 179 |     model.eval()
 907 | 180 |     logits = []
 908 | 181 |     labels = []
 909 | 182 |     for nodes in loader:
 910 | 183 |         logits.append(model(nodes))
 911 | 184 |         labels.append(y[nodes])
 912 | 185 |     logits = torch.cat(logits, dim=0).cpu()
 913 | 186 |     labels = torch.cat(labels, dim=0).cpu()
 914 | 187 |     logits = logits.argmax(1)
 915 | 188 |     metric_macro = metrics.f1_score(labels, logits, average='macro')
 916 | 189 |     metric_micro = metrics.f1_score(labels, logits, average='micro')
 917 | 190 |     return metric_macro, metric_micro
 918 | 191 | 
 919 | 192 | 
 920 | 193 | best_val_metric = test_metric = 0
 921 | 194 | start = time.time()
 922 | 195 | for epoch in range(1, args.epochs + 1):
 923 | 196 |     train()
 924 | 197 |     val_metric, test_metric = test(val_loader), test(test_loader)
 925 | 198 |     if val_metric[1] > best_val_metric:
 926 | 199 |         best_val_metric = val_metric[1]
 927 | 200 |         best_test_metric = test_metric
 928 | 201 |     end = time.time()
 929 | 202 |     print(
 930 | 203 |         f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
 931 | 204 | 
 932 | 205 | # save bianry node embeddings (spikes)
 933 | 206 | # emb = model.encode(torch.arange(data.num_nodes)).cpu()
 934 | 207 | # torch.save(emb, 'emb.pth')
 935 | ```
 936 | 
 937 | ## File: F:\SomeProjects\CSGNN\README.md
 938 | 
 939 | - Extension: .md
 940 | - Language: markdown
 941 | - Size: 8506 bytes
 942 | - Created: 2025-08-21 17:29:04
 943 | - Modified: 2025-09-14 15:06:45
 944 | 
 945 | ### Code
 946 | 
 947 | ```markdown
 948 |   1 | # Abstract
 949 |   2 | 
 950 |   3 | Recent years have seen a surge in research on dynamic graph representation learning, which aims to model temporal graphs that are dynamic and evolving constantly over time. However, current work typically models graph dynamics with recurrent neural networks (RNNs), making them suffer seriously from computation and memory overheads on large temporal graphs. So far, scalability of dynamic graph representation learning on large temporal graphs remains one of the major challenges. In this paper, we present a scalable framework, namely SpikeNet, to efficiently capture the temporal and structural patterns of temporal graphs. We explore a new direction in that we can capture the evolving dynamics of temporal graphs with spiking neural networks (SNNs) instead of RNNs. As a low-power alternative to RNNs, SNNs explicitly model graph dynamics as spike trains of neuron populations and enable spike-based propagation in an efficient way. Experiments on three large real-world temporal graph datasets demonstrate that SpikeNet outperforms strong baselines on the temporal node classification task with lower computational costs. Particularly, SpikeNet generalizes to a large temporal graph (2M nodes and 13M edges) with significantly fewer parameters and computation overheads.
 951 |   4 | 
 952 |   5 | # Dataset
 953 |   6 | 
 954 |   7 | ## Overview
 955 |   8 | 
 956 |   9 | |             | DBLP    | Tmall     | Patent     |
 957 |  10 | | ----------- | ------- | --------- | ---------- |
 958 |  11 | | #nodes      | 28,085  | 577,314   | 2,738,012  |
 959 |  12 | | #edges      | 236,894 | 4,807,545 | 13,960,811 |
 960 |  13 | | #time steps | 27      | 186       | 25         |
 961 |  14 | | #classes    | 10      | 5         | 6          |
 962 |  15 | 
 963 |  16 | ## Download datasets
 964 |  17 | 
 965 |  18 | + DBLP
 966 |  19 | + Tmall
 967 |  20 | + Patent
 968 |  21 | 
 969 |  22 | All dataset can be found at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0).
 970 |  23 | You can download the datasets and put them in the folder `data/`, e.g., `data/dblp`.
 971 |  24 | 
 972 |  25 | ## (Optional) Re-generate node features via DeepWalk
 973 |  26 | 
 974 |  27 | Since these datasets have no associated node features, we have generated node features via unsupervised DeepWalk method (saved as `.npy` format).
 975 |  28 | You can find them at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0) as well.
 976 |  29 | Only `dblp.npy` is uploaded due to size limit of Dropbox.
 977 |  30 | 
 978 |  31 | (Update) The generated node features for Tmall and Patent datasets have been shared through Aliyun Drive, and the link is as follows: https://www.aliyundrive.com/s/LH9qa9XZmXa.
 979 |  32 | 
 980 |  33 | Note: Since Aliyun Drive does not support direct sharing of npy files, you will need to manually change the file extension `.txt` to `.npy` after downloading.
 981 |  34 | 
 982 |  35 | We also provide the script to generate the node features. Alternatively, you can generate them on your end (this will take about minutes to hours):
 983 |  36 | 
 984 |  37 | ```bash
 985 |  38 | python generate_feature.py --dataset dblp
 986 |  39 | python generate_feature.py --dataset tmall --normalize
 987 |  40 | python generate_feature.py --dataset patent --normalize
 988 |  41 | ```
 989 |  42 | 
 990 |  43 | ## Overall file structure
 991 |  44 | 
 992 |  45 | ```bash
 993 |  46 | SpikeNet
 994 |  47 | ├── data
 995 |  48 | │   ├── dblp
 996 |  49 | │   │   ├── dblp.npy
 997 |  50 | │   │   ├── dblp.txt
 998 |  51 | │   │   └── node2label.txt
 999 |  52 | │   ├── tmall
1000 |  53 | │   │   ├── tmall.npy
1001 |  54 | │   │   └── tmall.txt
1002 |  55 | │   │   ├── node2label.txt
1003 |  56 | │   ├── patent
1004 |  57 | │   │   ├── patent_edges.json
1005 |  58 | │   │   ├── patent_nodes.json
1006 |  59 | │   │   └── patent.npy
1007 |  60 | ├── figs
1008 |  61 | │   └── spikenet.png
1009 |  62 | ├── spikenet
1010 |  63 | │   ├── ...
1011 |  64 | ├── spikenet_x
1012 |  65 | │   ├── __init__.py
1013 |  66 | │   ├── delayline.py
1014 |  67 | │   ├── lif_cell.py
1015 |  68 | │   ├── masked_ops.py
1016 |  69 | │   ├── minimal_example.py
1017 |  70 | │   ├── model.py
1018 |  71 | │   ├── rel_time.py
1019 |  72 | │   ├── spikenetx_layer.py
1020 |  73 | │   └── sta.py
1021 |  74 | ├── generate_feature.py
1022 |  75 | ├── main.py
1023 |  76 | ├── main_static.py
1024 |  77 | ├── README.md
1025 |  78 | ├── setup.py
1026 |  79 | ```
1027 |  80 | 
1028 |  81 | # Requirements
1029 |  82 | 
1030 |  83 | ```
1031 |  84 | gensim==4.2.0
1032 |  85 | numba==0.61.2
1033 |  86 | numpy==1.25.2
1034 |  87 | scikit_learn==1.1.3
1035 |  88 | scipy==1.16.2
1036 |  89 | setuptools==68.2.2
1037 |  90 | texttable==1.7.0
1038 |  91 | torch==1.13.0+cu117
1039 |  92 | torch_cluster==1.6.3
1040 |  93 | torch_geometric==2.6.1
1041 |  94 | torch_scatter==2.1.0+pt113cu117
1042 |  95 | tqdm==4.67.1
1043 |  96 | ```
1044 |  97 | 
1045 |  98 | In fact, the version of these packages does not have to be consistent to ours. For example, Pytorch 1.6~-1.12 should also work.
1046 |  99 | 
1047 | 100 | # Usage
1048 | 101 | 
1049 | 102 | ## Build neighborhood sampler
1050 | 103 | 
1051 | 104 | ```bash
1052 | 105 | python setup.py install
1053 | 106 | ```
1054 | 107 | 
1055 | 108 | ## Run SpikeNet
1056 | 109 | 
1057 | 110 | ```bash
1058 | 111 | # DBLP
1059 | 112 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4
1060 | 113 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.6
1061 | 114 | python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.8
1062 | 115 | 
1063 | 116 | # Tmall
1064 | 117 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.4
1065 | 118 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.6
1066 | 119 | python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.8
1067 | 120 | 
1068 | 121 | # Patent
1069 | 122 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.4
1070 | 123 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.6
1071 | 124 | python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 0.5 --train_size 0.8
1072 | 125 | ```
1073 | 126 | 
1074 | 127 | ## Run SpikeNet-X
1075 | 128 | 
1076 | 129 | SpikeNet-X is a new model variant that uses temporal attention. You can run it by specifying `--model spikenetx`.
1077 | 130 | 
1078 | 131 | ### Basic Training
1079 | 132 | 
1080 | 133 | This command starts a standard training run from scratch
1081 | 134 | 
1082 | 135 | ```bash
1083 | 136 | # DBLP Example
1084 | 137 | python main.py --model spikenetx --dataset dblp --hids 64 --sizes 2 --epochs 100 --lr 0.005 --heads 4 --topk 8 --W 8
1085 | 138 | ```
1086 | 139 | 
1087 | 140 | ### Training Workflow (with Checkpoints)
1088 | 141 | 
1089 | 142 | The training script now supports saving, resuming, and testing.
1090 | 143 | 
1091 | 144 | **1. Standard Training & Saving**
1092 | 145 | 
1093 | 146 | When you run a standard training, the script will automatically save the model with the best validation performance to the `checkpoints/` directory (or the directory specified by `--checkpoint_dir`).
1094 | 147 | 
1095 | 148 | ```bash
1096 | 149 | # The best model will be saved as checkpoints/best_model_DBLP.pth
1097 | 150 | python main.py --model spikenetx --dataset DBLP --epochs 100
1098 | 151 | ```
1099 | 152 | 
1100 | 153 | **2. Resuming Training**
1101 | 154 | 
1102 | 155 | If your training was interrupted, you can resume from the last saved checkpoint using the `--resume_path` argument.
1103 | 156 | 
1104 | 157 | ```bash
1105 | 158 | # This will load the model, optimizer, and epoch number from the checkpoint and continue training.
1106 | 159 | python main.py --model spikenetx --dataset DBLP --epochs 100 --resume_path checkpoints/best_model_DBLP.pth
1107 | 160 | ```
1108 | 161 | 
1109 | 162 | **3. Testing a Model**
1110 | 163 | 
1111 | 164 | To evaluate a trained model on the test set without running the full training loop, use the `--test_model_path` argument.
1112 | 165 | 
1113 | 166 | ```bash
1114 | 167 | # This will load the model, run evaluation on the test set, and print the results.
1115 | 168 | python main.py --model spikenetx --dataset DBLP --test_model_path checkpoints/best_model_DBLP.pth
1116 | 169 | ```
1117 | 170 | 
1118 | 171 | # On the extention to stastic graphs
1119 | 172 | 
1120 | 173 | Actually, SpikeNet is not only applicaple for temporal graphs, it is also straightforward to extend to stastic graphs by defining a time step hyperparameter $T$ manually.
1121 | 174 | In this way, the sampled subgraph at each time step naturally form graph snapshot. We can use SpikeNet to capture the *evolving* dynamics of sampled subgraphs.
1122 | 175 | Due to space limit, we did not discuss this part in our paper. However, we believe this is indeed necessary to show the effectiveness of our work.
1123 | 176 | 
1124 | 177 | We provide a simple example for the usage on stastic graphs datasets `Flickr` and `Reddit` (be sure you have PyTorch Geometric installed):
1125 | 178 | 
1126 | 179 | ```bash
1127 | 180 | # Flickr
1128 | 181 | python main_static.py --dataset flickr --surrogate super
1129 | 182 | 
1130 | 183 | # Reddit
1131 | 184 | python main_static.py --dataset reddit --surrogate super
1132 | 185 | ```
1133 | 186 | 
1134 | 187 | We report Micro-F1 score and the results are as follows:
1135 | 188 | 
1136 | 189 | | Method     | Flickr       | Reddit       |
1137 | 190 | | ---------- | ------------ | ------------ |
1138 | 191 | | GCN        | 0.492±0.003 | 0.933±0.000 |
1139 | 192 | | GraphSAGE  | 0.501±0.013 | 0.953±0.001 |
1140 | 193 | | FastGCN    | 0.504±0.001 | 0.924±0.001 |
1141 | 194 | | S-GCN      | 0.482±0.003 | 0.964±0.001 |
1142 | 195 | | AS-GCN     | 0.504±0.002 | 0.958±0.001 |
1143 | 196 | | ClusterGCN | 0.481±0.005 | 0.954±0.001 |
1144 | 197 | | GraphSAINT | 0.511±0.001 | 0.966±0.001 |
1145 | 198 | | SpikeNet   | 0.515±0.003 | 0.953±0.001 |
1146 | 199 | 
1147 | 200 | # Reference
1148 | 201 | 
1149 | 202 | ```bibtex
1150 | 203 | @inproceedings{li2023scaling,
1151 | 204 |   author    = {Jintang Li and
1152 | 205 |                Zhouxin Yu and
1153 | 206 |                Zulun Zhu and
1154 | 207 |                Liang Chen and
1155 | 208 |                Qi Yu and
1156 | 209 |                Zibin Zheng and
1157 | 210 |                Sheng Tian and
1158 | 211 |                Ruofan Wu and
1159 | 212 |                Changhua Meng},
1160 | 213 |   title     = {Scaling Up Dynamic Graph Representation Learning via Spiking Neural
1161 | 214 |                Networks},
1162 | 215 |   booktitle = {{AAAI}},
1163 | 216 |   pages     = {8588--8596},
1164 | 217 |   publisher = {{AAAI} Press},
1165 | 218 |   year      = {2023}
1166 | 219 | }
1167 | 220 | ```
1168 | ```
1169 | 
1170 | ## File: F:\SomeProjects\CSGNN\setup.py
1171 | 
1172 | - Extension: .py
1173 | - Language: python
1174 | - Size: 327 bytes
1175 | - Created: 2025-08-21 17:29:04
1176 | - Modified: 2023-09-27 17:42:24
1177 | 
1178 | ### Code
1179 | 
1180 | ```python
1181 |  1 | from setuptools import setup
1182 |  2 | from torch.utils.cpp_extension import BuildExtension, CppExtension
1183 |  3 | 
1184 |  4 | setup(
1185 |  5 |     name="sample_neighber",
1186 |  6 |     ext_modules=[
1187 |  7 |         CppExtension("sample_neighber", sources=["spikenet/sample_neighber.cpp"], extra_compile_args=['-g']),
1188 |  8 | 
1189 |  9 |     ],
1190 | 10 |     cmdclass={
1191 | 11 |         "build_ext": BuildExtension
1192 | 12 |     }
1193 | 13 | )
1194 | ```
1195 | 
1196 | ## File: F:\SomeProjects\CSGNN\提示词.md
1197 | 
1198 | - Extension: .md
1199 | - Language: markdown
1200 | - Size: 14446 bytes
1201 | - Created: 2025-08-22 12:38:17
1202 | - Modified: 2025-08-22 12:38:19
1203 | 
1204 | ### Code
1205 | 
1206 | ```markdown
1207 |   1 | # SpikeNet-X 技术规格（实现提示词）
1208 |   2 | 
1209 |   3 | ## 0. 目标与总览
1210 |   4 | **目标**：在动态图/时序图上，用事件驱动的**脉冲时序注意力聚合（STA）**替换传统时间池化，并在其前端串联**可学习多延迟通路（DelayLine）**，显式建模关系传播的**时间错位与传播时延**，保持 SNN 的稀疏事件驱动与可解释性。
1211 |   5 | 
1212 |   6 | **层级结构（自上而下）**  
1213 |   7 | 1) **DelayLine**：沿时间维的**因果深度可分离 1D 卷积**（K 个离散延迟 tap），学习不同延迟的响应；  
1214 |   8 | 2) **STA-GNN Aggregator**：对邻居在过去时间的事件序列进行**因果多头注意力**，以**相对时间编码**与**脉冲门控**（surrogate）做软选择，并用 **Top-k 稀疏化**控制成本；  
1215 |   9 | 3) **脉冲单元（LIF/GLIF）**：将上一步的聚合消息注入膜电位，阈值比较产生脉冲，使用代理梯度反传。
1216 |  10 | 
1217 |  11 | ---
1218 |  12 | 
1219 |  13 | ## 1. 记号与输入输出约定
1220 |  14 | 
1221 |  15 | - 时长 \(T\)、节点数 \(N\)、特征维 \(d_\text{in}\)、隐藏维 \(d\)、注意力头数 \(H\)、延迟 tap 数 \(K\)、相对时间编码维 \(d_\text{pe}\)。  
1222 |  16 | - 默认采用**时间优先格式**：`H_tn` 表示 `[T, N, d]`；脉冲张量 `S` 为 `[T, N]`（0/1 或 {0,1} 的浮点）。  
1223 |  17 | - 图结构用 `edge_index: LongTensor[2, E]`（PyG 风格），或邻接稀疏矩阵。若使用稠密掩码 `adj_mask: Bool[N, N]` 亦可。  
1224 |  18 | - **Batch**：建议将多图拼成大图，用 `batch: LongTensor[N_total]` 指示归属；若必须 `B×N×T×d`，可在外层再套一维 batch 并展开为大图。  
1225 |  19 | - **模块 API（核心）**：
1226 |  20 |   - `LearnableDelayLine.forward(H: [T, N, d_in]) -> H̃: [T, N, d_in]`
1227 |  21 |   - `SpikingTemporalAttention.forward(H̃: [T,N,d_qkv], S: [T,N], edge_index or adj_mask, time_idx: [T]) -> M: [T,N,d]`
1228 |  22 |   - `LIFCell.forward(M: [T,N,d], state0) -> (S: [T,N], V: [T,N], aux)`  
1229 |  23 |   其中 `d_qkv` 与 `d` 可相同或通过线性层投影。
1230 |  24 | 
1231 |  25 | ---
1232 |  26 | 
1233 |  27 | ## 2. DelayLine（可学习多延迟通路）
1234 |  28 | 
1235 |  29 | ### 原理
1236 |  30 | 对每个节点、每个通道沿时间做**因果卷积**，学习对不同**传播时延**的权重偏好。  
1237 |  31 | \[
1238 |  32 | \tilde{h}_{t}^{(c)}=\sum_{k=0}^{K-1} w_k^{(c)} \cdot h_{t-k}^{(c)},\quad 
1239 |  33 | w_k^{(c)}=\frac{\text{softplus}(u_k^{(c)})}{\sum_r \text{softplus}(u_r^{(c)})}\cdot \rho^k,\ \rho\in(0,1)
1240 |  34 | \]
1241 |  35 | - `softplus`+**归一化**保证稳定、可解释（权重非负，且随延迟指数折扣 \(\rho^k\)）。  
1242 |  36 | - 支持**通道共享**（每层统一一组 \(\{w_k\}\)）或**逐通道**（推荐：逐通道但分组实现）。
1243 |  37 | 
1244 |  38 | ### 接口与形状
1245 |  39 | - 入：`H: Float[T, N, d_in]`  
1246 |  40 | - 出：`H_tilde: Float[T, N, d_in]`（与输入同形）  
1247 |  41 | - 可选参数：`K=3~7`，`rho=0.85`，`per_channel=True`，`causal_pad='left'`。
1248 |  42 | 
1249 |  43 | ### 计算与实现要点
1250 |  44 | - **实现**：等价 `groups=d_in` 的 `Conv1d`（深度可分离），输入先转置为 `[N*d_in, 1, T]` 做分组 1D 卷积后还原；或用自定义 `causal_depthwise_conv1d`。  
1251 |  45 | - **复杂度**：\(O(T\cdot N\cdot d_\text{in}\cdot K)\)。  
1252 |  46 | - **正则**（可选）：`L1(w)` 或 `Entropy(w)` 促进稀疏/清晰峰值。  
1253 |  47 | - **边界**：对 `t<0` 采用零填充。
1254 |  48 | 
1255 |  49 | ---
1256 |  50 | 
1257 |  51 | ## 3. STA-GNN（脉冲时序注意力聚合）
1258 |  52 | 
1259 |  53 | ### 核心思想
1260 |  54 | 在**因果掩码**与**邻接掩码**下，使用**相对时间编码**的多头注意力，让每个节点在时刻 \(t\) 选择来自邻居在**过去各时刻 \(t' \le t\)** 的关键事件，并用**脉冲发放**对注意力进行**门控**，保持事件驱动与可导性。
1261 |  55 | 
1262 |  56 | ### 相对时间编码 \(\phi(\Delta t)\)
1263 |  57 | - 组合基函数（推荐维度 \(d_\text{pe}=8\)）：  
1264 |  58 |   - 指数衰减：\(\exp(-\Delta t/\tau_m)\), \(\exp(-\Delta t/\tau_s)\)  
1265 |  59 |   - 正弦基：\(\sin(\omega_r \Delta t),\ \cos(\omega_r \Delta t)\)（对数均匀频率）  
1266 |  60 |   - 分桶 one-hot：\(\text{bucket}(\Delta t)\)（对数间隔）  
1267 |  61 | - 预计算 `PE: Float[T, T, d_pe]`，仅在 \(t' \le t\) 使用。
1268 |  62 | 
1269 |  63 | ### 注意力计算（单头，后接 H 头拼接）
1270 |  64 | 给定 `H̃`（DelayLine 之后）：
1271 |  65 | \[
1272 |  66 | \begin{aligned}
1273 |  67 | q_{i,t}&=W_Q h_{i,t},\\
1274 |  68 | k_{j,t'}&=W_K [h_{j,t'} \,\|\, \phi(t-t')],\\
1275 |  69 | v_{j,t'}&=W_V h_{j,t'}.
1276 |  70 | \end{aligned}
1277 |  71 | \]
1278 |  72 | **门控**：将源端脉冲 \(s_{j,t'}\in\{0,1\}\) 通过 surrogate \(\sigma_{\text{sur}}(\cdot)\) 映射到 \([0,1]\)，作为注意力值的可导缩放因子。  
1279 |  73 | \[
1280 |  74 | e_{i,t,j,t'}=\frac{q_{i,t}\cdot k_{j,t'}}{\sqrt{d/H}} + b_{\Delta t} \quad (\text{可选相对偏置})
1281 |  75 | \]
1282 |  76 | \[
1283 |  77 | a_{i,t,j,t'}=\text{softmax}_{(j,t')\in\mathcal{N}(i),t'\le t}\big(e_{i,t,j,t'}\big)\cdot \sigma_{\text{sur}}(s_{j,t'})
1284 |  78 | \]
1285 |  79 | \[
1286 |  80 | m_{i,t}=\sum_{t'\le t}\sum_{j\in\mathcal{N}(i)} a_{i,t,j,t'} \, v_{j,t'}
1287 |  81 | \]
1288 |  82 | 
1289 |  83 | ### 稀疏化与掩码
1290 |  84 | - **因果掩码**：仅允许 \(t'\le t\)。  
1291 |  85 | - **邻接掩码**：仅允许 `j ∈ N(i)`。  
1292 |  86 | - **时间窗**：限制 \(t-t'\le W\)（建议 `W∈[16,128]` 根据任务/显存选择）。  
1293 |  87 | - **Top-k**：对每个 `(i,t)` 仅保留前 `k` 个 key（在 `(j,t')` 维度上），其余置零并重新归一化（`k=8~32`）。
1294 |  88 | 
1295 |  89 | ### 形状与接口
1296 |  90 | - 入：  
1297 |  91 |   - `H_tilde: Float[T, N, d_in]`  
1298 |  92 |   - `S: Float[T, N]`（0/1）  
1299 |  93 |   - `edge_index: Long[2, E]` 或 `adj_mask: Bool[N,N]`  
1300 |  94 |   - `time_idx: Long[T]`（通常为 `torch.arange(T)`）  
1301 |  95 | - 出：`M: Float[T, N, d]`  
1302 |  96 | - 可选：`dropout_attn`, `attn_temperature`, `relative_bias=True/False`。
1303 |  97 | 
1304 |  98 | ### 复杂度与两种实现模式
1305 |  99 | - **Dense 模式（小图/短序列）**：构建 `[T,T,N,N]` 掩码后矩阵化，配合 Top-k；实现简单，峰值显存较高。  
1306 | 100 | - **Sparse-edge 模式（推荐）**：以 `edge_index` 为骨架，仅对边上的 `(i,j)` 计算注意力；按时间窗为每条边滚动收集 \(t'\in[t-W,t]\) 的 key，使用分块与 `segment_softmax`。复杂度近似 \(O(H\cdot E\cdot W)\)。
1307 | 101 | 
1308 | 102 | ---
1309 | 103 | 
1310 | 104 | ## 4. 脉冲神经元（LIF/GLIF）与集成
1311 | 105 | 
1312 | 106 | ### LIF 更新（可换成你已有实现）
1313 | 107 | \[
1314 | 108 | \begin{aligned}
1315 | 109 | V_{i,t} &= \lambda V_{i,t-1}+ U m_{i,t} - \theta_{i,t-1} R_{i,t-1} \\
1316 | 110 | S_{i,t} &= \mathbb{1}[V_{i,t} > \theta_{i,t}] \\
1317 | 111 | V_{i,t} &\leftarrow V_{i,t} - S_{i,t}\cdot \theta_{i,t} \quad (\text{重置}) \\
1318 | 112 | \theta_{i,t} &= \tau_\theta \theta_{i,t-1} + \gamma S_{i,t-1} \quad (\text{自适应阈值，可选})
1319 | 113 | \end{aligned}
1320 | 114 | \]
1321 | 115 | - **反传**：使用 surrogate \(\sigma_{\text{sur}}'(V-\theta)\)（fast-tanh 或 piecewise-linear STE）。  
1322 | 116 | - **接口**：`LIFCell(M: [T,N,d]) -> S: [T,N], V: [T,N], (theta/R 等可选)`
1323 | 117 | 
1324 | 118 | ---
1325 | 119 | 
1326 | 120 | ## 5. SpikeNet-X 层与前向流程
1327 | 121 | 
1328 | 122 | ### SpikeNet-X 层伪接口
1329 | 123 | ```python
1330 | 124 | class SpikeNetXLayer(nn.Module):
1331 | 125 |     def __init__(self, d_in, d, heads=4, topk=16, W=64, K=5, rho=0.85,
1332 | 126 |                  use_rel_bias=True, attn_drop=0.1, temp=1.0, per_channel=True):
1333 | 127 |         self.delay = LearnableDelayLine(d_in, K, rho, per_channel=per_channel)
1334 | 128 |         self.sta   = SpikingTemporalAttention(d_in, d, heads, topk, W,
1335 | 129 |                                               use_rel_bias, attn_drop, temp)
1336 | 130 |         self.neuron = LIFCell(d, adaptive=True)  # 或接入你现有的 SNN 单元
1337 | 131 |         self.norm = LayerNorm(d)                 # 可选：Pre/LN
1338 | 132 |         self.ffn  = MLP(d, d)                    # 可选：残差前馈
1339 | 133 |     def forward(self, H, S_prev, edge_index, time_idx, adj_mask=None, batch=None):
1340 | 134 |         H̃ = self.delay(H)                                    # [T,N,d_in]
1341 | 135 |         M  = self.sta(H̃, S_prev, edge_index, time_idx, adj_mask)  # [T,N,d]
1342 | 136 |         S, V, aux = self.neuron(M)                            # [T,N], [T,N]
1343 | 137 |         Y = self.norm(M)                                      # 或对 M+FFN 做残差
1344 | 138 |         return S, V, Y, {"M": M, **aux}
1345 | 139 | ```
1346 | 140 | 
1347 | 141 | ### 整体网络（L 层堆叠）
1348 | 142 | - 时间维在外层循环或用并行张量化均可。推荐**张量化时间**（形状 `[T,N,·]`）以便 DelayLine 与 STA 使用缓存的 `PE`。  
1349 | 143 | - 层与层之间传递：`H_{l+1,t} = proj([H_{l,t} || Y_{l,t} || onehot(S_{l,t})])`（可选拼接上一层输出与脉冲 one-hot）。  
1350 | 144 | - 读出：  
1351 | 145 |   - 节点分类：`readout_t` 可择 `t=T` 或 `temporal_attention_pool`（轻量单头）  
1352 | 146 |   - 图级任务：按 batch 聚合（mean/max/attention）
1353 | 147 | 
1354 | 148 | ---
1355 | 149 | 
1356 | 150 | ## 6. 训练配方（默认值可直接用）
1357 | 151 | 
1358 | 152 | - **优化**：AdamW，`lr=2e-3`，`weight_decay=0.01`；线性 warmup 5% 步数；`grad_clip=1.0`。  
1359 | 153 | - **surrogate**：`fast_tanh`: \(\sigma'(x)=\beta(1-\tanh^2(\beta x))\)，`β=2.0`（前 10% epoch 用 `β=1.0` 软化）。  
1360 | 154 | - **正则**：  
1361 | 155 |   - 脉冲率 L1：\(\lambda_\text{spk}\in[1e-5,5e-5]\) 约束平均发放率；  
1362 | 156 |   - 注意力熵惩罚（温和）：\(\lambda_\text{ent}=1e-4\)；  
1363 | 157 |   - 延迟权重 L1/熵：\(\lambda_\text{delay}=1e-4\)。  
1364 | 158 | - **时间窗/稀疏**：`W=64`，`topk=16`（大图任务可改 `W=32, topk=8`）。  
1365 | 159 | - **混合精度**：AMP O2；**梯度检查点**：在 STA 内按 `(time block)` 分段。  
1366 | 160 | - **数据增强（可选）**：时间戳抖动（±1~2 tick），随机时间伸缩（0.9~1.1）。
1367 | 161 | 
1368 | 162 | ---
1369 | 163 | 
1370 | 164 | ## 7. 掩码与数值稳定性（务必实现）
1371 | 165 | 
1372 | 166 | 1) **softmax 掩码**：对被掩蔽位置赋 `-inf`（或非常负的数），再 softmax。  
1373 | 167 | 2) **Top-k**：在 logits 上选 k 大，再将非选中项 logits 置 `-inf`，避免“零后再归一”。  
1374 | 168 | 3) **温度**：`logits /= temp`（`temp∈[0.7,1.4]`），可缓解早期梯度噪声。  
1375 | 169 | 4) **归一**：DelayLine 权重用 `softplus`+`normalize`，数值安全加 `eps=1e-8`。  
1376 | 170 | 5) **空邻居/空窗口**：若 `(i,t)` 无可用 key，返回零向量（或残差直通 `h_{i,t}`）。
1377 | 171 | 
1378 | 172 | ---
1379 | 173 | 
1380 | 174 | ## 8. 复杂度与内存控制
1381 | 175 | 
1382 | 176 | - **理论**：STA 稀疏实现复杂度 \(O(H\cdot E\cdot W)\)，内存近似同量级；  
1383 | 177 | - **工程手段**：  
1384 | 178 |   - 分块时间 `T = sum(T_b)`，逐块缓存 `PE[t−t']`；  
1385 | 179 |   - 将 `edge_index` 排序（`coalesce`）以提升 `segment_softmax` 命中率；  
1386 | 180 |   - 对高入度节点可设**邻居 top-k**上限（先按入度采样邻居，再做时序 top-k）。
1387 | 181 | 
1388 | 182 | ---
1389 | 183 | 
1390 | 184 | ## 9. 相对时间编码实现建议
1391 | 185 | ```python
1392 | 186 | def rel_time_enc(time_idx, d_pe=8, taus=(4,16), n_freq=3):
1393 | 187 |     # time_idx: [T], return PE: [T, T, d_pe] for Δt>=0 else zero
1394 | 188 |     # channels: [exp(-Δt/τ_m), exp(-Δt/τ_s), sin/cos with log-spaced freq, log-bucket onehot]
1395 | 189 |     ...
1396 | 190 | ```
1397 | 191 | - 预计算仅对 \(\Delta t \in [0, W]\) 的子矩阵；其余赋零以节省显存。  
1398 | 192 | - 可选**相对偏置** \(b_{\Delta t}\)（标量表）：长度 `W+1` 的可学习向量。
1399 | 193 | 
1400 | 194 | ---
1401 | 195 | 
1402 | 196 | ## 10. 模块签名与断言（供代码生成器遵循）
1403 | 197 | 
1404 | 198 | ### `LearnableDelayLine`
1405 | 199 | - `__init__(d_in:int, K:int=5, rho:float=0.85, per_channel:bool=True)`.  
1406 | 200 | - `forward(H:[T,N,d_in])->[T,N,d_in]`.  
1407 | 201 | - **断言**：`K>=1`, `0<rho<1`, `H.dim()==3`.
1408 | 202 | 
1409 | 203 | ### `SpikingTemporalAttention`
1410 | 204 | - `__init__(d_in:int, d:int, heads:int=4, topk:int=16, W:int=64, use_rel_bias:bool=True, attn_drop:float=0.1, temp:float=1.0)`.  
1411 | 205 | - `forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E], time_idx:Long[T], adj_mask:Optional[Bool[N,N]]=None) -> [T,N,d]`.  
1412 | 206 | - **断言**：`topk>=1`, `W>=1`, `heads*d_head==d`。  
1413 | 207 | - **稀疏实现关键步骤**：  
1414 | 208 |   1) 对每条边 `(j->i)` 构造过去窗口 `t'∈[t-W,t]` 的键集合；  
1415 | 209 |   2) 计算 `q(i,t)` 与 `k(j,t')`，加上相对编码后做点积；  
1416 | 210 |   3) 在每个 `(i,t)` 的候选集合上做 Top-k，再 masked-softmax；  
1417 | 211 |   4) 加 `attn_drop`，与 `v(j,t')` 加权求和。
1418 | 212 | 
1419 | 213 | ### `LIFCell`
1420 | 214 | - `__init__(d:int, lambda_mem:float=0.95, tau_theta:float=0.99, gamma:float=0.1, adaptive:bool=True, surrogate:str='fast_tanh', beta:float=2.0)`.  
1421 | 215 | - `forward(M:[T,N,d])->Tuple[S:[T,N], V:[T,N], aux:Dict]`.
1422 | 216 | 
1423 | 217 | ---
1424 | 218 | 
1425 | 219 | ## 11. 训练与日志（必须记录的指标）
1426 | 220 | 
1427 | 221 | - 任务指标：Micro/Macro-F1 或 AUC。  
1428 | 222 | - SNN 指标：平均发放率（全局/分层）、失活率（持续 0 发放）、爆发率（>50% 发放）。  
1429 | 223 | - STA 指标：平均注意力熵、Top-k 选择比例、相对时间分布（\(\Delta t\) 直方图）。  
1430 | 224 | - DelayLine 指标：`w_k` 的分布热图；`argmax k` 的频率。  
1431 | 225 | - 资源指标：每 step 时间、峰值显存。  
1432 | 226 | - **可视化**：`r_t` 与注意力重心的时间轨迹，`w_k` 热图，`Δt` 权重柱状图。
1433 | 227 | 
1434 | 228 | ---
1435 | 229 | 
1436 | 230 | ## 12. 消融与开关（实现为 config flags）
1437 | 231 | 
1438 | 232 | - `use_delayline: bool`（False = 仅 STA）  
1439 | 233 | - `use_sta: bool`（False = 回退到原时间池化）  
1440 | 234 | - `topk: int in {0->不裁剪, 8, 16, 32}`  
1441 | 235 | - `W: int`（时间窗）  
1442 | 236 | - `use_rel_bias: bool`（相对偏置）  
1443 | 237 | - `per_channel_delay: bool`  
1444 | 238 | - `surrogate_beta_warmup: bool`（早期软梯度）
1445 | 239 | 
1446 | 240 | ---
1447 | 241 | 
1448 | 242 | ## 13. 失败模式与守护
1449 | 243 | 
1450 | 244 | - **注意力过密/显存爆**：启用/减小 `topk` 与 `W`；`d_head` 降低；开启分块。  
1451 | 245 | - **延迟学成平滑**：对 `w_k` 加熵惩罚或“中心惩罚”鼓励峰化；  
1452 | 246 | - **梯度震荡**：`attn_temperature ↑`、`grad_clip=1.0`、`AdamW β2=0.99`；  
1453 | 247 | - **空邻居**：返回零向量并走残差；  
1454 | 248 | - **发放塌陷**：提高 `λ_spk` 下限、软化 surrogate（小 `β`）、降低阈值上调 `γ`。
1455 | 249 | 
1456 | 250 | ---
1457 | 251 | 
1458 | 252 | ## 14. 参考默认配置（YAML 片段）
1459 | 253 | 
1460 | 254 | ```yaml
1461 | 255 | model:
1462 | 256 |   d_in: 128
1463 | 257 |   d: 256
1464 | 258 |   layers: 3
1465 | 259 |   heads: 4
1466 | 260 |   topk: 16
1467 | 261 |   W: 64
1468 | 262 |   delayline:
1469 | 263 |     use: true
1470 | 264 |     K: 5
1471 | 265 |     rho: 0.85
1472 | 266 |     per_channel: true
1473 | 267 |   sta:
1474 | 268 |     use_rel_bias: true
1475 | 269 |     attn_drop: 0.1
1476 | 270 |     temp: 1.0
1477 | 271 |   lif:
1478 | 272 |     lambda_mem: 0.95
1479 | 273 |     tau_theta: 0.99
1480 | 274 |     gamma: 0.10
1481 | 275 |     surrogate: fast_tanh
1482 | 276 |     beta: 2.0
1483 | 277 | train:
1484 | 278 |   lr: 0.002
1485 | 279 |   weight_decay: 0.01
1486 | 280 |   grad_clip: 1.0
1487 | 281 |   amp: true
1488 | 282 |   seed: 42
1489 | 283 | regularization:
1490 | 284 |   l1_spike: 2.0e-5
1491 | 285 |   attn_entropy: 1.0e-4
1492 | 286 |   delay_reg: 1.0e-4
1493 | 287 | ```
1494 | 288 | 
1495 | 289 | ---
1496 | 290 | 
1497 | 291 | ## 15. 最小工作示例（形状检查伪代码）
1498 | 292 | ```python
1499 | 293 | T, N, d_in, d, H = 64, 1024, 128, 256, 4
1500 | 294 | H0 = torch.randn(T, N, d_in)
1501 | 295 | S0 = torch.zeros(T, N)  # 若首层无前序脉冲，可用全 1 门控或上层脉冲
1502 | 296 | edge_index = ...        # [2,E]
1503 | 297 | time_idx = torch.arange(T)
1504 | 298 | 
1505 | 299 | layer = SpikeNetXLayer(d_in, d, heads=H, topk=16, W=64, K=5, rho=0.85)
1506 | 300 | S, V, Y, aux = layer(H0, S0, edge_index, time_idx)
1507 | 301 | assert S.shape == (T, N) and Y.shape == (T, N, d)
1508 | 302 | ```
1509 | 303 | 
1510 | 304 | ---
1511 | 305 | 
1512 | 306 | ## 16. 写作要点（供注释/文档使用）
1513 | 307 | - **创新点**：将**邻居选择（空间）× 时间对齐（时序）**统一到**事件驱动注意力**，并通过 DelayLine 显式建模**传播时延**；  
1514 | 308 | - **可解释性**：输出 `w_k`、`Δt` 权重与注意力热区；  
1515 | 309 | - **可扩展性**：STA 与 DelayLine 均为**即插即用**，可替换到任意脉冲/非脉冲时序图骨干。
1516 | 310 | 
1517 | 311 | ---
1518 | ```
1519 | 
1520 | ## File: F:\SomeProjects\CSGNN\cline_docs\activeContext.md
1521 | 
1522 | - Extension: .md
1523 | - Language: markdown
1524 | - Size: 2581 bytes
1525 | - Created: 2025-08-22 11:30:37
1526 | - Modified: 2025-08-23 01:10:57
1527 | 
1528 | ### Code
1529 | 
1530 | ```markdown
1531 |  1 | # 当前工作
1532 |  2 | 
1533 |  3 | `SpikeNet-X` 的训练流程已成功调试完毕并得到功能增强。核心的 `RuntimeError` 已被定位并修复，同时增加了模型持久化和评估的关键工程能力，为后续的模型调优和实验奠定了坚实基础。
1534 |  4 | 
1535 |  5 | # 最近的变更
1536 |  6 | 
1537 |  7 | 1.  **修复 `RuntimeError`**:
1538 |  8 |     *   **问题定位**: 使用 `torch.autograd.set_detect_anomaly(True)` 精准定位到 `spikenet_x/sta_sparse.py` 中 `forward` 函数内的 `in-place` 操作是导致梯度计算错误的根源。
1539 |  9 |     *   **解决方案**: 对 `sta_sparse.py` 进行了重构，将循环内对 `max_dst` 张量的 `in-place` 更新，修改为先将各时间步的最大值暂存入一个列表，然后在循环外通过 `torch.stack` 和 `.max()` 操作计算最终结果，彻底消除了 `in-place` 操作，解决了 `RuntimeError`。
1540 | 10 |     *   **验证**: 模型已能在 DBLP 数据集上无错地完成多个周期的训练。
1541 | 11 | 
1542 | 12 | 2.  **增强训练框架**:
1543 | 13 |     *   **模型保存**: 在 `main.py` 中实现了检查点（checkpoint）保存机制。当模型在验证集上取得更优性能时，会自动将模型权重、优化器状态、当前周期数及最佳验证分数保存到指定的 `--checkpoint_dir` 目录中。
1544 | 14 |     *   **断点续训**: 增加了 `--resume_path` 参数，允许从指定的检查点文件恢复训练，无缝衔接之前的训练进度。
1545 | 15 |     *   **独立测试**: 增加了 `--test_model_path` 参数，支持加载一个已保存的模型并仅在测试集上运行评估，方便快速验证模型性能。
1546 | 16 | 
1547 | 17 | # 下一步计划
1548 | 18 | 
1549 | 19 | 随着训练流程的稳定和功能的完善，现在的重点是系统性地进行实验和模型优化。
1550 | 20 | 
1551 | 21 | - **核心任务**:
1552 | 22 |     1.  **完整训练与性能评估**:
1553 | 23 |         *   在 DBLP 数据集上运行一次完整的训练（例如，100个周期），并保存最佳模型。
1554 | 24 |         *   使用独立的测试功能评估最终模型的性能指标（Macro-F1, Micro-F1）。
1555 | 25 |     2.  **超参数调优**:
1556 | 26 |         *   根据基线模型的性能，系统性地调整关键超参数，如学习率 (`--lr`)、时间窗口 (`--W`)、注意力头数 (`--heads`)、Top-K邻居 (`--topk`) 等，以寻找最优配置。
1557 | 27 |     3.  **结果分析与文档记录**:
1558 | 28 |         *   分析不同超参数对模型性能的影响。
1559 | 29 |         *   在 `progress.md` 中记录每次实验的结果和发现。
1560 | 30 | 
1561 | 31 | - **建议**:
1562 | 32 |     *   首先启动一个完整的 DBLP 训练任务，以获得一个基线性能结果。
1563 | 33 |     *   并行地，可以开始设计超参数搜索的实验方案。
1564 | ```
1565 | 
1566 | ## File: F:\SomeProjects\CSGNN\cline_docs\productContext.md
1567 | 
1568 | - Extension: .md
1569 | - Language: markdown
1570 | - Size: 1639 bytes
1571 | - Created: 2025-08-22 11:30:18
1572 | - Modified: 2025-08-22 11:30:21
1573 | 
1574 | ### Code
1575 | 
1576 | ```markdown
1577 |  1 | # 产品背景 (Product Context)
1578 |  2 | 
1579 |  3 | ## 为什么需要这个项目？
1580 |  4 | 该项目旨在解决动态图表示学习中，现有方法在处理大型时间图时，由于通常使用循环神经网络 (RNNs) 而导致的计算和内存开销严重的问题。随着时间图的规模不断扩大，可伸缩性成为一个主要挑战。
1581 |  5 | 
1582 |  6 | ## 解决什么问题？
1583 |  7 | SpikeNet 提出了一种可伸缩的框架，旨在高效地捕获时间图的时序和结构模式，同时显著降低计算成本。它通过使用脉冲神经网络 (SNNs) 替代 RNNs 来解决传统方法在大型动态图上的效率问题，SNNs 作为 RNNs 的低功耗替代方案，能够以高效的方式将图动态建模为神经元群的脉冲序列，并实现基于脉冲的传播。
1584 |  8 | 
1585 |  9 | ## 应该如何工作？
1586 | 10 | SpikeNet 框架通过 SNNs 建模时间图的演化动态。它通过实验证明，在时间节点分类任务上，相比现有基线，SpikeNet 具有更低的计算成本和更优的性能。特别地，它能够以显著更少的参数和计算开销扩展到大型时间图（2M 节点和 13M 边）。该项目还提供了扩展到静态图的示例。
1587 | 11 | 
1588 | 12 | 该项目包含以下主要部分：
1589 | 13 | - **数据处理**：支持 DBLP、Tmall、Patent 等大型时间图数据集，并提供节点特征生成脚本 (`generate_feature.py`)。
1590 | 14 | - **SpikeNet 模型实现**：核心模型逻辑可能位于 `spikenet/layers.py` 和 `spikenet/neuron.py` 等文件中。
1591 | 15 | - **邻居采样器**：通过 `setup.py` 进行构建以实现高效的邻居采样。
1592 | 16 | - **主训练脚本**：`main.py` 用于动态图，`main_static.py` 用于静态图。
1593 | ```
1594 | 
1595 | ## File: F:\SomeProjects\CSGNN\cline_docs\progress.md
1596 | 
1597 | - Extension: .md
1598 | - Language: markdown
1599 | - Size: 1843 bytes
1600 | - Created: 2025-08-22 12:19:46
1601 | - Modified: 2025-08-23 01:12:25
1602 | 
1603 | ### Code
1604 | 
1605 | ```markdown
1606 |  1 | # 项目进度
1607 |  2 | 
1608 |  3 | ## 已完成功能
1609 |  4 | 
1610 |  5 | - **`SpikeNet-X` 原型实现**:
1611 |  6 |     - `spikenet_x` 目录下的所有核心模块已完成。
1612 |  7 |     - 模型可以通过 `spikenet_x/minimal_example.py` 进行验证。
1613 |  8 | - **`main.py` 集成与稀疏 STA**:
1614 |  9 |     - `main.py` 支持通过 `--model spikenetx` 调用模型。
1615 | 10 |     - 实现了 O(E) 复杂度的稀疏 STA 注意力机制，解决了大型图上的内存溢出问题。
1616 | 11 | - **`SpikeNet-X` 训练流程修复**:
1617 | 12 |     - **状态**: **已完成**
1618 | 13 |     - **描述**: 成功定位并修复了稀疏 STA 实现中的 `RuntimeError` (in-place 操作错误)。模型现在可以在大型数据集上稳定运行。
1619 | 14 | - **训练框架功能增强**:
1620 | 15 |     - **状态**: **已完成**
1621 | 16 |     - **描述**: 为 `main.py` 增加了关键的工程能力，包括：
1622 | 17 |         - 基于验证集性能的模型自动保存。
1623 | 18 |         - 从检查点文件恢复训练的断点续训功能。
1624 | 19 |         - 用于快速评估已保存模型的独立测试模式。
1625 | 20 | 
1626 | 21 | ## 需要构建的内容
1627 | 22 | 
1628 | 23 | - **基线模型性能评估**:
1629 | 24 |     - **状态**: **未开始**
1630 | 25 |     - **描述**: 在 DBLP 数据集上运行一次完整的端到端训练，获得基线性能指标（如 Macro/Micro-F1），为后续优化提供参考。
1631 | 26 | 
1632 | 27 | - **超参数调优**:
1633 | 28 |     - **状态**: **未开始**
1634 | 29 |     - **描述**: 在获得基线性能后，系统性地对模型的关键超参数（如学习率, 时间窗口, 注意力头数等）进行调优，以最大化模型性能。
1635 | 30 | 
1636 | 31 | ## 进度状态
1637 | 32 | 
1638 | 33 | - **`SpikeNet-X` 功能**: **已就绪 (READY FOR EXPERIMENTS)**
1639 | 34 | - **原因**: 核心的 `RuntimeError` 已修复，训练流程稳定。同时，模型保存、断点续训等关键功能的加入，使得进行系统性的实验和调优成为可能。项目已从“功能解锁”阶段推进到“实验与优化”阶段。
1640 | ```
1641 | 
1642 | ## File: F:\SomeProjects\CSGNN\cline_docs\systemPatterns.md
1643 | 
1644 | - Extension: .md
1645 | - Language: markdown
1646 | - Size: 4180 bytes
1647 | - Created: 2025-08-22 11:32:15
1648 | - Modified: 2025-08-22 11:32:22
1649 | 
1650 | ### Code
1651 | 
1652 | ```markdown
1653 |  1 | # 系统模式 (System Patterns)
1654 |  2 | 
1655 |  3 | ## 系统如何构建？
1656 |  4 | SpikeNet 项目是一个基于 PyTorch 实现的动态图表示学习框架，核心在于利用脉冲神经网络 (SNNs) 处理时间图数据。其主要组件和构建方式如下：
1657 |  5 | 
1658 |  6 | 1.  **数据层 (`spikenet/dataset.py`)**：
1659 |  7 |     *   提供 `Dataset` 基类，以及 DBLP、Tmall、Patent 等具体数据集的实现。
1660 |  8 |     *   负责从文件中读取节点特征（`.npy`）、边（`.txt` 或 `.json`）和标签（`node2label.txt` 或 `.json`）。
1661 |  9 |     *   支持对节点特征进行标准化。
1662 | 10 |     *   将边列表转换为稀疏邻接矩阵 (`scipy.sparse.csr_matrix`)。
1663 | 11 |     *   实现节点和边的时间切片与划分，以模拟图的动态演化。
1664 | 12 |     *   数据集迭代器允许按时间步访问图快照。
1665 | 13 | 
1666 | 14 | 2.  **核心模型组件 (`spikenet/neuron.py`, `spikenet/layers.py`)**：
1667 | 15 |     *   **神经元模型 (`spikenet/neuron.py`)**：定义了基本的脉冲神经元（如 IF, LIF, PLIF）。这些神经元模型负责电压积分、发放脉冲和重置。
1668 | 16 |     *   **替代梯度 (`spikenet/neuron.py`)**：由于 SNN 的脉冲函数不可导，使用了多种替代梯度技术（如 SuperSpike, MultiGaussSpike, TriangleSpike, ArctanSpike, SigmoidSpike）来实现反向传播训练。
1669 | 17 |     *   **图聚合器 (`spikenet/layers.py`)**：包含了 `SAGEAggregator`，表明网络层可能采用了 GraphSAGE 风格的邻居特征聚合机制。它将中心节点特征与聚合后的邻居特征进行组合。
1670 | 18 | 
1671 | 19 | 3.  **图采样器 (`spikenet/utils.py`, `spikenet/sample_neighber.cpp`)**：
1672 | 20 |     *   `spikenet/utils.py` 中定义了 `Sampler` 和 `RandomWalkSampler` 类，用于从邻接矩阵中采样邻居。
1673 | 21 |     *   `Sampler` 类利用了外部 C++ 实现 `sample_neighber_cpu` 进行高效的邻居采样，这可能是为了性能优化。
1674 | 22 |     *   `RandomWalkSampler` 在可选依赖 `torch_cluster` 存在时提供随机游走采样功能。
1675 | 23 | 
1676 | 24 | 4.  **特征生成 (`generate_feature.py`, `spikenet/deepwalk.py`)**：
1677 | 25 |     *   `generate_feature.py` 脚本用于为不带原始特征的数据集生成节点特征，通过无监督的 DeepWalk 方法实现，其核心逻辑可能在 `spikenet/deepwalk.py` 中。
1678 | 26 | 
1679 | 27 | 5.  **训练入口 (`main.py`, `main_static.py`)**：
1680 | 28 |     *   `main.py` 是用于动态图训练的主脚本，配置数据集、模型参数和训练过程。
1681 | 29 |     *   `main_static.py` 是用于静态图训练的脚本，可能适配了不同的数据集和训练流程。
1682 | 30 | 
1683 | 31 | ## 关键技术决策
1684 | 32 | *   **SNNs 用于动态图**：核心创新是将 SNNs 应用于动态图表示学习，以解决传统 RNNs 在大规模图上的计算和内存效率问题。
1685 | 33 | *   **替代梯度**：采用替代梯度方法来训练 SNNs，使其能够通过反向传播进行优化。
1686 | 34 | *   **GraphSAGE 风格聚合**：使用聚合器从邻居节点收集信息，这是图神经网络中的常见模式。
1687 | 35 | *   **C++ 优化采样**：通过 `sample_neighber.cpp` 提供的 C++ 实现进行邻居采样，以提高性能和处理大规模图的能力。
1688 | 36 | *   **模块化设计**：将神经元模型、网络层、数据处理和采样器等功能分别封装在不同的模块中，提高了代码的可维护性和可扩展性。
1689 | 37 | *   **数据集支持**：设计了通用的 `Dataset` 接口，并为多个真实世界大型时间图数据集提供了具体实现。
1690 | 38 | 
1691 | 39 | ## 架构模式
1692 | 40 | *   **时间序列图处理**：通过迭代时间步来处理图快照，捕获图的动态演化。
1693 | 41 | *   **消息传递范式**：聚合器（如 `SAGEAggregator`）遵循图神经网络的消息传递范式，其中节点通过聚合邻居信息来更新其表示。
1694 | 42 | *   **分离的数据加载与模型逻辑**：`dataset.py` 负责数据管理，而 `neuron.py` 和 `layers.py` 负责模型核心逻辑，实现了关注点分离。
1695 | 43 | *   **参数化神经元行为**：神经元模型（如 LIF）通过可配置的参数（如 `tau`, `v_threshold`, `alpha`）和可选择的替代梯度类型，提供了灵活性。
1696 | 44 | *   **命令行参数配置**：`main.py` 和 `main_static.py` 通过命令行参数 (`argparse`) 配置训练过程，方便实验和调优。
1697 | ```
1698 | 
1699 | ## File: F:\SomeProjects\CSGNN\cline_docs\techContext.md
1700 | 
1701 | - Extension: .md
1702 | - Language: markdown
1703 | - Size: 2097 bytes
1704 | - Created: 2025-08-22 11:33:19
1705 | - Modified: 2025-08-22 11:33:22
1706 | 
1707 | ### Code
1708 | 
1709 | ```markdown
1710 |  1 | # 技术背景 (Tech Context)
1711 |  2 | 
1712 |  3 | ## 使用的技术
1713 |  4 | *   **Python**：主要的编程语言。
1714 |  5 | *   **PyTorch**：核心深度学习框架，用于构建和训练神经网络。
1715 |  6 | *   **NumPy**：用于数值计算和数组操作。
1716 |  7 | *   **SciPy**：用于科学计算，特别是稀疏矩阵操作 (`scipy.sparse`)。
1717 |  8 | *   **Scikit-learn**：用于数据预处理（如 `LabelEncoder`）和模型评估。
1718 |  9 | *   **tqdm**：用于显示进度条。
1719 | 10 | *   **texttable**：用于命令行参数的表格化输出。
1720 | 11 | *   **Numba**：一个 JIT 编译器，可能用于加速某些 Python 代码。
1721 | 12 | *   **C++**：用于高性能的邻居采样模块 (`sample_neighber.cpp`)，通过 `setup.py` 进行编译和集成。
1722 | 13 | *   **torch_cluster (可选)**：如果安装，用于更高级的图采样操作，如随机游走。
1723 | 14 | 
1724 | 15 | ## 开发设置
1725 | 16 | *   **环境**：项目支持在 PyTorch 环境下运行。
1726 | 17 | *   **依赖**：`requirements` 部分列出了具体的包及其版本，包括 `tqdm`, `scipy`, `texttable`, `torch`, `numpy`, `numba`, `scikit_learn` 和可选的 `torch_cluster`。
1727 | 18 | *   **邻居采样器构建**：需要通过运行 `python setup.py install` 来编译和安装 C++ 实现的邻居采样器。
1728 | 19 | *   **数据准备**：数据集需要下载并放置在 `data/` 目录下。对于没有原始节点特征的数据集，可以通过 `generate_feature.py` 脚本使用 DeepWalk 生成特征。
1729 | 20 | 
1730 | 21 | ## 技术约束
1731 | 22 | *   **大规模图处理**：设计目标是处理包含数百万节点和数千万边的大型时间图，对计算和内存效率有较高要求。
1732 | 23 | *   **SNN 训练挑战**：脉冲函数不可导，需要依赖替代梯度方法进行训练。
1733 | 24 | *   **数据格式**：需要适配不同数据集的特定文件格式（`.txt`, `.json`, `.npy`）。
1734 | 25 | *   **PyTorch 版本兼容性**：代码应兼容 PyTorch 1.6-1.12 版本。
1735 | 26 | *   **C++ 依赖**：邻居采样器依赖 C++ 编译，可能需要相应的编译环境。
1736 | 27 | *   **`torch_cluster` 依赖 (可选)**：随机游走采样功能依赖于 `torch_cluster` 库，如果未安装则无法使用该功能。
1737 | ```
1738 | 
1739 | ## File: F:\SomeProjects\CSGNN\spikenet\dataset.py
1740 | 
1741 | - Extension: .py
1742 | - Language: python
1743 | - Size: 11862 bytes
1744 | - Created: 2025-08-21 17:29:04
1745 | - Modified: 2023-09-27 17:42:24
1746 | 
1747 | ### Code
1748 | 
1749 | ```python
1750 |   1 | import math
1751 |   2 | import os.path as osp
1752 |   3 | from collections import defaultdict, namedtuple
1753 |   4 | from typing import Optional
1754 |   5 | 
1755 |   6 | import numpy as np
1756 |   7 | import scipy.sparse as sp
1757 |   8 | import torch
1758 |   9 | from sklearn import preprocessing
1759 |  10 | from sklearn.model_selection import train_test_split
1760 |  11 | from sklearn.preprocessing import LabelEncoder
1761 |  12 | from tqdm import tqdm
1762 |  13 | 
1763 |  14 | Data = namedtuple('Data', ['x', 'edge_index'])
1764 |  15 | 
1765 |  16 | 
1766 |  17 | def standard_normalization(arr):
1767 |  18 |     n_steps, n_node, n_dim = arr.shape
1768 |  19 |     arr_norm = preprocessing.scale(np.reshape(arr, [n_steps, n_node * n_dim]), axis=1)
1769 |  20 |     arr_norm = np.reshape(arr_norm, [n_steps, n_node, n_dim])
1770 |  21 |     return arr_norm
1771 |  22 | 
1772 |  23 | 
1773 |  24 | def edges_to_adj(edges, num_nodes, undirected=True):
1774 |  25 |     row, col = edges
1775 |  26 |     data = np.ones(len(row))
1776 |  27 |     N = num_nodes
1777 |  28 |     adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
1778 |  29 |     if undirected:
1779 |  30 |         adj = adj.maximum(adj.T)
1780 |  31 |     adj[adj > 1] = 1
1781 |  32 |     return adj
1782 |  33 | 
1783 |  34 | 
1784 |  35 | class Dataset:
1785 |  36 |     def __init__(self, name=None, root="./data"):
1786 |  37 |         self.name = name
1787 |  38 |         self.root = root
1788 |  39 |         self.x = None
1789 |  40 |         self.y = None
1790 |  41 |         self.num_features = None
1791 |  42 |         self.adj = []
1792 |  43 |         self.adj_evolve = []
1793 |  44 |         self.edges = []
1794 |  45 |         self.edges_evolve = []
1795 |  46 | 
1796 |  47 |     def _read_feature(self):
1797 |  48 |         filename = osp.join(self.root, self.name, f"{self.name}.npy")
1798 |  49 |         if osp.exists(filename):
1799 |  50 |             return np.load(filename)
1800 |  51 |         else:
1801 |  52 |             return None
1802 |  53 | 
1803 |  54 |     def split_nodes(
1804 |  55 |         self,
1805 |  56 |         train_size: float = 0.4,
1806 |  57 |         val_size: float = 0.0,
1807 |  58 |         test_size: float = 0.6,
1808 |  59 |         random_state: Optional[int] = None,
1809 |  60 |     ):
1810 |  61 |         val_size = 0. if val_size is None else val_size
1811 |  62 |         assert train_size + val_size + test_size <= 1.0
1812 |  63 | 
1813 |  64 |         y = self.y
1814 |  65 |         train_nodes, test_nodes = train_test_split(
1815 |  66 |             torch.arange(y.size(0)),
1816 |  67 |             train_size=train_size + val_size,
1817 |  68 |             test_size=test_size,
1818 |  69 |             random_state=random_state,
1819 |  70 |             stratify=y)
1820 |  71 | 
1821 |  72 |         if val_size:
1822 |  73 |             train_nodes, val_nodes = train_test_split(
1823 |  74 |                 train_nodes,
1824 |  75 |                 train_size=train_size / (train_size + val_size),
1825 |  76 |                 random_state=random_state,
1826 |  77 |                 stratify=y[train_nodes])
1827 |  78 |         else:
1828 |  79 |             val_nodes = None
1829 |  80 | 
1830 |  81 |         self.train_nodes = train_nodes
1831 |  82 |         self.val_nodes = val_nodes
1832 |  83 |         self.test_nodes = test_nodes
1833 |  84 | 
1834 |  85 |     def split_edges(
1835 |  86 |         self,
1836 |  87 |         train_stamp: float = 0.7,
1837 |  88 |         train_size: float = None,
1838 |  89 |         val_size: float = 0.1,
1839 |  90 |         test_size: float = 0.2,
1840 |  91 |         random_state: int = None,
1841 |  92 |     ):
1842 |  93 | 
1843 |  94 |         if random_state is not None:
1844 |  95 |             torch.manual_seed(random_state)
1845 |  96 | 
1846 |  97 |         num_edges = self.edges[-1].size(-1)
1847 |  98 |         train_stamp = train_stamp if train_stamp >= 1 else math.ceil(len(self) * train_stamp)
1848 |  99 | 
1849 | 100 |         train_edges = torch.LongTensor(np.hstack(self.edges_evolve[:train_stamp]))
1850 | 101 |         if train_size is not None:
1851 | 102 |             assert 0 < train_size < 1
1852 | 103 |             num_train = math.floor(train_size * num_edges)
1853 | 104 |             perm = torch.randperm(train_edges.size(1))[:num_train]
1854 | 105 |             train_edges = train_edges[:, perm]
1855 | 106 | 
1856 | 107 |         num_val = math.floor(val_size * num_edges)
1857 | 108 |         num_test = math.floor(test_size * num_edges)
1858 | 109 |         testing_edges = torch.LongTensor(np.hstack(self.edges_evolve[train_stamp:]))
1859 | 110 |         perm = torch.randperm(testing_edges.size(1))
1860 | 111 | 
1861 | 112 |         assert num_val + num_test <= testing_edges.size(1)
1862 | 113 | 
1863 | 114 |         self.train_stamp = train_stamp
1864 | 115 |         self.train_edges = train_edges
1865 | 116 |         self.val_edges = testing_edges[:, perm[:num_val]]
1866 | 117 |         self.test_edges = testing_edges[:, perm[num_val:num_val + num_test]]
1867 | 118 | 
1868 | 119 |     def __getitem__(self, time_index: int):
1869 | 120 |         x = self.x[time_index]
1870 | 121 |         edge_index = self.edges[time_index]
1871 | 122 |         snapshot = Data(x=x, edge_index=edge_index)
1872 | 123 |         return snapshot
1873 | 124 | 
1874 | 125 |     def __next__(self):
1875 | 126 |         if self.t < len(self):
1876 | 127 |             snapshot = self.__getitem__(self.t)
1877 | 128 |             self.t = self.t + 1
1878 | 129 |             return snapshot
1879 | 130 |         else:
1880 | 131 |             self.t = 0
1881 | 132 |             raise StopIteration
1882 | 133 | 
1883 | 134 |     def __iter__(self):
1884 | 135 |         self.t = 0
1885 | 136 |         return self
1886 | 137 | 
1887 | 138 |     def __len__(self):
1888 | 139 |         return len(self.adj)
1889 | 140 | 
1890 | 141 |     def __repr__(self):
1891 | 142 |         return self.name
1892 | 143 | 
1893 | 144 | 
1894 | 145 | class DBLP(Dataset):
1895 | 146 |     def __init__(self, root="./data", normalize=True):
1896 | 147 |         super().__init__(name='dblp', root=root)
1897 | 148 |         edges_evolve, self.num_nodes = self._read_graph()
1898 | 149 |         x = self._read_feature()
1899 | 150 |         y = self._read_label()
1900 | 151 | 
1901 | 152 |         if x is not None:
1902 | 153 |             if normalize:
1903 | 154 |                 x = standard_normalization(x)
1904 | 155 |             self.num_features = x.shape[-1]
1905 | 156 |             self.x = torch.FloatTensor(x)
1906 | 157 | 
1907 | 158 |         self.num_classes = y.max() + 1
1908 | 159 | 
1909 | 160 |         edges = [edges_evolve[0]]
1910 | 161 |         for e_now in edges_evolve[1:]:
1911 | 162 |             e_last = edges[-1]
1912 | 163 |             edges.append(np.hstack([e_last, e_now]))
1913 | 164 | 
1914 | 165 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
1915 | 166 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
1916 | 167 |         self.edges = [torch.LongTensor(edge) for edge in edges]
1917 | 168 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
1918 | 169 | 
1919 | 170 |         self.y = torch.LongTensor(y)
1920 | 171 | 
1921 | 172 |     def _read_graph(self):
1922 | 173 |         filename = osp.join(self.root, self.name, f"{self.name}.txt")
1923 | 174 |         d = defaultdict(list)
1924 | 175 |         N = 0
1925 | 176 |         with open(filename) as f:
1926 | 177 |             for line in f:
1927 | 178 |                 x, y, t = line.strip().split()
1928 | 179 |                 x, y = int(x), int(y)
1929 | 180 |                 d[t].append((x, y))
1930 | 181 |                 N = max(N, x)
1931 | 182 |                 N = max(N, y)
1932 | 183 |         N += 1
1933 | 184 |         edges = []
1934 | 185 |         for time in sorted(d):
1935 | 186 |             row, col = zip(*d[time])
1936 | 187 |             edge_now = np.vstack([row, col])
1937 | 188 |             edges.append(edge_now)
1938 | 189 |         return edges, N
1939 | 190 | 
1940 | 191 |     def _read_label(self):
1941 | 192 |         filename = osp.join(self.root, self.name, "node2label.txt")
1942 | 193 |         nodes = []
1943 | 194 |         labels = []
1944 | 195 |         with open(filename) as f:
1945 | 196 |             for line in f:
1946 | 197 |                 node, label = line.strip().split()
1947 | 198 |                 nodes.append(int(node))
1948 | 199 |                 labels.append(label)
1949 | 200 | 
1950 | 201 |         nodes = np.array(nodes)
1951 | 202 |         labels = LabelEncoder().fit_transform(labels)
1952 | 203 | 
1953 | 204 |         assert np.allclose(nodes, np.arange(nodes.size))
1954 | 205 |         return labels
1955 | 206 | 
1956 | 207 | 
1957 | 208 | def merge(edges, step=1):
1958 | 209 |     if step == 1:
1959 | 210 |         return edges
1960 | 211 |     i = 0
1961 | 212 |     length = len(edges)
1962 | 213 |     out = []
1963 | 214 |     while i < length:
1964 | 215 |         e = edges[i:i + step]
1965 | 216 |         if len(e):
1966 | 217 |             out.append(np.hstack(e))
1967 | 218 |         i += step
1968 | 219 |     print(f'Edges has been merged from {len(edges)} timestamps to {len(out)} timestamps')
1969 | 220 |     return out
1970 | 221 | 
1971 | 222 | 
1972 | 223 | class Tmall(Dataset):
1973 | 224 |     def __init__(self, root="./data", normalize=True):
1974 | 225 |         super().__init__(name='tmall', root=root)
1975 | 226 |         edges_evolve, self.num_nodes = self._read_graph()
1976 | 227 |         x = self._read_feature()
1977 | 228 | 
1978 | 229 |         y, labeled_nodes = self._read_label()
1979 | 230 |         # reindexing
1980 | 231 |         others = set(range(self.num_nodes)) - set(labeled_nodes.tolist())
1981 | 232 |         new_index = np.hstack([labeled_nodes, list(others)])
1982 | 233 |         whole_nodes = np.arange(self.num_nodes)
1983 | 234 |         mapping_dict = dict(zip(new_index, whole_nodes))
1984 | 235 |         mapping = np.vectorize(mapping_dict.get)(whole_nodes)
1985 | 236 |         edges_evolve = [mapping[e] for e in edges_evolve]
1986 | 237 | 
1987 | 238 |         edges_evolve = merge(edges_evolve, step=10)
1988 | 239 | 
1989 | 240 |         if x is not None:
1990 | 241 |             if normalize:
1991 | 242 |                 x = standard_normalization(x)
1992 | 243 |             self.num_features = x.shape[-1]
1993 | 244 |             self.x = torch.FloatTensor(x)
1994 | 245 | 
1995 | 246 |         self.num_classes = y.max() + 1
1996 | 247 | 
1997 | 248 |         edges = [edges_evolve[0]]
1998 | 249 |         for e_now in edges_evolve[1:]:
1999 | 250 |             e_last = edges[-1]
2000 | 251 |             edges.append(np.hstack([e_last, e_now]))
2001 | 252 | 
2002 | 253 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
2003 | 254 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
2004 | 255 |         self.edges = [torch.LongTensor(edge) for edge in edges]
2005 | 256 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
2006 | 257 | 
2007 | 258 |         self.mapping = mapping
2008 | 259 |         self.y = torch.LongTensor(y)
2009 | 260 | 
2010 | 261 |     def _read_graph(self):
2011 | 262 |         filename = osp.join(self.root, self.name, f"{self.name}.txt")
2012 | 263 |         d = defaultdict(list)
2013 | 264 |         N = 0
2014 | 265 |         with open(filename) as f:
2015 | 266 |             for line in tqdm(f, desc='loading edges'):
2016 | 267 |                 x, y, t = line.strip().split()
2017 | 268 |                 x, y = int(x), int(y)
2018 | 269 |                 d[t].append((x, y))
2019 | 270 |                 N = max(N, x)
2020 | 271 |                 N = max(N, y)
2021 | 272 |         N += 1
2022 | 273 |         edges = []
2023 | 274 |         for time in sorted(d):
2024 | 275 |             row, col = zip(*d[time])
2025 | 276 |             edge_now = np.vstack([row, col])
2026 | 277 |             edges.append(edge_now)
2027 | 278 |         return edges, N
2028 | 279 | 
2029 | 280 |     def _read_label(self):
2030 | 281 |         filename = osp.join(self.root, self.name, "node2label.txt")
2031 | 282 |         nodes = []
2032 | 283 |         labels = []
2033 | 284 |         with open(filename) as f:
2034 | 285 |             for line in tqdm(f, desc='loading nodes'):
2035 | 286 |                 node, label = line.strip().split()
2036 | 287 |                 nodes.append(int(node))
2037 | 288 |                 labels.append(label)
2038 | 289 | 
2039 | 290 |         labeled_nodes = np.array(nodes)
2040 | 291 |         labels = LabelEncoder().fit_transform(labels)
2041 | 292 |         return labels, labeled_nodes
2042 | 293 | 
2043 | 294 | 
2044 | 295 | class Patent(Dataset):
2045 | 296 |     def __init__(self, root="./data", normalize=True):
2046 | 297 |         super().__init__(name='patent', root=root)
2047 | 298 |         edges_evolve = self._read_graph()
2048 | 299 |         y = self._read_label()
2049 | 300 |         edges_evolve = merge(edges_evolve, step=2)
2050 | 301 |         x = self._read_feature()
2051 | 302 | 
2052 | 303 |         if x is not None:
2053 | 304 |             if normalize:
2054 | 305 |                 x = standard_normalization(x)
2055 | 306 |             self.num_features = x.shape[-1]
2056 | 307 |             self.x = torch.FloatTensor(x)
2057 | 308 | 
2058 | 309 |         self.num_nodes = y.size
2059 | 310 |         self.num_features = x.shape[-1]
2060 | 311 |         self.num_classes = y.max() + 1
2061 | 312 | 
2062 | 313 |         edges = [edges_evolve[0]]
2063 | 314 |         for e_now in edges_evolve[1:]:
2064 | 315 |             e_last = edges[-1]
2065 | 316 |             edges.append(np.hstack([e_last, e_now]))
2066 | 317 | 
2067 | 318 |         self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
2068 | 319 |         self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
2069 | 320 |         self.edges = [torch.LongTensor(edge) for edge in edges]
2070 | 321 |         self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately
2071 | 322 | 
2072 | 323 |         self.x = torch.FloatTensor(x)
2073 | 324 |         self.y = torch.LongTensor(y)
2074 | 325 | 
2075 | 326 |     def _read_graph(self):
2076 | 327 |         filename = osp.join(self.root, self.name, f"{self.name}_edges.json")
2077 | 328 |         time_edges = defaultdict(list)
2078 | 329 |         with open(filename) as f:
2079 | 330 |             for line in tqdm(f, desc='loading patent_edges'):
2080 | 331 |                 # src nodeID, dst nodeID, date, src originalID, dst originalID
2081 | 332 |                 src, dst, date, _, _ = eval(line)
2082 | 333 |                 date = date // 1e4
2083 | 334 |                 time_edges[date].append((src, dst))
2084 | 335 | 
2085 | 336 |         edges = []
2086 | 337 |         for time in sorted(time_edges):
2087 | 338 |             edges.append(np.transpose(time_edges[time]))
2088 | 339 |         return edges
2089 | 340 | 
2090 | 341 |     def _read_label(self):
2091 | 342 |         filename = osp.join(self.root, self.name, f"{self.name}_nodes.json")
2092 | 343 |         labels = []
2093 | 344 |         with open(filename) as f:
2094 | 345 |             for line in tqdm(f, desc='loading patent_nodes'):
2095 | 346 |                 # nodeID, originalID, date, node class
2096 | 347 |                 node, _, date, label = eval(line)
2097 | 348 |                 date = date // 1e4
2098 | 349 |                 labels.append(label - 1)
2099 | 350 |         labels = np.array(labels)
2100 | 351 |         return labels
2101 | ```
2102 | 
2103 | ## File: F:\SomeProjects\CSGNN\spikenet\deepwalk.py
2104 | 
2105 | - Extension: .py
2106 | - Language: python
2107 | - Size: 5290 bytes
2108 | - Created: 2025-08-21 17:29:04
2109 | - Modified: 2023-09-27 17:42:24
2110 | 
2111 | ### Code
2112 | 
2113 | ```python
2114 |   1 | from distutils.version import LooseVersion
2115 |   2 | 
2116 |   3 | import gensim
2117 |   4 | import numpy as np
2118 |   5 | import scipy.sparse as sp
2119 |   6 | from gensim.models import Word2Vec as _Word2Vec
2120 |   7 | from numba import njit
2121 |   8 | from sklearn import preprocessing
2122 |   9 | 
2123 |  10 | 
2124 |  11 | class DeepWalk:
2125 |  12 |     r"""Implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
2126 |  13 |     from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
2127 |  14 |     The procedure uses random walks to approximate the pointwise mutual information
2128 |  15 |     matrix obtained by pooling normalized adjacency matrix powers. This matrix
2129 |  16 |     is decomposed by an approximate factorization technique.
2130 |  17 |     """
2131 |  18 | 
2132 |  19 |     def __init__(self, dimensions: int = 64,
2133 |  20 |                  walk_length: int = 80,
2134 |  21 |                  walk_number: int = 10,
2135 |  22 |                  workers: int = 3,
2136 |  23 |                  window_size: int = 5,
2137 |  24 |                  epochs: int = 1,
2138 |  25 |                  learning_rate: float = 0.025,
2139 |  26 |                  negative: int = 1,
2140 |  27 |                  name: str = None,
2141 |  28 |                  seed: int = None):
2142 |  29 | 
2143 |  30 |         kwargs = locals()
2144 |  31 |         kwargs.pop("self")
2145 |  32 |         kwargs.pop("__class__", None)
2146 |  33 | 
2147 |  34 |         self.set_hyparas(kwargs)
2148 |  35 | 
2149 |  36 |     def set_hyparas(self, kwargs: dict):
2150 |  37 |         for k, v in kwargs.items():
2151 |  38 |             setattr(self, k, v)
2152 |  39 |         self.hyparas = kwargs
2153 |  40 | 
2154 |  41 |     def fit(self, graph: sp.csr_matrix):
2155 |  42 |         walks = RandomWalker(walk_length=self.walk_length,
2156 |  43 |                              walk_number=self.walk_number).walk(graph)
2157 |  44 |         sentences = [list(map(str, walk)) for walk in walks]
2158 |  45 |         model = Word2Vec(sentences,
2159 |  46 |                          sg=1,
2160 |  47 |                          hs=0,
2161 |  48 |                          alpha=self.learning_rate,
2162 |  49 |                          iter=self.epochs,
2163 |  50 |                          size=self.dimensions,
2164 |  51 |                          window=self.window_size,
2165 |  52 |                          workers=self.workers,
2166 |  53 |                          negative=self.negative,
2167 |  54 |                          seed=self.seed)
2168 |  55 |         self._embedding = model.get_embedding()
2169 |  56 | 
2170 |  57 |     def get_embedding(self, normalize=True) -> np.array:
2171 |  58 |         """Getting the node embedding."""
2172 |  59 |         embedding = self._embedding
2173 |  60 |         if normalize:
2174 |  61 |             embedding = preprocessing.normalize(embedding)
2175 |  62 |         return embedding
2176 |  63 | 
2177 |  64 | 
2178 |  65 | class RandomWalker:
2179 |  66 |     """Fast first-order random walks in DeepWalk
2180 |  67 | 
2181 |  68 |     Parameters:
2182 |  69 |     -----------
2183 |  70 |     walk_number (int): Number of random walks. Default is 10.
2184 |  71 |     walk_length (int): Length of random walks. Default is 80.
2185 |  72 |     """
2186 |  73 | 
2187 |  74 |     def __init__(self, walk_length: int = 80, walk_number: int = 10):
2188 |  75 |         self.walk_length = walk_length
2189 |  76 |         self.walk_number = walk_number
2190 |  77 | 
2191 |  78 |     def walk(self, graph: sp.csr_matrix):
2192 |  79 |         walks = self.random_walk(graph.indices,
2193 |  80 |                                  graph.indptr,
2194 |  81 |                                  walk_length=self.walk_length,
2195 |  82 |                                  walk_number=self.walk_number)
2196 |  83 |         return walks
2197 |  84 | 
2198 |  85 |     @staticmethod
2199 |  86 |     @njit(nogil=True)
2200 |  87 |     def random_walk(indices,
2201 |  88 |                     indptr,
2202 |  89 |                     walk_length,
2203 |  90 |                     walk_number):
2204 |  91 |         N = len(indptr) - 1
2205 |  92 |         for _ in range(walk_number):
2206 |  93 |             for n in range(N):
2207 |  94 |                 walk = [n]
2208 |  95 |                 current_node = n
2209 |  96 |                 for _ in range(walk_length - 1):
2210 |  97 |                     neighbors = indices[
2211 |  98 |                         indptr[current_node]:indptr[current_node + 1]]
2212 |  99 |                     if neighbors.size == 0:
2213 | 100 |                         break
2214 | 101 |                     current_node = np.random.choice(neighbors)
2215 | 102 |                     walk.append(current_node)
2216 | 103 | 
2217 | 104 |                 yield walk
2218 | 105 | 
2219 | 106 | 
2220 | 107 | class Word2Vec(_Word2Vec):
2221 | 108 |     """A compatible version of Word2Vec"""
2222 | 109 | 
2223 | 110 |     def __init__(self, sentences=None, sg=0, hs=0, alpha=0.025, iter=5, size=100, window=5, workers=3, negative=5, seed=None, **kwargs):
2224 | 111 |         if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
2225 | 112 |             super().__init__(sentences,
2226 | 113 |                              size=size,
2227 | 114 |                              window=window,
2228 | 115 |                              min_count=0,
2229 | 116 |                              alpha=alpha,
2230 | 117 |                              sg=sg,
2231 | 118 |                              workers=workers,
2232 | 119 |                              iter=iter,
2233 | 120 |                              negative=negative,
2234 | 121 |                              hs=hs,
2235 | 122 |                              compute_loss=True,
2236 | 123 |                              seed=seed, **kwargs)
2237 | 124 | 
2238 | 125 |         else:
2239 | 126 |             super().__init__(sentences,
2240 | 127 |                              vector_size=size,
2241 | 128 |                              window=window,
2242 | 129 |                              min_count=0,
2243 | 130 |                              alpha=alpha,
2244 | 131 |                              sg=sg,
2245 | 132 |                              workers=workers,
2246 | 133 |                              epochs=iter,
2247 | 134 |                              negative=negative,
2248 | 135 |                              hs=hs,
2249 | 136 |                              compute_loss=True,
2250 | 137 |                              seed=seed, **kwargs)
2251 | 138 | 
2252 | 139 |     def get_embedding(self):
2253 | 140 |         if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
2254 | 141 |             embedding = self.wv.vectors[np.fromiter(
2255 | 142 |                 map(int, self.wv.index2word), np.int32).argsort()]
2256 | 143 |         else:
2257 | 144 |             embedding = self.wv.vectors[np.fromiter(
2258 | 145 |                 map(int, self.wv.index_to_key), np.int32).argsort()]
2259 | 146 | 
2260 | 147 |         return embedding
2261 | ```
2262 | 
2263 | ## File: F:\SomeProjects\CSGNN\spikenet\layers.py
2264 | 
2265 | - Extension: .py
2266 | - Language: python
2267 | - Size: 1225 bytes
2268 | - Created: 2025-08-21 17:29:04
2269 | - Modified: 2023-09-27 17:42:24
2270 | 
2271 | ### Code
2272 | 
2273 | ```python
2274 |  1 | import torch
2275 |  2 | import torch.nn as nn
2276 |  3 | 
2277 |  4 | 
2278 |  5 | class SAGEAggregator(nn.Module):
2279 |  6 |     def __init__(self, in_features, out_features,
2280 |  7 |                  aggr='mean',
2281 |  8 |                  concat=False,
2282 |  9 |                  bias=False):
2283 | 10 | 
2284 | 11 |         super().__init__()
2285 | 12 |         self.in_features = in_features
2286 | 13 |         self.out_features = out_features
2287 | 14 |         self.concat = concat
2288 | 15 | 
2289 | 16 |         self.aggr = aggr
2290 | 17 |         self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]
2291 | 18 | 
2292 | 19 |         self.lin_l = nn.Linear(in_features, out_features, bias=bias)
2293 | 20 |         self.lin_r = nn.Linear(in_features, out_features, bias=bias)
2294 | 21 | 
2295 | 22 |     def forward(self, x, neigh_x):
2296 | 23 |         if not isinstance(x, torch.Tensor):
2297 | 24 |             x = torch.cat(x, dim=0)
2298 | 25 | 
2299 | 26 |         if not isinstance(neigh_x, torch.Tensor):
2300 | 27 |             neigh_x = torch.cat([self.aggregator(h, dim=1)
2301 | 28 |                                 for h in neigh_x], dim=0)
2302 | 29 |         else:
2303 | 30 |             neigh_x = self.aggregator(neigh_x, dim=1)
2304 | 31 | 
2305 | 32 |         x = self.lin_l(x)
2306 | 33 |         neigh_x = self.lin_r(neigh_x)
2307 | 34 |         out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
2308 | 35 |         return out
2309 | 36 | 
2310 | 37 |     def __repr__(self):
2311 | 38 |         return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"
2312 | ```
2313 | 
2314 | ## File: F:\SomeProjects\CSGNN\spikenet\neuron.py
2315 | 
2316 | - Extension: .py
2317 | - Language: python
2318 | - Size: 7039 bytes
2319 | - Created: 2025-08-21 17:29:04
2320 | - Modified: 2023-09-27 17:42:24
2321 | 
2322 | ### Code
2323 | 
2324 | ```python
2325 |   1 | from math import pi
2326 |   2 | 
2327 |   3 | import torch
2328 |   4 | import torch.nn as nn
2329 |   5 | 
2330 |   6 | gamma = 0.2
2331 |   7 | thresh_decay = 0.7
2332 |   8 | 
2333 |   9 | 
2334 |  10 | def reset_net(net: nn.Module):
2335 |  11 |     for m in net.modules():
2336 |  12 |         if hasattr(m, 'reset'):
2337 |  13 |             m.reset()
2338 |  14 | 
2339 |  15 | 
2340 |  16 | def heaviside(x: torch.Tensor):
2341 |  17 |     return x.ge(0)
2342 |  18 | 
2343 |  19 | 
2344 |  20 | def gaussian(x, mu, sigma):
2345 |  21 |     """
2346 |  22 |     Gaussian PDF with broadcasting.
2347 |  23 |     """
2348 |  24 |     return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))
2349 |  25 | 
2350 |  26 | 
2351 |  27 | class BaseSpike(torch.autograd.Function):
2352 |  28 |     """
2353 |  29 |     Baseline spiking function.
2354 |  30 |     """
2355 |  31 | 
2356 |  32 |     @staticmethod
2357 |  33 |     def forward(ctx, x, alpha):
2358 |  34 |         ctx.save_for_backward(x, alpha)
2359 |  35 |         return x.gt(0).float()
2360 |  36 | 
2361 |  37 |     @staticmethod
2362 |  38 |     def backward(ctx, grad_output):
2363 |  39 |         raise NotImplementedError
2364 |  40 | 
2365 |  41 | 
2366 |  42 | class SuperSpike(BaseSpike):
2367 |  43 |     """
2368 |  44 |     Spike function with SuperSpike surrogate gradient from
2369 |  45 |     "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.
2370 |  46 | 
2371 |  47 |     Design choices:
2372 |  48 |     - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
2373 |  49 |     - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
2374 |  50 |     """
2375 |  51 | 
2376 |  52 |     @staticmethod
2377 |  53 |     def backward(ctx, grad_output):
2378 |  54 |         x, alpha = ctx.saved_tensors
2379 |  55 |         grad_input = grad_output.clone()
2380 |  56 |         sg = 1 / (1 + alpha * x.abs()) ** 2
2381 |  57 |         return grad_input * sg, None
2382 |  58 | 
2383 |  59 | 
2384 |  60 | class MultiGaussSpike(BaseSpike):
2385 |  61 |     """
2386 |  62 |     Spike function with multi-Gaussian surrogate gradient from
2387 |  63 |     "Accurate and efficient time-domain classification...", Yin et al. 2021.
2388 |  64 | 
2389 |  65 |     Design choices:
2390 |  66 |     - Hyperparameters determined through grid search (Yin et al. 2021)
2391 |  67 |     """
2392 |  68 | 
2393 |  69 |     @staticmethod
2394 |  70 |     def backward(ctx, grad_output):
2395 |  71 |         x, alpha = ctx.saved_tensors
2396 |  72 |         grad_input = grad_output.clone()
2397 |  73 |         zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
2398 |  74 |         sg = (
2399 |  75 |             1.15 * gaussian(x, zero, alpha)
2400 |  76 |             - 0.15 * gaussian(x, alpha, 6 * alpha)
2401 |  77 |             - 0.15 * gaussian(x, -alpha, 6 * alpha)
2402 |  78 |         )
2403 |  79 |         return grad_input * sg, None
2404 |  80 | 
2405 |  81 | 
2406 |  82 | class TriangleSpike(BaseSpike):
2407 |  83 |     """
2408 |  84 |     Spike function with triangular surrogate gradient
2409 |  85 |     as in Bellec et al. 2020.
2410 |  86 |     """
2411 |  87 | 
2412 |  88 |     @staticmethod
2413 |  89 |     def backward(ctx, grad_output):
2414 |  90 |         x, alpha = ctx.saved_tensors
2415 |  91 |         grad_input = grad_output.clone()
2416 |  92 |         sg = torch.nn.functional.relu(1 - alpha * x.abs())
2417 |  93 |         return grad_input * sg, None
2418 |  94 | 
2419 |  95 | 
2420 |  96 | class ArctanSpike(BaseSpike):
2421 |  97 |     """
2422 |  98 |     Spike function with derivative of arctan surrogate gradient.
2423 |  99 |     Featured in Fang et al. 2020/2021.
2424 | 100 |     """
2425 | 101 | 
2426 | 102 |     @staticmethod
2427 | 103 |     def backward(ctx, grad_output):
2428 | 104 |         x, alpha = ctx.saved_tensors
2429 | 105 |         grad_input = grad_output.clone()
2430 | 106 |         sg = 1 / (1 + alpha * x * x)
2431 | 107 |         return grad_input * sg, None
2432 | 108 | 
2433 | 109 | 
2434 | 110 | class SigmoidSpike(BaseSpike):
2435 | 111 | 
2436 | 112 |     @staticmethod
2437 | 113 |     def backward(ctx, grad_output):
2438 | 114 |         x, alpha = ctx.saved_tensors
2439 | 115 |         grad_input = grad_output.clone()
2440 | 116 |         sgax = (x * alpha).sigmoid_()
2441 | 117 |         sg = (1. - sgax) * sgax * alpha
2442 | 118 |         return grad_input * sg, None
2443 | 119 | 
2444 | 120 | 
2445 | 121 | def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
2446 | 122 |     return SuperSpike.apply(x - thresh, alpha)
2447 | 123 | 
2448 | 124 | 
2449 | 125 | def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
2450 | 126 |     return MultiGaussSpike.apply(x - thresh, alpha)
2451 | 127 | 
2452 | 128 | 
2453 | 129 | def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
2454 | 130 |     return SigmoidSpike.apply(x - thresh, alpha)
2455 | 131 | 
2456 | 132 | 
2457 | 133 | def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
2458 | 134 |     return TriangleSpike.apply(x - thresh, alpha)
2459 | 135 | 
2460 | 136 | 
2461 | 137 | def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
2462 | 138 |     return ArctanSpike.apply(x - thresh, alpha)
2463 | 139 | 
2464 | 140 | 
2465 | 141 | SURROGATE = {'sigmoid': sigmoidspike, 'triangle': trianglespike, 'arctan': arctanspike,
2466 | 142 |              'mg': mgspike, 'super': superspike}
2467 | 143 | 
2468 | 144 | 
2469 | 145 | class IF(nn.Module):
2470 | 146 |     def __init__(self, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
2471 | 147 |         super().__init__()
2472 | 148 |         self.v_threshold = v_threshold
2473 | 149 |         self.v_reset = v_reset
2474 | 150 |         self.surrogate = SURROGATE.get(surrogate)
2475 | 151 |         self.register_buffer("alpha", torch.as_tensor(
2476 | 152 |             alpha, dtype=torch.float32))
2477 | 153 |         self.reset()
2478 | 154 | 
2479 | 155 |     def reset(self):
2480 | 156 |         self.v = 0.
2481 | 157 |         self.v_th = self.v_threshold
2482 | 158 | 
2483 | 159 |     def forward(self, dv):
2484 | 160 |         # 1. charge
2485 | 161 |         self.v += dv
2486 | 162 |         # 2. fire
2487 | 163 |         spike = self.surrogate(self.v, self.v_threshold, self.alpha)
2488 | 164 |         # 3. reset
2489 | 165 |         self.v = (1 - spike) * self.v + spike * self.v_reset
2490 | 166 |         # 4. threhold updates
2491 | 167 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
2492 | 168 |         self.v_th = gamma * spike + self.v_th * thresh_decay
2493 | 169 |         return spike
2494 | 170 | 
2495 | 171 | 
2496 | 172 | class LIF(nn.Module):
2497 | 173 |     def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
2498 | 174 |         super().__init__()
2499 | 175 |         self.v_threshold = v_threshold
2500 | 176 |         self.v_reset = v_reset
2501 | 177 |         self.surrogate = SURROGATE.get(surrogate)
2502 | 178 |         self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
2503 | 179 |         self.register_buffer("alpha", torch.as_tensor(
2504 | 180 |             alpha, dtype=torch.float32))
2505 | 181 |         self.reset()
2506 | 182 | 
2507 | 183 |     def reset(self):
2508 | 184 |         self.v = 0.
2509 | 185 |         self.v_th = self.v_threshold
2510 | 186 | 
2511 | 187 |     def forward(self, dv):
2512 | 188 |         # 1. charge
2513 | 189 |         self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
2514 | 190 |         # 2. fire
2515 | 191 |         spike = self.surrogate(self.v, self.v_th, self.alpha)
2516 | 192 |         # 3. reset
2517 | 193 |         self.v = (1 - spike) * self.v + spike * self.v_reset
2518 | 194 |         # 4. threhold updates
2519 | 195 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
2520 | 196 |         self.v_th = gamma * spike + self.v_th * thresh_decay
2521 | 197 |         return spike
2522 | 198 | 
2523 | 199 | 
2524 | 200 | class PLIF(nn.Module):
2525 | 201 |     def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
2526 | 202 |         super().__init__()
2527 | 203 |         self.v_threshold = v_threshold
2528 | 204 |         self.v_reset = v_reset
2529 | 205 |         self.surrogate = SURROGATE.get(surrogate)
2530 | 206 |         self.register_parameter("tau", nn.Parameter(
2531 | 207 |             torch.as_tensor(tau, dtype=torch.float32)))
2532 | 208 |         self.register_buffer("alpha", torch.as_tensor(
2533 | 209 |             alpha, dtype=torch.float32))
2534 | 210 |         self.reset()
2535 | 211 | 
2536 | 212 |     def reset(self):
2537 | 213 |         self.v = 0.
2538 | 214 |         self.v_th = self.v_threshold
2539 | 215 | 
2540 | 216 |     def forward(self, dv):
2541 | 217 |         # 1. charge
2542 | 218 |         self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
2543 | 219 |         # 2. fire
2544 | 220 |         spike = self.surrogate(self.v, self.v_th, self.alpha)
2545 | 221 |         # 3. reset
2546 | 222 |         self.v = (1 - spike) * self.v + spike * self.v_reset
2547 | 223 |         # 4. threhold updates
2548 | 224 |         # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
2549 | 225 |         self.v_th = gamma * spike + self.v_th * thresh_decay
2550 | 226 |         return spike
2551 | ```
2552 | 
2553 | ## File: F:\SomeProjects\CSGNN\spikenet\sample_neighber.cpp
2554 | 
2555 | - Extension: .cpp
2556 | - Language: cpp
2557 | - Size: 5664 bytes
2558 | - Created: 2025-08-21 17:29:04
2559 | - Modified: 2023-09-27 17:42:24
2560 | 
2561 | ### Code
2562 | 
2563 | ```cpp
2564 |   1 | #include <torch/extension.h>
2565 |   2 | #define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
2566 |   3 | #define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
2567 |   4 | 
2568 |   5 | #define AT_DISPATCH_HAS_VALUE(optional_value, ...) \
2569 |   6 |     [&] {                                          \
2570 |   7 |         if (optional_value.has_value())            \
2571 |   8 |         {                                          \
2572 |   9 |             const bool HAS_VALUE = true;           \
2573 |  10 |             return __VA_ARGS__();                  \
2574 |  11 |         }                                          \
2575 |  12 |         else                                       \
2576 |  13 |         {                                          \
2577 |  14 |             const bool HAS_VALUE = false;          \
2578 |  15 |             return __VA_ARGS__();                  \
2579 |  16 |         }                                          \
2580 |  17 |     }()
2581 |  18 | 
2582 |  19 | torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
2583 |  20 |                int64_t num_neighbors, bool replace);
2584 |  21 | 
2585 |  22 | // Returns `rowptr`, `col`, `n_id`, `e_id`
2586 |  23 | torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
2587 |  24 |                int64_t num_neighbors, bool replace)
2588 |  25 | {
2589 |  26 |     CHECK_CPU(rowptr);
2590 |  27 |     CHECK_CPU(col);
2591 |  28 |     CHECK_CPU(idx);
2592 |  29 |     CHECK_INPUT(idx.dim() == 1);
2593 |  30 | 
2594 |  31 |     auto rowptr_data = rowptr.data_ptr<int64_t>();
2595 |  32 |     auto col_data = col.data_ptr<int64_t>();
2596 |  33 |     auto idx_data = idx.data_ptr<int64_t>();
2597 |  34 | 
2598 |  35 |     std::vector<int64_t> n_ids;
2599 |  36 | 
2600 |  37 |     int64_t i;
2601 |  38 |     
2602 |  39 | 
2603 |  40 |     int64_t n, c, e, row_start, row_end, row_count;
2604 |  41 | 
2605 |  42 |     if (num_neighbors < 0)
2606 |  43 |     { // No sampling ======================================
2607 |  44 | 
2608 |  45 |         for (int64_t i = 0; i < idx.numel(); i++)
2609 |  46 |         {
2610 |  47 |             n = idx_data[i];
2611 |  48 |             row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
2612 |  49 |             row_count = row_end - row_start;
2613 |  50 | 
2614 |  51 |             for (int64_t j = 0; j < row_count; j++)
2615 |  52 |             {
2616 |  53 |                 e = row_start + j;
2617 |  54 |                 c = col_data[e];
2618 |  55 |                 n_ids.push_back(c);
2619 |  56 |             }
2620 |  57 |         }
2621 |  58 |     }
2622 |  59 | 
2623 |  60 |     else if (replace)
2624 |  61 |     { // Sample with replacement ===============================
2625 |  62 |         for (int64_t i = 0; i < idx.numel(); i++)
2626 |  63 |         {
2627 |  64 |             n = idx_data[i];
2628 |  65 |             row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
2629 |  66 |             row_count = row_end - row_start;
2630 |  67 | 
2631 |  68 |             std::unordered_set<int64_t> perm;
2632 |  69 |             if (row_count <= num_neighbors)
2633 |  70 |             {
2634 |  71 |                 for (int64_t j = 0; j < row_count; j++)
2635 |  72 |                     perm.insert(j);
2636 |  73 |                 for (int64_t j = 0; j < num_neighbors-row_count; j++){
2637 |  74 |                     e = row_start + rand() % row_count;
2638 |  75 |                     c = col_data[e];
2639 |  76 |                     n_ids.push_back(c);
2640 |  77 |                 }
2641 |  78 |             }
2642 |  79 |             else
2643 |  80 |             { // See: https://www.nowherenearithaca.com/2013/05/
2644 |  81 |                 //      robert-floyds-tiny-and-beautiful.html
2645 |  82 |                 for (int64_t j = row_count - num_neighbors; j < row_count; j++)
2646 |  83 |                 {
2647 |  84 |                     if (!perm.insert(rand() % j).second)
2648 |  85 |                         perm.insert(j);
2649 |  86 |                 }
2650 |  87 |             }
2651 |  88 | 
2652 |  89 |             
2653 |  90 |             for (const int64_t &p : perm)
2654 |  91 |             {
2655 |  92 |                 e = row_start + p;
2656 |  93 |                 c = col_data[e];
2657 |  94 |                 n_ids.push_back(c);
2658 |  95 |             }
2659 |  96 |             
2660 |  97 |         }
2661 |  98 |         // for (int64_t i = 0; i < idx.numel(); i++)
2662 |  99 |         // {
2663 | 100 |         //     n = idx_data[i];
2664 | 101 |         //     row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
2665 | 102 |         //     row_count = row_end - row_start;
2666 | 103 |         //     // std::vector<int64_t>().swap(temp);
2667 | 104 |         //     // for (int64_t j = 0; j < row_count; j++)
2668 | 105 |         //     // {
2669 | 106 |         //     //     temp.push_back(j);
2670 | 107 |         //     // }
2671 | 108 |         //     // if (row_count<num_neighbors){
2672 | 109 |         //     //     for (int64_t j = 0; j <num_neighbors-row_count; j++){
2673 | 110 |         //     //         temp.push_back(rand() % row_count);
2674 | 111 |         //     //     }
2675 | 112 |         //     // }
2676 | 113 |         //     // std::random_shuffle(temp.begin(), temp.end());
2677 | 114 |         //     std::unordered_set<int64_t> perm;
2678 | 115 |         //     for (int64_t j = 0; j < num_neighbors; j++)
2679 | 116 |         //     {
2680 | 117 |         //         e = row_start + rand() % row_count;
2681 | 118 |         //         // e = row_start + temp[j];
2682 | 119 |         //         c = col_data[e];
2683 | 120 |         //         n_ids.push_back(c);
2684 | 121 |         //     }
2685 | 122 |         // }
2686 | 123 |     }
2687 | 124 |     else
2688 | 125 |     { // Sample without replacement via Robert Floyd algorithm ============
2689 | 126 | 
2690 | 127 |         for (int64_t i = 0; i < idx.numel(); i++)
2691 | 128 |         {
2692 | 129 |             n = idx_data[i];
2693 | 130 |             row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
2694 | 131 |             row_count = row_end - row_start;
2695 | 132 | 
2696 | 133 |             std::unordered_set<int64_t> perm;
2697 | 134 |             if (row_count <= num_neighbors)
2698 | 135 |             {
2699 | 136 |                 for (int64_t j = 0; j < row_count; j++)
2700 | 137 |                     perm.insert(j);
2701 | 138 |             }
2702 | 139 |             else
2703 | 140 |             { // See: https://www.nowherenearithaca.com/2013/05/
2704 | 141 |                 //      robert-floyds-tiny-and-beautiful.html
2705 | 142 |                 for (int64_t j = row_count - num_neighbors; j < row_count; j++)
2706 | 143 |                 {
2707 | 144 |                     if (!perm.insert(rand() % j).second)
2708 | 145 |                         perm.insert(j);
2709 | 146 |                 }
2710 | 147 |             }
2711 | 148 | 
2712 | 149 |             for (const int64_t &p : perm)
2713 | 150 |             {
2714 | 151 |                 e = row_start + p;
2715 | 152 |                 c = col_data[e];
2716 | 153 |                 n_ids.push_back(c);
2717 | 154 |             }
2718 | 155 |         }
2719 | 156 |     }
2720 | 157 | 
2721 | 158 |     int64_t N = n_ids.size();
2722 | 159 |     auto out_n_id = torch::from_blob(n_ids.data(), {N}, col.options()).clone();
2723 | 160 | 
2724 | 161 |     return out_n_id;
2725 | 162 | }
2726 | 163 | PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
2727 | 164 |     m.def("sample_neighber_cpu", &sample_neighber_cpu, "Node neighborhood sampler");
2728 | 165 | }
2729 | ```
2730 | 
2731 | ## File: F:\SomeProjects\CSGNN\spikenet\utils.py
2732 | 
2733 | - Extension: .py
2734 | - Language: python
2735 | - Size: 2759 bytes
2736 | - Created: 2025-08-21 17:29:04
2737 | - Modified: 2023-09-27 17:42:24
2738 | 
2739 | ### Code
2740 | 
2741 | ```python
2742 |  1 | import numpy as np
2743 |  2 | import scipy.sparse as sp
2744 |  3 | import torch
2745 |  4 | from sample_neighber import sample_neighber_cpu
2746 |  5 | from texttable import Texttable
2747 |  6 | 
2748 |  7 | try:
2749 |  8 |     import torch_cluster
2750 |  9 | except ImportError:
2751 | 10 |     torch_cluster = None
2752 | 11 | 
2753 | 12 | 
2754 | 13 | def set_seed(seed):
2755 | 14 |     np.random.seed(seed)
2756 | 15 |     torch.manual_seed(seed)
2757 | 16 |     torch.cuda.manual_seed(seed)
2758 | 17 | 
2759 | 18 | def tab_printer(args):
2760 | 19 |     """Function to print the logs in a nice tabular format.
2761 | 20 |     
2762 | 21 |     Note
2763 | 22 |     ----
2764 | 23 |     Package `Texttable` is required.
2765 | 24 |     Run `pip install Texttable` if was not installed.
2766 | 25 |     
2767 | 26 |     Parameters
2768 | 27 |     ----------
2769 | 28 |     args: Parameters used for the model.
2770 | 29 |     """
2771 | 30 |     args = vars(args)
2772 | 31 |     keys = sorted(args.keys())
2773 | 32 |     t = Texttable() 
2774 | 33 |     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
2775 | 34 |     print(t.draw())
2776 | 35 | 
2777 | 36 |     
2778 | 37 | class Sampler:
2779 | 38 |     def __init__(self, adj_matrix: sp.csr_matrix):
2780 | 39 |         self.rowptr = torch.LongTensor(adj_matrix.indptr)
2781 | 40 |         self.col = torch.LongTensor(adj_matrix.indices)
2782 | 41 | 
2783 | 42 |     def __call__(self, nodes, size, replace=True):
2784 | 43 |         nbr = sample_neighber_cpu(self.rowptr, self.col, nodes, size, replace)
2785 | 44 |         return nbr
2786 | 45 |     
2787 | 46 |     
2788 | 47 | class RandomWalkSampler:
2789 | 48 |     def __init__(self, adj_matrix: sp.csr_matrix, p: float = 1.0, q: float = 1.0):
2790 | 49 |         self.rowptr = torch.LongTensor(adj_matrix.indptr)
2791 | 50 |         self.col = torch.LongTensor(adj_matrix.indices)
2792 | 51 |         self.p = p
2793 | 52 |         self.q = q
2794 | 53 |         assert torch_cluster, "Please install 'torch_cluster' first."
2795 | 54 | 
2796 | 55 |     def __call__(self, nodes, size, replace=True):
2797 | 56 |         nbr = torch.ops.torch_cluster.random_walk(self.rowptr, self.col, nodes, size, self.p, self.q)[0][:, 1:] 
2798 | 57 |         return nbr
2799 | 58 | 
2800 | 59 | 
2801 | 60 | def eliminate_selfloops(adj_matrix):
2802 | 61 |     """eliminate selfloops for adjacency matrix.
2803 | 62 | 
2804 | 63 |     >>>eliminate_selfloops(adj) # return an adjacency matrix without selfloops
2805 | 64 | 
2806 | 65 |     Parameters
2807 | 66 |     ----------
2808 | 67 |     adj_matrix: Scipy matrix or Numpy array
2809 | 68 | 
2810 | 69 |     Returns
2811 | 70 |     -------
2812 | 71 |     Single Scipy sparse matrix or Numpy matrix.
2813 | 72 | 
2814 | 73 |     """
2815 | 74 |     if sp.issparse(adj_matrix):
2816 | 75 |         adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
2817 | 76 |         adj_matrix.eliminate_zeros()
2818 | 77 |     else:
2819 | 78 |         adj_matrix = adj_matrix - np.diag(adj_matrix)
2820 | 79 |     return adj_matrix
2821 | 80 | 
2822 | 81 | 
2823 | 82 | def add_selfloops(adj_matrix: sp.csr_matrix):
2824 | 83 |     """add selfloops for adjacency matrix.
2825 | 84 | 
2826 | 85 |     >>>add_selfloops(adj) # return an adjacency matrix with selfloops
2827 | 86 | 
2828 | 87 |     Parameters
2829 | 88 |     ----------
2830 | 89 |     adj_matrix: Scipy matrix or Numpy array
2831 | 90 | 
2832 | 91 |     Returns
2833 | 92 |     -------
2834 | 93 |     Single sparse matrix or Numpy matrix.
2835 | 94 | 
2836 | 95 |     """
2837 | 96 |     adj_matrix = eliminate_selfloops(adj_matrix)
2838 | 97 | 
2839 | 98 |     return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')
2840 | ```
2841 | 
2842 | ## File: F:\SomeProjects\CSGNN\spikenet_x\delayline.py
2843 | 
2844 | - Extension: .py
2845 | - Language: python
2846 | - Size: 4626 bytes
2847 | - Created: 2025-08-22 12:55:40
2848 | - Modified: 2025-08-22 12:56:05
2849 | 
2850 | ### Code
2851 | 
2852 | ```python
2853 |   1 | # -*- coding: utf-8 -*-
2854 |   2 | """
2855 |   3 | LearnableDelayLine: 可学习多延迟通路（因果深度可分离 1D 卷积）
2856 |   4 | 
2857 |   5 | - 输入输出形状:
2858 |   6 |   forward(H: Float[T, N, d_in]) -> Float[T, N, d_in]
2859 |   7 | 
2860 |   8 | - 设计:
2861 |   9 |   对每个通道 c, 时间步 t:
2862 |  10 |       H_tilde[t, :, c] = sum_{k=0..K-1} w_c[k] * H[t-k, :, c]
2863 |  11 |   其中 w_c[k] = softplus(u_c[k]) * rho^k / sum_r softplus(u_c[r]) * rho^r
2864 |  12 |   并在 t-k < 0 使用因果左填充 0。
2865 |  13 | 
2866 |  14 | - 实现:
2867 |  15 |   将输入重排为 [N, d_in, T]，使用 groups=d_in 的 Conv1d 执行深度可分离因果卷积。
2868 |  16 | """
2869 |  17 | 
2870 |  18 | from __future__ import annotations
2871 |  19 | 
2872 |  20 | from typing import Optional
2873 |  21 | 
2874 |  22 | import torch
2875 |  23 | import torch.nn as nn
2876 |  24 | import torch.nn.functional as F
2877 |  25 | 
2878 |  26 | 
2879 |  27 | class LearnableDelayLine(nn.Module):
2880 |  28 |     def __init__(
2881 |  29 |         self,
2882 |  30 |         d_in: int,
2883 |  31 |         K: int = 5,
2884 |  32 |         rho: float = 0.85,
2885 |  33 |         per_channel: bool = True,
2886 |  34 |         eps: float = 1e-8,
2887 |  35 |     ) -> None:
2888 |  36 |         """
2889 |  37 |         参数
2890 |  38 |         ----
2891 |  39 |         d_in : 输入通道数（特征维）
2892 |  40 |         K : 延迟 tap 数 (>=1)
2893 |  41 |         rho : 指数折扣因子，(0,1)
2894 |  42 |         per_channel : True 表示逐通道独立权重；False 表示所有通道共享一组权重
2895 |  43 |         eps : 归一化时的数值稳定项
2896 |  44 |         """
2897 |  45 |         super().__init__()
2898 |  46 |         assert d_in >= 1, "d_in 必须 >= 1"
2899 |  47 |         assert K >= 1, "K 必须 >= 1"
2900 |  48 |         assert 0.0 < rho < 1.0, "rho 必须在 (0,1)"
2901 |  49 | 
2902 |  50 |         self.d_in = int(d_in)
2903 |  51 |         self.K = int(K)
2904 |  52 |         self.rho = float(rho)
2905 |  53 |         self.per_channel = bool(per_channel)
2906 |  54 |         self.eps = float(eps)
2907 |  55 | 
2908 |  56 |         # 原始可学习参数 u，经 softplus 后非负
2909 |  57 |         if self.per_channel:
2910 |  58 |             self.u = nn.Parameter(torch.zeros(self.d_in, self.K))
2911 |  59 |         else:
2912 |  60 |             self.u = nn.Parameter(torch.zeros(self.K))
2913 |  61 | 
2914 |  62 |         # 预先缓存 rho 的幂 [K]
2915 |  63 |         rho_pow = torch.tensor([self.rho ** k for k in range(self.K)], dtype=torch.float32)
2916 |  64 |         self.register_buffer("rho_pow", rho_pow, persistent=True)
2917 |  65 | 
2918 |  66 |     def extra_repr(self) -> str:
2919 |  67 |         return f"d_in={self.d_in}, K={self.K}, rho={self.rho}, per_channel={self.per_channel}"
2920 |  68 | 
2921 |  69 |     @torch.no_grad()
2922 |  70 |     def get_delay_weights(self) -> torch.Tensor:
2923 |  71 |         """
2924 |  72 |         返回当前归一化后的延迟权重 w，形状:
2925 |  73 |           - per_channel=True: [d_in, K]
2926 |  74 |           - per_channel=False: [K]
2927 |  75 |         便于监控/可视化。
2928 |  76 |         """
2929 |  77 |         if self.per_channel:
2930 |  78 |             # [d_in, K]
2931 |  79 |             sp = F.softplus(self.u)
2932 |  80 |             num = sp * self.rho_pow  # 广播到 [d_in, K]
2933 |  81 |             den = num.sum(dim=1, keepdim=True).clamp_min(self.eps)
2934 |  82 |             w = num / den
2935 |  83 |             return w
2936 |  84 |         else:
2937 |  85 |             # [K]
2938 |  86 |             sp = F.softplus(self.u)
2939 |  87 |             num = sp * self.rho_pow
2940 |  88 |             den = num.sum().clamp_min(self.eps)
2941 |  89 |             w = num / den
2942 |  90 |             return w
2943 |  91 | 
2944 |  92 |     def _build_depthwise_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
2945 |  93 |         """
2946 |  94 |         构造 Conv1d 的 depthwise 卷积核:
2947 |  95 |           - 形状 [d_in, 1, K]
2948 |  96 |           - 每个通道自有一条核（若 per_channel=True），否则共享一条核并复制
2949 |  97 |         """
2950 |  98 |         if self.per_channel:
2951 |  99 |             # [d_in, K] -> [d_in, 1, K]
2952 | 100 |             w = self.get_delay_weights().to(device=device, dtype=dtype).unsqueeze(1)
2953 | 101 |         else:
2954 | 102 |             # 共享核 [K] -> [d_in, 1, K]
2955 | 103 |             w_shared = self.get_delay_weights().to(device=device, dtype=dtype).view(1, 1, self.K)
2956 | 104 |             w = w_shared.expand(self.d_in, 1, self.K).contiguous()
2957 | 105 |         return w  # Float[d_in, 1, K]
2958 | 106 | 
2959 | 107 |     def forward(self, H: torch.Tensor) -> torch.Tensor:
2960 | 108 |         """
2961 | 109 |         参数
2962 | 110 |         ----
2963 | 111 |         H : Float[T, N, d_in]
2964 | 112 | 
2965 | 113 |         返回
2966 | 114 |         ----
2967 | 115 |         H_tilde : Float[T, N, d_in]
2968 | 116 |         """
2969 | 117 |         assert H.dim() == 3, "输入 H 形状应为 [T, N, d_in]"
2970 | 118 |         T, N, Din = H.shape
2971 | 119 |         assert Din == self.d_in, f"d_in 不匹配: 期望 {self.d_in}, 实得 {Din}"
2972 | 120 | 
2973 | 121 |         # [T, N, d] -> [N, d, T]
2974 | 122 |         x = H.permute(1, 2, 0).contiguous()
2975 | 123 | 
2976 | 124 |         # 因果左填充 K-1
2977 | 125 |         pad_left = self.K - 1
2978 | 126 |         if pad_left > 0:
2979 | 127 |             x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)  # 在时间维左侧填充
2980 | 128 | 
2981 | 129 |         # 深度可分离卷积 (groups=d_in)
2982 | 130 |         weight = self._build_depthwise_kernel(H.device, H.dtype)  # [d, 1, K]
2983 | 131 |         y = F.conv1d(x, weight=weight, bias=None, stride=1, padding=0, groups=self.d_in)
2984 | 132 |         # y: [N, d, T]
2985 | 133 | 
2986 | 134 |         # 回到 [T, N, d]
2987 | 135 |         H_tilde = y.permute(2, 0, 1).contiguous()
2988 | 136 |         return H_tilde
2989 | ```
2990 | 
2991 | ## File: F:\SomeProjects\CSGNN\spikenet_x\lif_cell.py
2992 | 
2993 | - Extension: .py
2994 | - Language: python
2995 | - Size: 4914 bytes
2996 | - Created: 2025-08-22 12:57:19
2997 | - Modified: 2025-08-22 12:57:41
2998 | 
2999 | ### Code
3000 | 
3001 | ```python
3002 |   1 | # -*- coding: utf-8 -*-
3003 |   2 | """
3004 |   3 | LIFCell: 脉冲神经元单元（支持自适应阈值与 fast-tanh 代理梯度）
3005 |   4 | 
3006 |   5 | 接口
3007 |   6 | ----
3008 |   7 | forward(M: Float[T, N, d]) -> Tuple[S: Float[T, N], V: Float[T, N], aux: Dict]
3009 |   8 | - M 为从聚合器得到的消息（电流输入）
3010 |   9 | - 先用线性投影 U: R^d -> R 将通道聚合为标量电流 I_tn
3011 |  10 | - 递推更新膜电位与阈值，产生脉冲
3012 |  11 | 
3013 |  12 | 参考公式（提示词）
3014 |  13 | ----------------
3015 |  14 | V_{i,t} = λ V_{i,t-1} + U m_{i,t} - θ_{i,t-1} R_{i,t-1}
3016 |  15 | S_{i,t} = 𝟙[V_{i,t} > θ_{i,t}]
3017 |  16 | V_{i,t} ← V_{i,t} - S_{i,t} · θ_{i,t}          (重置)
3018 |  17 | θ_{i,t} = τ_θ θ_{i,t-1} + γ S_{i,t-1}          (自适应阈值，可选)
3019 |  18 | 
3020 |  19 | 训练
3021 |  20 | ----
3022 |  21 | - 使用 fast-tanh 代理梯度:
3023 |  22 |   y = H(x) + (tanh(βx) - tanh(βx).detach())
3024 |  23 |   其中 H(x) 为硬阶跃 (x>0)
3025 |  24 | """
3026 |  25 | 
3027 |  26 | from __future__ import annotations
3028 |  27 | 
3029 |  28 | from typing import Dict, Optional, Tuple
3030 |  29 | 
3031 |  30 | import torch
3032 |  31 | import torch.nn as nn
3033 |  32 | 
3034 |  33 | 
3035 |  34 | def _fast_tanh_surrogate(x: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
3036 |  35 |     """
3037 |  36 |     硬触发 + 平滑梯度的 STE 实现:
3038 |  37 |       forward: step(x)
3039 |  38 |       backward: tanh(βx) 的导数 (≈ β * (1 - tanh^2(βx)))
3040 |  39 |     """
3041 |  40 |     hard = (x > 0).to(x.dtype)
3042 |  41 |     soft = torch.tanh(beta * x)
3043 |  42 |     return hard + (soft - soft.detach())
3044 |  43 | 
3045 |  44 | 
3046 |  45 | class LIFCell(nn.Module):
3047 |  46 |     def __init__(
3048 |  47 |         self,
3049 |  48 |         d: int,
3050 |  49 |         lambda_mem: float = 0.95,
3051 |  50 |         tau_theta: float = 0.99,
3052 |  51 |         gamma: float = 0.10,
3053 |  52 |         adaptive: bool = True,
3054 |  53 |         surrogate: str = "fast_tanh",
3055 |  54 |         beta: float = 2.0,
3056 |  55 |     ) -> None:
3057 |  56 |         super().__init__()
3058 |  57 |         assert 0.0 <= lambda_mem <= 1.0
3059 |  58 |         assert 0.0 <= tau_theta <= 1.0
3060 |  59 |         assert gamma >= 0.0
3061 |  60 | 
3062 |  61 |         self.d = int(d)
3063 |  62 |         self.adaptive = bool(adaptive)
3064 |  63 |         self.surrogate = str(surrogate)
3065 |  64 |         self.beta = float(beta)
3066 |  65 | 
3067 |  66 |         # U: R^d -> R（共享于所有节点），无偏置避免电流漂移
3068 |  67 |         self.proj = nn.Linear(d, 1, bias=False)
3069 |  68 | 
3070 |  69 |         # 将标量参数注册为 buffer，便于脚本化与移动设备
3071 |  70 |         self.register_buffer("lambda_mem", torch.as_tensor(lambda_mem, dtype=torch.float32))
3072 |  71 |         self.register_buffer("tau_theta", torch.as_tensor(tau_theta, dtype=torch.float32))
3073 |  72 |         self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float32))
3074 |  73 | 
3075 |  74 |     def _spike(self, x: torch.Tensor) -> torch.Tensor:
3076 |  75 |         if self.surrogate == "fast_tanh":
3077 |  76 |             return _fast_tanh_surrogate(x, beta=self.beta)
3078 |  77 |         # 兜底：纯硬阈值（无代理梯度）
3079 |  78 |         return (x > 0).to(x.dtype)
3080 |  79 | 
3081 |  80 |     @torch.no_grad()
3082 |  81 |     def reset_parameters(self) -> None:
3083 |  82 |         nn.init.xavier_uniform_(self.proj.weight)
3084 |  83 | 
3085 |  84 |     def forward(
3086 |  85 |         self,
3087 |  86 |         M: torch.Tensor,                # [T, N, d]
3088 |  87 |         state0: Optional[Dict] = None,  # 可选: {"V": [N], "theta": [N], "S": [N]}
3089 |  88 |     ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
3090 |  89 |         assert M.dim() == 3, "M 形状应为 [T, N, d]"
3091 |  90 |         T, N, d = M.shape
3092 |  91 |         assert d == self.d, f"d 不匹配: 期望 {self.d}, 实得 {d}"
3093 |  92 | 
3094 |  93 |         device = M.device
3095 |  94 |         dtype = M.dtype
3096 |  95 | 
3097 |  96 |         # 初始状态
3098 |  97 |         if state0 is None:
3099 |  98 |             V = torch.zeros(N, device=device, dtype=dtype)
3100 |  99 |             theta = torch.ones(N, device=device, dtype=dtype)  # 初始阈值 1.0
3101 | 100 |             S_prev = torch.zeros(N, device=device, dtype=dtype)
3102 | 101 |         else:
3103 | 102 |             V = state0.get("V", torch.zeros(N, device=device, dtype=dtype)).to(dtype)
3104 | 103 |             theta = state0.get("theta", torch.ones(N, device=device, dtype=dtype)).to(dtype)
3105 | 104 |             S_prev = state0.get("S", torch.zeros(N, device=device, dtype=dtype)).to(dtype)
3106 | 105 | 
3107 | 106 |         S_seq = []
3108 | 107 |         V_seq = []
3109 | 108 |         theta_seq = []
3110 | 109 | 
3111 | 110 |         lam = self.lambda_mem
3112 | 111 |         tau = self.tau_theta
3113 | 112 |         gam = self.gamma
3114 | 113 | 
3115 | 114 |         for t in range(T):
3116 | 115 |             # 投影到标量电流 I_tn: [N]
3117 | 116 |             I = self.proj(M[t]).squeeze(-1)  # [N]
3118 | 117 | 
3119 | 118 |             # 记忆衰减 + 输入累积
3120 | 119 |             V = lam * V + I - (theta * S_prev)  # 包含上一步的 Refractory 抑制项
3121 | 120 | 
3122 | 121 |             # 触发条件与代理梯度
3123 | 122 |             x = V - theta
3124 | 123 |             S = self._spike(x)  # [N] in [0,1]
3125 | 124 | 
3126 | 125 |             # 重置：发放处扣除阈值
3127 | 126 |             V = V - S * theta
3128 | 127 | 
3129 | 128 |             # 自适应阈值
3130 | 129 |             if self.adaptive:
3131 | 130 |                 theta = tau * theta + gam * S_prev
3132 | 131 | 
3133 | 132 |             # 记录
3134 | 133 |             S_seq.append(S)
3135 | 134 |             V_seq.append(V)
3136 | 135 |             theta_seq.append(theta)
3137 | 136 | 
3138 | 137 |             # 更新上一时刻的发放
3139 | 138 |             S_prev = S
3140 | 139 | 
3141 | 140 |         S_out = torch.stack(S_seq, dim=0)  # [T, N]
3142 | 141 |         V_out = torch.stack(V_seq, dim=0)  # [T, N]
3143 | 142 | 
3144 | 143 |         aux = {
3145 | 144 |             "theta": torch.stack(theta_seq, dim=0),   # [T, N]
3146 | 145 |             "spike_rate": S_out.mean().detach(),      # 标量，便于监控
3147 | 146 |         }
3148 | 147 |         return S_out, V_out, aux
3149 | ```
3150 | 
3151 | ## File: F:\SomeProjects\CSGNN\spikenet_x\masked_ops.py
3152 | 
3153 | - Extension: .py
3154 | - Language: python
3155 | - Size: 4560 bytes
3156 | - Created: 2025-08-22 12:50:15
3157 | - Modified: 2025-08-22 12:50:55
3158 | 
3159 | ### Code
3160 | 
3161 | ```python
3162 |   1 | # -*- coding: utf-8 -*-
3163 |   2 | """
3164 |   3 | 掩码与 Top-k 相关的张量操作（纯 PyTorch 实现）
3165 |   4 | 
3166 |   5 | 函数约定
3167 |   6 | --------
3168 |   7 | - 所有 logits/score 相关的函数在被掩蔽位置填充为 -inf（或非常负的数），
3169 |   8 |   再做 softmax，以确保数值与归一化正确。
3170 |   9 | - 所有张量形状均保持与输入一致，除非特别说明。
3171 |  10 | 
3172 |  11 | 作者: Cline
3173 |  12 | """
3174 |  13 | 
3175 |  14 | from typing import Optional, Tuple
3176 |  15 | 
3177 |  16 | import torch
3178 |  17 | import torch.nn.functional as F
3179 |  18 | 
3180 |  19 | 
3181 |  20 | NEG_INF = -1e30  # 作为 -inf 的数值近似，避免部分设备对 -inf 的不一致处理
3182 |  21 | 
3183 |  22 | 
3184 |  23 | def fill_masked_(logits: torch.Tensor, mask: torch.Tensor, value: float = NEG_INF) -> torch.Tensor:
3185 |  24 |     """
3186 |  25 |     原地将 mask==0 的位置填充为 value（默认近似 -inf）。
3187 |  26 | 
3188 |  27 |     参数
3189 |  28 |     ----
3190 |  29 |     logits : Float[...]
3191 |  30 |         任意形状的分数张量
3192 |  31 |     mask : Bool/Byte[...]
3193 |  32 |         与 logits 同形，True/1 表示可用，False/0 表示被掩蔽
3194 |  33 |     value : float
3195 |  34 |         被掩蔽位置写入的值
3196 |  35 | 
3197 |  36 |     返回
3198 |  37 |     ----
3199 |  38 |     logits : Float[...]
3200 |  39 |         与输入同一引用的张量（原地修改）
3201 |  40 |     """
3202 |  41 |     if mask.dtype != torch.bool:
3203 |  42 |         mask = mask != 0
3204 |  43 |     logits.masked_fill_(~mask, value)
3205 |  44 |     return logits
3206 |  45 | 
3207 |  46 | 
3208 |  47 | def masked_softmax(
3209 |  48 |     logits: torch.Tensor,
3210 |  49 |     mask: Optional[torch.Tensor] = None,
3211 |  50 |     dim: int = -1,
3212 |  51 |     temperature: float = 1.0,
3213 |  52 | ) -> torch.Tensor:
3214 |  53 |     """
3215 |  54 |     在给定维度上对带掩码的 logits 进行 softmax。
3216 |  55 | 
3217 |  56 |     - 先在掩蔽位置写入 -inf，再做 softmax，避免“先置零后归一”导致的数值偏差。
3218 |  57 |     - 支持温度缩放：logits / temperature
3219 |  58 | 
3220 |  59 |     参数
3221 |  60 |     ----
3222 |  61 |     logits : Float[...]
3223 |  62 |     mask : Bool/Byte[...] or None
3224 |  63 |         与 logits 广播兼容；为 None 时等价于全 True
3225 |  64 |     dim : int
3226 |  65 |     temperature : float
3227 |  66 | 
3228 |  67 |     返回
3229 |  68 |     ----
3230 |  69 |     probs : Float[...]
3231 |  70 |     """
3232 |  71 |     if temperature != 1.0:
3233 |  72 |         logits = logits / float(temperature)
3234 |  73 | 
3235 |  74 |     if mask is not None:
3236 |  75 |         # 为避免修改外部张量，做一份拷贝
3237 |  76 |         logits = logits.clone()
3238 |  77 |         fill_masked_(logits, mask, NEG_INF)
3239 |  78 | 
3240 |  79 |     # 数值稳定 softmax
3241 |  80 |     max_val, _ = torch.max(logits, dim=dim, keepdim=True)
3242 |  81 |     shifted = logits - max_val
3243 |  82 |     exp = torch.exp(shifted)
3244 |  83 |     if mask is not None:
3245 |  84 |         if mask.dtype != torch.bool:
3246 |  85 |             mask = mask != 0
3247 |  86 |         exp = exp * mask.to(dtype=exp.dtype)
3248 |  87 | 
3249 |  88 |     denom = torch.clamp(exp.sum(dim=dim, keepdim=True), min=1e-12)
3250 |  89 |     return exp / denom
3251 |  90 | 
3252 |  91 | 
3253 |  92 | def topk_mask_logits(
3254 |  93 |     logits: torch.Tensor,
3255 |  94 |     k: int,
3256 |  95 |     dim: int = -1,
3257 |  96 |     inplace: bool = False,
3258 |  97 | ) -> Tuple[torch.Tensor, torch.Tensor]:
3259 |  98 |     """
3260 |  99 |     在维度 dim 上选出前 k 的元素，其余位置置为 -inf（或近似值）。
3261 | 100 | 
3262 | 101 |     注意：
3263 | 102 |     - 该函数只在 logits 上执行 Top-k 筛选，不做 softmax。
3264 | 103 |     - 返回 (new_logits, keep_mask)
3265 | 104 | 
3266 | 105 |     参数
3267 | 106 |     ----
3268 | 107 |     logits : Float[...]
3269 | 108 |     k : int
3270 | 109 |         k >= 1
3271 | 110 |     dim : int
3272 | 111 |     inplace : bool
3273 | 112 |         是否原地写回
3274 | 113 | 
3275 | 114 |     返回
3276 | 115 |     ----
3277 | 116 |     new_logits : Float[...]
3278 | 117 |         仅保留 Top-k 的 logits；其余位置为 -inf
3279 | 118 |     keep_mask : Bool[...]
3280 | 119 |         True 表示该位置被保留
3281 | 120 |     """
3282 | 121 |     assert k >= 1, "topk must be >= 1"
3283 | 122 |     # 取 Top-k 的阈值
3284 | 123 |     topk_vals, topk_idx = torch.topk(logits, k=k, dim=dim)
3285 | 124 |     # 构造保留 mask
3286 | 125 |     keep_mask = torch.zeros_like(logits, dtype=torch.bool)
3287 | 126 |     keep_mask.scatter_(dim, topk_idx, True)
3288 | 127 | 
3289 | 128 |     if inplace:
3290 | 129 |         out = fill_masked_(logits, keep_mask, NEG_INF)
3291 | 130 |         return out, keep_mask
3292 | 131 |     else:
3293 | 132 |         new_logits = torch.where(keep_mask, logits, torch.full_like(logits, NEG_INF))
3294 | 133 |         return new_logits, keep_mask
3295 | 134 | 
3296 | 135 | 
3297 | 136 | def masked_topk_softmax(
3298 | 137 |     logits: torch.Tensor,
3299 | 138 |     mask: Optional[torch.Tensor],
3300 | 139 |     k: int,
3301 | 140 |     dim: int = -1,
3302 | 141 |     temperature: float = 1.0,
3303 | 142 | ) -> torch.Tensor:
3304 | 143 |     """
3305 | 144 |     组合操作：先对 logits 进行掩码，随后 Top-k 截断，再做 softmax。
3306 | 145 | 
3307 | 146 |     等价步骤：
3308 | 147 |       1) logits[~mask] = -inf
3309 | 148 |       2) 仅保留维度 dim 上的 Top-k，其余 = -inf
3310 | 149 |       3) softmax(dim)
3311 | 150 | 
3312 | 151 |     参数
3313 | 152 |     ----
3314 | 153 |     logits : Float[...]
3315 | 154 |     mask : Optional[Bool/Byte[...] ]
3316 | 155 |     k : int
3317 | 156 |     dim : int
3318 | 157 |     temperature : float
3319 | 158 | 
3320 | 159 |     返回
3321 | 160 |     ----
3322 | 161 |     probs : Float[...]
3323 | 162 |     """
3324 | 163 |     if mask is not None:
3325 | 164 |         logits = logits.clone()
3326 | 165 |         fill_masked_(logits, mask, NEG_INF)
3327 | 166 |     logits, _ = topk_mask_logits(logits, k=k, dim=dim, inplace=False)
3328 | 167 |     return masked_softmax(logits, mask=None, dim=dim, temperature=temperature)
3329 | ```
3330 | 
3331 | ## File: F:\SomeProjects\CSGNN\spikenet_x\minimal_example.py
3332 | 
3333 | - Extension: .py
3334 | - Language: python
3335 | - Size: 2572 bytes
3336 | - Created: 2025-08-22 13:11:54
3337 | - Modified: 2025-08-22 23:52:28
3338 | 
3339 | ### Code
3340 | 
3341 | ```python
3342 |  1 | # -*- coding: utf-8 -*-
3343 |  2 | """
3344 |  3 | SpikeNet-X 最小可运行示例
3345 |  4 | 
3346 |  5 | 运行方法：
3347 |  6 |     python -m spikenet_x.minimal_example
3348 |  7 | """
3349 |  8 | 
3350 |  9 | import torch
3351 | 10 | 
3352 | 11 | # 动态地添加 spikenet_x 包的父目录到 sys.path
3353 | 12 | # 以便在 spikenet_x 目录外也能运行此脚本（例如从项目根目录）
3354 | 13 | import os
3355 | 14 | import sys
3356 | 15 | if __package__ is None or __package__ == '':
3357 | 16 |     # a bit of a hack to get relative imports working when running as a script
3358 | 17 |     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
3359 | 18 |     from spikenet_x.model import SpikeNetX
3360 | 19 | else:
3361 | 20 |     from .model import SpikeNetX
3362 | 21 | 
3363 | 22 | 
3364 | 23 | def erdos_renyi_edge_index(num_nodes: int, p: float = 0.1, seed: int = 42) -> torch.Tensor:
3365 | 24 |     g = torch.Generator().manual_seed(seed)
3366 | 25 |     mask = torch.rand((num_nodes, num_nodes), generator=g) < p
3367 | 26 |     # 去除自环
3368 | 27 |     mask.fill_diagonal_(False)
3369 | 28 |     src, dst = mask.nonzero(as_tuple=True)
3370 | 29 |     # edge_index: [2, E]，列为 (src, dst)
3371 | 30 |     return torch.stack([src, dst], dim=0).to(torch.long)
3372 | 31 | 
3373 | 32 | 
3374 | 33 | def main():
3375 | 34 |     print("--- SpikeNet-X Minimal Example & Shape Check ---")
3376 | 35 |     
3377 | 36 |     T, N, d_in, d, Hs, L = 16, 64, 32, 64, 4, 2
3378 | 37 |     E = N * 5
3379 | 38 | 
3380 | 39 |     print(f"Params: T={T}, N={N}, d_in={d_in}, d={d}, heads={Hs}, layers={L}")
3381 | 40 | 
3382 | 41 |     H0 = torch.randn(T, N, d_in)
3383 | 42 |     edge_index = erdos_renyi_edge_index(N, p=0.05, seed=1)
3384 | 43 |     time_idx = torch.arange(T)
3385 | 44 | 
3386 | 45 |     print(f"Input shapes: H={H0.shape}, edge_index={edge_index.shape}, time_idx={time_idx.shape}")
3387 | 46 | 
3388 | 47 |     model = SpikeNetX(
3389 | 48 |         d_in=d_in,
3390 | 49 |         d=d,
3391 | 50 |         layers=L,
3392 | 51 |         heads=Hs,
3393 | 52 |         topk=8,
3394 | 53 |         W=8,
3395 | 54 |         attn_impl="sparse",
3396 | 55 |         out_dim=5,
3397 | 56 |     )
3398 | 57 |     model.eval() # 禁用 dropout
3399 | 58 |     print(f"\nModel:\n{model}\n")
3400 | 59 | 
3401 | 60 |     with torch.no_grad():
3402 | 61 |         out = model(H0, edge_index=edge_index, time_idx=time_idx)
3403 | 62 | 
3404 | 63 |     print("--- Output Shape Check ---")
3405 | 64 |     print(f"Y_last (final features): {out['Y_last'].shape}")
3406 | 65 |     print(f"S_list (spikes per layer): {out['S_list'].shape}")
3407 | 66 |     print(f"V_list (voltages per layer): {out['V_list'].shape}")
3408 | 67 |     print(f"logits (readout): {out['logits'].shape if out.get('logits') is not None else 'N/A'}")
3409 | 68 |     print(f"M_last (last layer msg): {out['M_last'].shape}")
3410 | 69 | 
3411 | 70 |     # 检查形状是否符合预期
3412 | 71 |     assert out["Y_last"].shape == (T, N, d)
3413 | 72 |     assert out["S_list"].shape == (L, T, N)
3414 | 73 |     assert out["V_list"].shape == (L, T, N)
3415 | 74 |     if out.get("logits") is not None:
3416 | 75 |         assert out["logits"].shape == (N, 5)
3417 | 76 | 
3418 | 77 |     print("\n✅ All shapes are correct.")
3419 | 78 | 
3420 | 79 | 
3421 | 80 | if __name__ == "__main__":
3422 | 81 |     main()
3423 | ```
3424 | 
3425 | ## File: F:\SomeProjects\CSGNN\spikenet_x\model.py
3426 | 
3427 | - Extension: .py
3428 | - Language: python
3429 | - Size: 7649 bytes
3430 | - Created: 2025-08-22 13:07:03
3431 | - Modified: 2025-08-22 23:32:42
3432 | 
3433 | ### Code
3434 | 
3435 | ```python
3436 |   1 | # -*- coding: utf-8 -*-
3437 |   2 | """
3438 |   3 | SpikeNet-X: multi-layer network composed of (DelayLine -> STA -> LIF) blocks.
3439 |   4 | 
3440 |   5 | This module provides a thin, task-agnostic backbone `SpikeNetX` that stacks
3441 |   6 | `SpikeNetXLayer` L times and (optionally) a lightweight readout head.
3442 |   7 | 
3443 |   8 | Key ideas follow `提示词.md`:
3444 |   9 | - Time-first tensors: H: [T, N, d_in], S: [T, N]
3445 |  10 | - Event-driven STA with causal window W and Top-k sparsification
3446 |  11 | - Learnable DelayLine in front of STA to model propagation delay
3447 |  12 | - LIF cell generates spikes that can gate attention in the next layer
3448 |  13 | 
3449 |  14 | Typical usage
3450 |  15 | -------------
3451 |  16 | >>> import torch
3452 |  17 | >>> from spikenet_x import SpikeNetX
3453 |  18 | >>> T, N, d_in, d, Hs, L = 32, 128, 64, 128, 4, 2
3454 |  19 | >>> H0 = torch.randn(T, N, d_in)
3455 |  20 | >>> edge_index = torch.randint(0, N, (2, 4*N))  # toy edges
3456 |  21 | >>> time_idx = torch.arange(T)
3457 |  22 | >>> model = SpikeNetX(d_in=d_in, d=d, layers=L, heads=Hs, topk=16, W=32, out_dim=10)
3458 |  23 | >>> out = model(H0, edge_index=edge_index, time_idx=time_idx)
3459 |  24 | >>> out["logits"].shape  # [N, out_dim] by default (last-time readout)
3460 |  25 | torch.Size([128, 10])
3461 |  26 | """
3462 |  27 | 
3463 |  28 | from __future__ import annotations
3464 |  29 | 
3465 |  30 | from typing import Dict, List, Optional, Tuple
3466 |  31 | 
3467 |  32 | import torch
3468 |  33 | import torch.nn as nn
3469 |  34 | 
3470 |  35 | from .spikenetx_layer import SpikeNetXLayer
3471 |  36 | 
3472 |  37 | 
3473 |  38 | class SpikeNetX(nn.Module):
3474 |  39 |     """
3475 |  40 |     A stack of SpikeNetXLayer blocks with optional classifier head.
3476 |  41 | 
3477 |  42 |     Args
3478 |  43 |     ----
3479 |  44 |     d_in: int
3480 |  45 |         Input feature dimension.
3481 |  46 |     d: int
3482 |  47 |         Hidden/STA output dimension per layer.
3483 |  48 |     layers: int
3484 |  49 |         Number of stacked layers.
3485 |  50 |     heads: int
3486 |  51 |         Number of attention heads per layer.
3487 |  52 |     topk: int
3488 |  53 |         Top-k candidates kept per (i,t) in STA.
3489 |  54 |     W: int
3490 |  55 |         Causal attention time window.
3491 |  56 |     K: int
3492 |  57 |         DelayLine taps.
3493 |  58 |     rho: float
3494 |  59 |         DelayLine exponential discount.
3495 |  60 |     use_rel_bias: bool
3496 |  61 |         Whether to use learnable relative bias b[Δt].
3497 |  62 |     attn_drop: float
3498 |  63 |         Attention dropout prob.
3499 |  64 |     temp: float
3500 |  65 |         Softmax temperature for attention logits.
3501 |  66 |     per_channel: bool
3502 |  67 |         Per-channel DelayLine weights if True (recommended).
3503 |  68 |     ffn_hidden_mult: int
3504 |  69 |         Multiplier of FFN hidden width inside each layer.
3505 |  70 |     ffn_drop: float
3506 |  71 |         Dropout inside layer FFN.
3507 |  72 |     lif_*: see LIFCell.
3508 |  73 |     out_dim: Optional[int]
3509 |  74 |         If set, attach a linear head to produce logits for node-level tasks.
3510 |  75 |     readout: str
3511 |  76 |         'last' (default): use last time-step T-1 for logits,
3512 |  77 |         'mean': temporal mean pooling over T.
3513 |  78 |     """
3514 |  79 | 
3515 |  80 |     def __init__(
3516 |  81 |         self,
3517 |  82 |         d_in: int,
3518 |  83 |         d: int,
3519 |  84 |         layers: int = 2,
3520 |  85 |         heads: int = 4,
3521 |  86 |         topk: int = 16,
3522 |  87 |         W: int = 64,
3523 |  88 |         K: int = 5,
3524 |  89 |         rho: float = 0.85,
3525 |  90 |         use_rel_bias: bool = True,
3526 |  91 |         attn_drop: float = 0.1,
3527 |  92 |         temp: float = 1.0,
3528 |  93 |         attn_impl: str = "dense",
3529 |  94 |         per_channel: bool = True,
3530 |  95 |         ffn_hidden_mult: int = 4,
3531 |  96 |         ffn_drop: float = 0.1,
3532 |  97 |         lif_lambda_mem: float = 0.95,
3533 |  98 |         lif_tau_theta: float = 0.99,
3534 |  99 |         lif_gamma: float = 0.10,
3535 | 100 |         lif_adaptive: bool = True,
3536 | 101 |         lif_surrogate: str = "fast_tanh",
3537 | 102 |         lif_beta: float = 2.0,
3538 | 103 |         out_dim: Optional[int] = None,
3539 | 104 |         readout: str = "last",
3540 | 105 |     ) -> None:
3541 | 106 |         super().__init__()
3542 | 107 |         assert layers >= 1, "layers must be >= 1"
3543 | 108 |         assert readout in ("last", "mean"), "readout must be 'last' or 'mean'"
3544 | 109 | 
3545 | 110 |         self.layers = int(layers)
3546 | 111 |         self.readout = readout
3547 | 112 |         self.out_dim = out_dim
3548 | 113 |         self.attn_impl = attn_impl
3549 | 114 |         assert self.attn_impl in ("dense", "sparse"), "attn_impl must be 'dense' or 'sparse'"
3550 | 115 | 
3551 | 116 |         mods: List[SpikeNetXLayer] = []
3552 | 117 |         for l in range(layers):
3553 | 118 |             in_dim = d_in if l == 0 else d
3554 | 119 |             mods.append(
3555 | 120 |                 SpikeNetXLayer(
3556 | 121 |                     d_in=in_dim,
3557 | 122 |                     d=d,
3558 | 123 |                     heads=heads,
3559 | 124 |                     topk=topk,
3560 | 125 |                     W=W,
3561 | 126 |                     K=K,
3562 | 127 |                     rho=rho,
3563 | 128 |                     use_rel_bias=use_rel_bias,
3564 | 129 |                     attn_drop=attn_drop,
3565 | 130 |                     temp=temp,
3566 | 131 |                     attn_impl=attn_impl,
3567 | 132 |                     per_channel=per_channel,
3568 | 133 |                     ffn_hidden_mult=ffn_hidden_mult,
3569 | 134 |                     ffn_drop=ffn_drop,
3570 | 135 |                     lif_lambda_mem=lif_lambda_mem,
3571 | 136 |                     lif_tau_theta=lif_tau_theta,
3572 | 137 |                     lif_gamma=lif_gamma,
3573 | 138 |                     lif_adaptive=lif_adaptive,
3574 | 139 |                     lif_surrogate=lif_surrogate,
3575 | 140 |                     lif_beta=lif_beta,
3576 | 141 |                 )
3577 | 142 |             )
3578 | 143 |         self.blocks = nn.ModuleList(mods)
3579 | 144 | 
3580 | 145 |         self.head = nn.Linear(d, out_dim, bias=True) if out_dim is not None else None
3581 | 146 |         if self.head is not None:
3582 | 147 |             nn.init.xavier_uniform_(self.head.weight)
3583 | 148 |             nn.init.zeros_(self.head.bias)
3584 | 149 | 
3585 | 150 |     def forward(
3586 | 151 |         self,
3587 | 152 |         H: torch.Tensor,                        # [T, N, d_in]
3588 | 153 |         edge_index: Optional[torch.Tensor],     # [2, E] or None (if adj_mask provided)
3589 | 154 |         time_idx: torch.Tensor,                 # [T]
3590 | 155 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool or None
3591 | 156 |         S0: Optional[torch.Tensor] = None,        # initial spikes for layer-0 gating [T, N] (optional)
3592 | 157 |     ) -> Dict[str, torch.Tensor]:
3593 | 158 |         assert H.dim() == 3, "H should be [T, N, d_in]"
3594 | 159 |         T, N, _ = H.shape
3595 | 160 |         assert time_idx.dim() == 1 and time_idx.numel() == T, "time_idx must be [T]"
3596 | 161 | 
3597 | 162 |         S_prev = S0  # first layer gating; None -> all-ones gating inside block
3598 | 163 |         Y = None
3599 | 164 |         S_list: List[torch.Tensor] = []
3600 | 165 |         V_list: List[torch.Tensor] = []
3601 | 166 | 
3602 | 167 |         X = H
3603 | 168 |         aux_last: Dict[str, torch.Tensor] = {}
3604 | 169 |         for blk in self.blocks:
3605 | 170 |             S, V, Y, aux = blk(
3606 | 171 |                 H=X,
3607 | 172 |                 S_prev=S_prev,
3608 | 173 |                 edge_index=edge_index,
3609 | 174 |                 time_idx=time_idx,
3610 | 175 |                 adj_mask=adj_mask,
3611 | 176 |             )
3612 | 177 |             S_list.append(S)   # each: [T, N]
3613 | 178 |             V_list.append(V)
3614 | 179 |             S_prev = S         # spikes feed-forward as gate for next layer
3615 | 180 |             X = Y              # features for next layer
3616 | 181 |             aux_last = aux
3617 | 182 | 
3618 | 183 |         # Readout
3619 | 184 |         if self.readout == "last":
3620 | 185 |             z = Y[-1]  # [N, d]
3621 | 186 |         else:  # "mean"
3622 | 187 |             z = Y.mean(dim=0)  # [N, d]
3623 | 188 | 
3624 | 189 |         logits = self.head(z) if self.head is not None else None
3625 | 190 | 
3626 | 191 |         out: Dict[str, torch.Tensor] = {
3627 | 192 |             "repr": z,                       # [N, d]
3628 | 193 |             "Y_last": Y,                     # [T, N, d]
3629 | 194 |             "S_list": torch.stack(S_list),   # [L, T, N]
3630 | 195 |             "V_list": torch.stack(V_list),   # [L, T, N]
3631 | 196 |         }
3632 | 197 |         if logits is not None:
3633 | 198 |             out["logits"] = logits           # [N, out_dim]
3634 | 199 | 
3635 | 200 |         # for convenience: expose a few internals when available
3636 | 201 |         if "M" in aux_last:
3637 | 202 |             out["M_last"] = aux_last["M"]    # [T, N, d]
3638 | 203 | 
3639 | 204 |         return out
3640 | 205 | 
3641 | 206 | 
3642 | 207 | def shape_check_demo() -> Tuple[torch.Size, Optional[torch.Size]]:
3643 | 208 |     """
3644 | 209 |     Minimal shape check (no training). Returns (Y_last_shape, logits_shape).
3645 | 210 |     """
3646 | 211 |     T, N, d_in, d, Hs, L = 16, 64, 32, 64, 4, 2
3647 | 212 |     E = N * 4
3648 | 213 |     H0 = torch.randn(T, N, d_in)
3649 | 214 |     edge_index = torch.randint(0, N, (2, E))
3650 | 215 |     time_idx = torch.arange(T)
3651 | 216 | 
3652 | 217 |     model = SpikeNetX(d_in=d_in, d=d, layers=L, heads=Hs, topk=8, W=8, out_dim=5)
3653 | 218 |     out = model(H0, edge_index=edge_index, time_idx=time_idx)
3654 | 219 |     return out["Y_last"].shape, out.get("logits", None).shape if "logits" in out else None
3655 | ```
3656 | 
3657 | ## File: F:\SomeProjects\CSGNN\spikenet_x\rel_time.py
3658 | 
3659 | - Extension: .py
3660 | - Language: python
3661 | - Size: 6544 bytes
3662 | - Created: 2025-08-22 12:52:55
3663 | - Modified: 2025-08-22 12:53:38
3664 | 
3665 | ### Code
3666 | 
3667 | ```python
3668 |   1 | # -*- coding: utf-8 -*-
3669 |   2 | """
3670 |   3 | 相对时间编码与相对偏置
3671 |   4 | 
3672 |   5 | - 提供 RelativeTimeEncoding(nn.Module)
3673 |   6 |   * forward(time_idx: Long[T], W:int) -> (pe_table: Float[W+1, d_pe], rel_bias: Float[W+1])
3674 |   7 |   * pe_table[k] 表示 Δt = k 的编码向量（仅用于 0..W）
3675 |   8 |   * rel_bias 为可学习标量偏置 b[Δt]，长度 W+1
3676 |   9 | 
3677 |  10 | - 设计：
3678 |  11 |   * 指数衰减对（tau_m, tau_s）
3679 |  12 |   * 对数间隔频率的 sin/cos 对（n_freq 个频率）
3680 |  13 |   * 可选 log-bucket one-hot（num_buckets=0 表示关闭）
3681 |  14 |   * 输出维度 d_pe = 2 + 2*n_freq + num_buckets
3682 |  15 | 
3683 |  16 | - 数值与工程：
3684 |  17 |   * 仅构造 0..W 的 Δt 子表，避免构建完整 [T,T] 矩阵
3685 |  18 |   * Δt 使用 float 计算后堆叠为编码
3686 |  19 | """
3687 |  20 | 
3688 |  21 | from __future__ import annotations
3689 |  22 | 
3690 |  23 | from typing import Tuple
3691 |  24 | 
3692 |  25 | import math
3693 |  26 | import torch
3694 |  27 | import torch.nn as nn
3695 |  28 | 
3696 |  29 | 
3697 |  30 | class RelativeTimeEncoding(nn.Module):
3698 |  31 |     """
3699 |  32 |     相对时间编码（仅依赖 Δt），并带可学习相对偏置表 b[0..W]。
3700 |  33 | 
3701 |  34 |     参数
3702 |  35 |     ----
3703 |  36 |     taus : Tuple[float, float]
3704 |  37 |         指数衰减的两个时间常数 (tau_m, tau_s)
3705 |  38 |     n_freq : int
3706 |  39 |         正弦/余弦的对数间隔频率个数
3707 |  40 |     num_buckets : int
3708 |  41 |         log-bucket one-hot 的桶数（0 表示关闭）
3709 |  42 |     """
3710 |  43 | 
3711 |  44 |     def __init__(
3712 |  45 |         self,
3713 |  46 |         taus: Tuple[float, float] = (4.0, 16.0),
3714 |  47 |         n_freq: int = 3,
3715 |  48 |         num_buckets: int = 0,
3716 |  49 |     ) -> None:
3717 |  50 |         super().__init__()
3718 |  51 |         assert len(taus) == 2 and taus[0] > 0 and taus[1] > 0
3719 |  52 |         assert n_freq >= 0
3720 |  53 |         assert num_buckets >= 0
3721 |  54 | 
3722 |  55 |         self.tau_m = float(taus[0])
3723 |  56 |         self.tau_s = float(taus[1])
3724 |  57 |         self.n_freq = int(n_freq)
3725 |  58 |         self.num_buckets = int(num_buckets)
3726 |  59 | 
3727 |  60 |         # 缓存最近一次构造的 W，以便重用不同 batch 的同一窗口
3728 |  61 |         self._cached_W = None
3729 |  62 |         self.register_buffer("_cached_pe", None, persistent=False)
3730 |  63 |         self.register_buffer("_cached_bias", None, persistent=False)
3731 |  64 | 
3732 |  65 |         # 注意：rel_bias 的长度依赖于 W，故在 forward 时按需创建/扩展为 Parameter
3733 |  66 |         self.register_parameter("_rel_bias", None)
3734 |  67 | 
3735 |  68 |     @property
3736 |  69 |     def d_pe(self) -> int:
3737 |  70 |         # 2 (双指数) + 2*n_freq (sin/cos 对) + num_buckets (one-hot)
3738 |  71 |         return 2 + 2 * self.n_freq + self.num_buckets
3739 |  72 | 
3740 |  73 |     @staticmethod
3741 |  74 |     def _log_spaced_frequencies(n_freq: int, W: int) -> torch.Tensor:
3742 |  75 |         """
3743 |  76 |         生成对数间隔频率（角频率 ω），范围大致覆盖 [1/(2W), 1/2]（经验值）。
3744 |  77 |         """
3745 |  78 |         if n_freq <= 0:
3746 |  79 |             return torch.empty(0)
3747 |  80 |         f_min = 1.0 / max(2.0 * W, 1.0)
3748 |  81 |         f_max = 0.5
3749 |  82 |         freqs = torch.logspace(
3750 |  83 |             start=math.log10(f_min),
3751 |  84 |             end=math.log10(f_max),
3752 |  85 |             steps=n_freq,
3753 |  86 |         )
3754 |  87 |         # 转为角频率
3755 |  88 |         return 2.0 * math.pi * freqs
3756 |  89 | 
3757 |  90 |     @staticmethod
3758 |  91 |     def _log_bucketize(delta: torch.Tensor, num_buckets: int) -> torch.Tensor:
3759 |  92 |         """
3760 |  93 |         将 Δt 做对数分桶并返回 one-hot；delta >= 0 的整数张量。
3761 |  94 |         """
3762 |  95 |         if num_buckets <= 0:
3763 |  96 |             return torch.empty(delta.shape + (0,), device=delta.device, dtype=delta.dtype)
3764 |  97 | 
3765 |  98 |         # +1 防止 log(0)
3766 |  99 |         logv = torch.log2(delta.to(torch.float32) + 1.0)
3767 | 100 |         # 线性映射到桶 [0, num_buckets-1]
3768 | 101 |         idx = torch.clamp((logv / torch.clamp(logv.max(), min=1.0e-6)) * (num_buckets - 1), 0, num_buckets - 1)
3769 | 102 |         idx = idx.round().to(torch.long)
3770 | 103 | 
3771 | 104 |         one_hot = torch.zeros(delta.shape + (num_buckets,), device=delta.device, dtype=delta.dtype)
3772 | 105 |         one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)
3773 | 106 |         return one_hot
3774 | 107 | 
3775 | 108 |     def _build_pe_for_window(self, W: int, device: torch.device) -> torch.Tensor:
3776 | 109 |         """
3777 | 110 |         构建长度 W+1 的相对时间编码表：pe[k] = phi(Δt=k)，形状 [W+1, d_pe]
3778 | 111 |         """
3779 | 112 |         delta = torch.arange(0, W + 1, device=device, dtype=torch.float32)  # [0..W]
3780 | 113 | 
3781 | 114 |         # 双指数衰减
3782 | 115 |         exp_m = torch.exp(-delta / self.tau_m)  # [W+1]
3783 | 116 |         exp_s = torch.exp(-delta / self.tau_s)  # [W+1]
3784 | 117 | 
3785 | 118 |         # 正弦/余弦
3786 | 119 |         omegas = self._log_spaced_frequencies(self.n_freq, W).to(device)
3787 | 120 |         if omegas.numel() > 0:
3788 | 121 |             # [W+1, n_freq]
3789 | 122 |             arg = delta.unsqueeze(-1) * omegas.unsqueeze(0)
3790 | 123 |             sinv = torch.sin(arg)
3791 | 124 |             cosv = torch.cos(arg)
3792 | 125 |             sincos = torch.cat([sinv, cosv], dim=-1)  # [W+1, 2*n_freq]
3793 | 126 |         else:
3794 | 127 |             sincos = torch.empty((W + 1, 0), device=device, dtype=torch.float32)
3795 | 128 | 
3796 | 129 |         # log-bucket one-hot
3797 | 130 |         if self.num_buckets > 0:
3798 | 131 |             buckets = self._log_bucketize(delta.to(torch.long), self.num_buckets).to(torch.float32)  # [W+1, B]
3799 | 132 |         else:
3800 | 133 |             buckets = torch.empty((W + 1, 0), device=device, dtype=torch.float32)
3801 | 134 | 
3802 | 135 |         pe = torch.cat(
3803 | 136 |             [
3804 | 137 |                 exp_m.unsqueeze(-1),
3805 | 138 |                 exp_s.unsqueeze(-1),
3806 | 139 |                 sincos,
3807 | 140 |                 buckets,
3808 | 141 |             ],
3809 | 142 |             dim=-1,
3810 | 143 |         )
3811 | 144 |         return pe  # [W+1, d_pe]
3812 | 145 | 
3813 | 146 |     def _ensure_bias(self, W: int, device: torch.device) -> torch.Tensor:
3814 | 147 |         """
3815 | 148 |         确保存在长度 W+1 的可学习偏置表；如果已有更短表，做扩展并保留已学部分。
3816 | 149 |         """
3817 | 150 |         if self._rel_bias is None:
3818 | 151 |             self._rel_bias = nn.Parameter(torch.zeros(W + 1, device=device))
3819 | 152 |         elif self._rel_bias.numel() < (W + 1):
3820 | 153 |             old = self._rel_bias.data
3821 | 154 |             new = torch.zeros(W + 1, device=device)
3822 | 155 |             new[: old.numel()] = old
3823 | 156 |             self._rel_bias = nn.Parameter(new)
3824 | 157 |         return self._rel_bias
3825 | 158 | 
3826 | 159 |     def forward(
3827 | 160 |         self,
3828 | 161 |         time_idx: torch.Tensor,  # Long[T]，通常为 arange(T)
3829 | 162 |         W: int,
3830 | 163 |     ) -> Tuple[torch.Tensor, torch.Tensor]:
3831 | 164 |         """
3832 | 165 |         构建窗口 0..W 的相对时间编码子表与相对偏置。
3833 | 166 | 
3834 | 167 |         返回
3835 | 168 |         ----
3836 | 169 |         pe_table : Float[W+1, d_pe]
3837 | 170 |         rel_bias : Float[W+1]
3838 | 171 |         """
3839 | 172 |         assert time_idx.dim() == 1, "time_idx 应为一维 LongTensor [T]"
3840 | 173 |         assert W >= 0, "W >= 0"
3841 | 174 | 
3842 | 175 |         device = time_idx.device
3843 | 176 |         # 缓存与重用
3844 | 177 |         if self._cached_W == W and self._cached_pe is not None and self._cached_pe.device == device:
3845 | 178 |             pe = self._cached_pe
3846 | 179 |         else:
3847 | 180 |             pe = self._build_pe_for_window(W, device)
3848 | 181 |             self._cached_W = W
3849 | 182 |             self._cached_pe = pe
3850 | 183 | 
3851 | 184 |         rel_bias = self._ensure_bias(W, device)
3852 | 185 |         return pe, rel_bias
3853 | ```
3854 | 
3855 | ## File: F:\SomeProjects\CSGNN\spikenet_x\spikenetx_layer.py
3856 | 
3857 | - Extension: .py
3858 | - Language: python
3859 | - Size: 5988 bytes
3860 | - Created: 2025-08-22 13:04:32
3861 | - Modified: 2025-08-22 23:31:01
3862 | 
3863 | ### Code
3864 | 
3865 | ```python
3866 |   1 | # -*- coding: utf-8 -*-
3867 |   2 | """
3868 |   3 | SpikeNetXLayer: DelayLine + Spiking Temporal Attention (STA) + LIF 单层封装
3869 |   4 | 
3870 |   5 | 根据《提示词.md》的签名：
3871 |   6 |     class SpikeNetXLayer(nn.Module):
3872 |   7 |         def __init__(...):
3873 |   8 |             ...
3874 |   9 |         def forward(self, H, S_prev, edge_index, time_idx, adj_mask=None, batch=None):
3875 |  10 |             H̃ = self.delay(H)                                   # [T,N,d_in]
3876 |  11 |             M  = self.sta(H̃, S_prev, edge_index, time_idx, adj_mask)  # [T,N,d]
3877 |  12 |             S, V, aux = self.neuron(M)                           # [T,N], [T,N]
3878 |  13 |             Y = self.norm(M + self.ffn(M))                       # 残差 + 归一
3879 |  14 |             return S, V, Y, {"M": M, **aux}
3880 |  15 | 
3881 |  16 | 说明
3882 |  17 | ----
3883 |  18 | - 稠密 STA 回退实现在 spikenet_x/sta.py 中，适合小图或功能验证；
3884 |  19 | - 稀疏边 rolling-window 版本可在后续新增（接口保持一致）；
3885 |  20 | - 本层不改变时间长度 T 与节点数 N，仅改变通道维（d_in -> d）。
3886 |  21 | 
3887 |  22 | """
3888 |  23 | 
3889 |  24 | from __future__ import annotations
3890 |  25 | 
3891 |  26 | from typing import Dict, Optional, Tuple
3892 |  27 | 
3893 |  28 | import torch
3894 |  29 | import torch.nn as nn
3895 |  30 | import torch.nn.functional as F
3896 |  31 | 
3897 |  32 | from .delayline import LearnableDelayLine
3898 |  33 | from .sta import SpikingTemporalAttention
3899 |  34 | from .sta_sparse import SparseSpikingTemporalAttention
3900 |  35 | from .lif_cell import LIFCell
3901 |  36 | 
3902 |  37 | 
3903 |  38 | class MLP(nn.Module):
3904 |  39 |     def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.1) -> None:
3905 |  40 |         super().__init__()
3906 |  41 |         hidden = d * hidden_mult
3907 |  42 |         self.fc1 = nn.Linear(d, hidden, bias=True)
3908 |  43 |         self.fc2 = nn.Linear(hidden, d, bias=True)
3909 |  44 |         self.drop = nn.Dropout(dropout)
3910 |  45 | 
3911 |  46 |         self.reset_parameters()
3912 |  47 | 
3913 |  48 |     def reset_parameters(self) -> None:
3914 |  49 |         nn.init.xavier_uniform_(self.fc1.weight)
3915 |  50 |         nn.init.zeros_(self.fc1.bias)
3916 |  51 |         nn.init.xavier_uniform_(self.fc2.weight)
3917 |  52 |         nn.init.zeros_(self.fc2.bias)
3918 |  53 | 
3919 |  54 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3920 |  55 |         x = self.fc1(x)
3921 |  56 |         x = F.gelu(x)
3922 |  57 |         x = self.drop(x)
3923 |  58 |         x = self.fc2(x)
3924 |  59 |         return self.drop(x)
3925 |  60 | 
3926 |  61 | 
3927 |  62 | class SpikeNetXLayer(nn.Module):
3928 |  63 |     def __init__(
3929 |  64 |         self,
3930 |  65 |         d_in: int,
3931 |  66 |         d: int,
3932 |  67 |         heads: int = 4,
3933 |  68 |         topk: int = 16,
3934 |  69 |         W: int = 64,
3935 |  70 |         K: int = 5,
3936 |  71 |         rho: float = 0.85,
3937 |  72 |         use_rel_bias: bool = True,
3938 |  73 |         attn_drop: float = 0.1,
3939 |  74 |         temp: float = 1.0,
3940 |  75 |         attn_impl: str = "dense",
3941 |  76 |         per_channel: bool = True,
3942 |  77 |         ffn_hidden_mult: int = 4,
3943 |  78 |         ffn_drop: float = 0.1,
3944 |  79 |         lif_lambda_mem: float = 0.95,
3945 |  80 |         lif_tau_theta: float = 0.99,
3946 |  81 |         lif_gamma: float = 0.10,
3947 |  82 |         lif_adaptive: bool = True,
3948 |  83 |         lif_surrogate: str = "fast_tanh",
3949 |  84 |         lif_beta: float = 2.0,
3950 |  85 |     ) -> None:
3951 |  86 |         super().__init__()
3952 |  87 |         self.d_in = int(d_in)
3953 |  88 |         self.d = int(d)
3954 |  89 |         self.attn_impl = str(attn_impl)
3955 |  90 |         assert self.attn_impl in ("dense", "sparse"), "attn_impl must be 'dense' or 'sparse'"
3956 |  91 | 
3957 |  92 |         # 1) DelayLine（因果深度可分离 1D 卷积）
3958 |  93 |         self.delay = LearnableDelayLine(d_in=d_in, K=K, rho=rho, per_channel=per_channel)
3959 |  94 | 
3960 |  95 |         # 2) STA 聚合：根据 attn_impl 选择稠密/稀疏实现
3961 |  96 |         if self.attn_impl == "sparse":
3962 |  97 |             # 稀疏版本不支持 topk（后续可流式实现）
3963 |  98 |             self.sta = SparseSpikingTemporalAttention(
3964 |  99 |                 d_in=d_in,
3965 | 100 |                 d=d,
3966 | 101 |                 heads=heads,
3967 | 102 |                 W=W,
3968 | 103 |                 use_rel_bias=use_rel_bias,
3969 | 104 |                 attn_drop=attn_drop,
3970 | 105 |                 temp=temp,
3971 | 106 |             )
3972 | 107 |         else:
3973 | 108 |             self.sta = SpikingTemporalAttention(
3974 | 109 |                 d_in=d_in,
3975 | 110 |                 d=d,
3976 | 111 |                 heads=heads,
3977 | 112 |                 topk=topk,
3978 | 113 |                 W=W,
3979 | 114 |                 use_rel_bias=use_rel_bias,
3980 | 115 |                 attn_drop=attn_drop,
3981 | 116 |                 temp=temp,
3982 | 117 |             )
3983 | 118 | 
3984 | 119 |         # 3) 脉冲单元
3985 | 120 |         self.neuron = LIFCell(
3986 | 121 |             d=d,
3987 | 122 |             lambda_mem=lif_lambda_mem,
3988 | 123 |             tau_theta=lif_tau_theta,
3989 | 124 |             gamma=lif_gamma,
3990 | 125 |             adaptive=lif_adaptive,
3991 | 126 |             surrogate=lif_surrogate,
3992 | 127 |             beta=lif_beta,
3993 | 128 |         )
3994 | 129 | 
3995 | 130 |         # 归一与 FFN（残差）
3996 | 131 |         self.norm = nn.LayerNorm(d)
3997 | 132 |         self.ffn = MLP(d=d, hidden_mult=ffn_hidden_mult, dropout=ffn_drop)
3998 | 133 | 
3999 | 134 |     def forward(
4000 | 135 |         self,
4001 | 136 |         H: torch.Tensor,                       # [T, N, d_in]
4002 | 137 |         S_prev: Optional[torch.Tensor],        # [T, N] 或 None（None 时采用全 1 门控）
4003 | 138 |         edge_index: Optional[torch.Tensor],    # [2, E] 或 None（若提供 adj_mask 可为 None）
4004 | 139 |         time_idx: torch.Tensor,                # [T]
4005 | 140 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool
4006 | 141 |         batch: Optional[torch.Tensor] = None,     # 预留；当前未使用
4007 | 142 |     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
4008 | 143 |         assert H.dim() == 3, "H 形状应为 [T, N, d_in]"
4009 | 144 |         T, N, Din = H.shape
4010 | 145 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
4011 | 146 | 
4012 | 147 |         device = H.device
4013 | 148 |         dtype = H.dtype
4014 | 149 | 
4015 | 150 |         # 若上一层脉冲缺省，则使用全 1 门控（不抑制注意力）
4016 | 151 |         if S_prev is None:
4017 | 152 |             S_gate = torch.ones((T, N), device=device, dtype=dtype)
4018 | 153 |         else:
4019 | 154 |             assert S_prev.shape == (T, N), "S_prev 形状应为 [T, N]"
4020 | 155 |             S_gate = S_prev.to(device=device, dtype=dtype)
4021 | 156 | 
4022 | 157 |         # 1) DelayLine
4023 | 158 |         H_tilde = self.delay(H)  # [T, N, d_in]
4024 | 159 | 
4025 | 160 |         # 2) STA 聚合为 d 维消息
4026 | 161 |         M = self.sta(H_tilde, S_gate, edge_index=edge_index, time_idx=time_idx, adj_mask=adj_mask)  # [T, N, d]
4027 | 162 | 
4028 | 163 |         # 3) LIF 发放
4029 | 164 |         S, V, aux = self.neuron(M)  # S:[T,N], V:[T,N]
4030 | 165 |         aux = {"M": M, **aux}
4031 | 166 | 
4032 | 167 |         # 4) 残差 + 归一（Pre-LN 的一个简化变体）
4033 | 168 |         Y = self.norm(M + self.ffn(M))  # [T, N, d]
4034 | 169 | 
4035 | 170 |         return S, V, Y, aux
4036 | ```
4037 | 
4038 | ## File: F:\SomeProjects\CSGNN\spikenet_x\sta.py
4039 | 
4040 | - Extension: .py
4041 | - Language: python
4042 | - Size: 9234 bytes
4043 | - Created: 2025-08-22 12:59:28
4044 | - Modified: 2025-08-22 16:32:53
4045 | 
4046 | ### Code
4047 | 
4048 | ```python
4049 |   1 | # -*- coding: utf-8 -*-
4050 |   2 | """
4051 |   3 | SpikingTemporalAttention（STA）——稠密回退实现（小图/验证用）
4052 |   4 | 
4053 |   5 | 功能
4054 |   6 | ----
4055 |   7 | - 在因果与邻接掩码下，对 (邻居 j, 过去时间 t') 的键值进行多头注意力；
4056 |   8 | - 将源端脉冲 S[j,t'] 作为门控（在 logit 上加 log(S+eps) 等价于概率缩放）；
4057 |   9 | - 支持 Top-k 稀疏化（在 (j,t') 的联合候选维度上执行）；
4058 |  10 | - 支持相对时间编码与可学习相对偏置 b[Δt]；
4059 |  11 | - 输出每个 (t,i) 的聚合消息 M[t,i,:]，形状 [T, N, d]。
4060 |  12 | 
4061 |  13 | 复杂度
4062 |  14 | ------
4063 |  15 | Dense 回退：O(T * (W+1) * H * N^2)。在大图上请实现/切换稀疏边版本。
4064 |  16 | 
4065 |  17 | 接口（与《提示词.md》一致）
4066 |  18 | ------------------------
4067 |  19 | forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E] 或 adj_mask:Bool[N,N],
4068 |  20 |         time_idx:Long[T]) -> M:[T,N,d]
4069 |  21 | """
4070 |  22 | 
4071 |  23 | from __future__ import annotations
4072 |  24 | 
4073 |  25 | from typing import Optional, Tuple
4074 |  26 | 
4075 |  27 | import torch
4076 |  28 | import torch.nn as nn
4077 |  29 | import torch.nn.functional as F
4078 |  30 | 
4079 |  31 | from .masked_ops import masked_topk_softmax
4080 |  32 | from .rel_time import RelativeTimeEncoding
4081 |  33 | 
4082 |  34 | 
4083 |  35 | def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
4084 |  36 |     """
4085 |  37 |     将 edge_index (2,E), 其中列为 (src, dst)，转换为稠密邻接掩码 adj[i,j]：
4086 |  38 |     True 表示存在 j -> i 的边（即 dst=i, src=j）。
4087 |  39 |     """
4088 |  40 |     assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
4089 |  41 |     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
4090 |  42 |     if edge_index.numel() == 0:
4091 |  43 |         return adj
4092 |  44 |     src = edge_index[0].to(torch.long)
4093 |  45 |     dst = edge_index[1].to(torch.long)
4094 |  46 |     adj[dst, src] = True
4095 |  47 |     return adj
4096 |  48 | 
4097 |  49 | 
4098 |  50 | class SpikingTemporalAttention(nn.Module):
4099 |  51 |     def __init__(
4100 |  52 |         self,
4101 |  53 |         d_in: int,
4102 |  54 |         d: int,
4103 |  55 |         heads: int = 4,
4104 |  56 |         topk: int = 16,
4105 |  57 |         W: int = 64,
4106 |  58 |         use_rel_bias: bool = True,
4107 |  59 |         attn_drop: float = 0.1,
4108 |  60 |         temp: float = 1.0,
4109 |  61 |         # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8，符合提示词推荐）
4110 |  62 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
4111 |  63 |         pe_n_freq: int = 3,
4112 |  64 |         pe_num_buckets: int = 0,
4113 |  65 |     ) -> None:
4114 |  66 |         super().__init__()
4115 |  67 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
4116 |  68 |         assert topk >= 1, "topk 必须 >= 1"
4117 |  69 |         assert W >= 0, "W 必须 >= 0"
4118 |  70 | 
4119 |  71 |         self.d_in = int(d_in)
4120 |  72 |         self.d = int(d)
4121 |  73 |         self.heads = int(heads)
4122 |  74 |         self.d_head = self.d // self.heads
4123 |  75 |         self.topk = int(topk)
4124 |  76 |         self.W = int(W)
4125 |  77 |         self.use_rel_bias = bool(use_rel_bias)
4126 |  78 |         self.temp = float(temp)
4127 |  79 | 
4128 |  80 |         # 相对时间编码
4129 |  81 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq, num_buckets=pe_num_buckets)
4130 |  82 |         d_pe = self.rel_enc.d_pe
4131 |  83 | 
4132 |  84 |         # 线性投影（K 拼接相对时间编码）
4133 |  85 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
4134 |  86 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
4135 |  87 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
4136 |  88 | 
4137 |  89 |         self.attn_drop = nn.Dropout(attn_drop) if attn_drop and attn_drop > 0 else nn.Identity()
4138 |  90 |         self.scale = self.d_head ** -0.5
4139 |  91 | 
4140 |  92 |         self.reset_parameters()
4141 |  93 | 
4142 |  94 |     def reset_parameters(self) -> None:
4143 |  95 |         nn.init.xavier_uniform_(self.W_q.weight)
4144 |  96 |         nn.init.xavier_uniform_(self.W_k.weight)
4145 |  97 |         nn.init.xavier_uniform_(self.W_v.weight)
4146 |  98 | 
4147 |  99 |     @staticmethod
4148 | 100 |     def _build_adj_mask(
4149 | 101 |         N: int,
4150 | 102 |         edge_index: Optional[torch.Tensor],
4151 | 103 |         adj_mask: Optional[torch.Tensor],
4152 | 104 |         device: torch.device,
4153 | 105 |     ) -> torch.Tensor:
4154 | 106 |         """
4155 | 107 |         构造邻接掩码 A: Bool[N,N]，A[i,j]=True 表示 j->i 存在边（j 属于 i 的邻居）
4156 | 108 |         """
4157 | 109 |         if adj_mask is not None:
4158 | 110 |             A = adj_mask.to(device=device, dtype=torch.bool)
4159 | 111 |             assert A.shape == (N, N), f"adj_mask 形状应为 [{N},{N}]"
4160 | 112 |             return A
4161 | 113 |         assert edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 2, \
4162 | 114 |             "未提供 adj_mask 时必须提供 edge_index，形状 [2,E]"
4163 | 115 |         return _edge_index_to_dense_adj(edge_index.to(device), N, device=device)
4164 | 116 | 
4165 | 117 |     def forward(
4166 | 118 |         self,
4167 | 119 |         H_tilde: torch.Tensor,            # [T, N, d_in]
4168 | 120 |         S: torch.Tensor,                  # [T, N] in [0,1]
4169 | 121 |         edge_index: Optional[torch.Tensor],  # [2, E] 或 None
4170 | 122 |         time_idx: torch.Tensor,           # [T]
4171 | 123 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool 或 None
4172 | 124 |     ) -> torch.Tensor:
4173 | 125 |         assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
4174 | 126 |         assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
4175 | 127 |         assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"
4176 | 128 | 
4177 | 129 |         T, N, Din = H_tilde.shape
4178 | 130 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
4179 | 131 | 
4180 | 132 |         device = H_tilde.device
4181 | 133 |         dtype = H_tilde.dtype
4182 | 134 | 
4183 | 135 |         # 邻接掩码（稠密回退）
4184 | 136 |         A = self._build_adj_mask(N, edge_index, adj_mask, device)  # [N,N] Bool
4185 | 137 | 
4186 | 138 |         # 相对时间编码与（可选）偏置（仅构造 0..W）
4187 | 139 |         pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
4188 | 140 |         if not self.use_rel_bias:
4189 | 141 |             rel_bias = torch.zeros_like(rel_bias)
4190 | 142 | 
4191 | 143 |         # 预计算所有时刻的 Q、V
4192 | 144 |         Q_all = self.W_q(H_tilde)  # [T, N, d]
4193 | 145 |         V_all = self.W_v(H_tilde)  # [T, N, d]
4194 | 146 | 
4195 | 147 |         # 输出容器
4196 | 148 |         M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
4197 | 149 | 
4198 | 150 |         eps_gate = 1.0e-6
4199 | 151 | 
4200 | 152 |         for t in range(T):
4201 | 153 |             W_eff = min(self.W, t)
4202 | 154 | 
4203 | 155 |             # 多头视图
4204 | 156 |             # Q_t: [N, H, d_h] -> 转为 [H, N, d_h] 便于后续计算
4205 | 157 |             Q_t = Q_all[t].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
4206 | 158 | 
4207 | 159 |             logits_chunks = []   # 每个块 [H,N,N]
4208 | 160 |             mask_chunks = []     # 每个块 [H,N,N]
4209 | 161 |             V_chunks = []        # 每个块 [H,N,d_h]
4210 | 162 |             gate_chunks = []     # 每个块 [1,1,N]（用于 log-domain 加法；在拼接后按候选展开）
4211 | 163 | 
4212 | 164 |             for dt in range(W_eff + 1):
4213 | 165 |                 t_prime = t - dt
4214 | 166 | 
4215 | 167 |                 # 相对时间编码拼接到 K 输入
4216 | 168 |                 pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
4217 | 169 |                 pe_expand = pe.view(1, 1, -1).expand(N, -1, -1)   # [N,1,d_pe] -> 与 H_tilde[t'] 拼接
4218 | 170 |                 K_in = torch.cat([H_tilde[t_prime], pe_expand.squeeze(1)], dim=-1)  # [N, d_in + d_pe]
4219 | 171 | 
4220 | 172 |                 # 线性映射并切分多头
4221 | 173 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
4222 | 174 |                 V_tp = V_all[t_prime].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
4223 | 175 | 
4224 | 176 |                 # 注意力 logits（缩放点积）
4225 | 177 |                 # scores[h,i,j] = <Q_t[h,i,:], K_tp[h,j,:]>
4226 | 178 |                 scores = torch.einsum("hid,hjd->hij", Q_t, K_tp) * self.scale  # [H,N,N]
4227 | 179 | 
4228 | 180 |                 # 相对偏置共享到 (i,j)
4229 | 181 |                 if rel_bias is not None:
4230 | 182 |                     scores = scores + float(rel_bias[dt])
4231 | 183 | 
4232 | 184 |                 # 源端脉冲门控（log-domain 加法）
4233 | 185 |                 gate_j = torch.clamp(S[t_prime], 0.0, 1.0).to(dtype=dtype)  # [N]
4234 | 186 |                 gate_chunks.append(torch.log(gate_j + eps_gate).view(1, 1, N))  # [1,1,N]
4235 | 187 | 
4236 | 188 |                 # 邻接掩码广播到各头
4237 | 189 |                 mask_hij = A.view(1, N, N).expand(self.heads, -1, -1)  # [H,N,N]
4238 | 190 | 
4239 | 191 |                 logits_chunks.append(scores)
4240 | 192 |                 mask_chunks.append(mask_hij)
4241 | 193 |                 V_chunks.append(V_tp)
4242 | 194 | 
4243 | 195 |             if not logits_chunks:
4244 | 196 |                 # 该 t 无可用键（仅 t=0 且 W=0 时可能发生）
4245 | 197 |                 continue
4246 | 198 | 
4247 | 199 |             # 拼接候选维（按 dt 依次拼接）
4248 | 200 |             # logits_flat: [H,N,(W_eff+1)*N]
4249 | 201 |             logits_flat = torch.cat(logits_chunks, dim=2)
4250 | 202 |             mask_flat = torch.cat(mask_chunks, dim=2)  # [H,N,(W_eff+1)*N]
4251 | 203 |             gate_log_flat = torch.cat(
4252 | 204 |                 [g.expand(self.heads, N, -1) for g in gate_chunks], dim=2
4253 | 205 |             )  # [H,N,(W_eff+1)*N]
4254 | 206 |             logits_flat = logits_flat + gate_log_flat
4255 | 207 | 
4256 | 208 |             # Top-k + masked softmax（温度缩放在函数内部）
4257 | 209 |             k_eff = min(self.topk, logits_flat.size(-1))
4258 | 210 |             probs = masked_topk_softmax(
4259 | 211 |                 logits_flat, mask_flat, k=k_eff, dim=-1, temperature=self.temp
4260 | 212 |             )  # [H,N,(W_eff+1)*N]
4261 | 213 |             probs = self.attn_drop(probs)
4262 | 214 | 
4263 | 215 |             # 构造对应的值向量拼接：V_cat: [H,(W_eff+1)*N,d_h]
4264 | 216 |             V_cat = torch.cat(V_chunks, dim=1)  # [H, (W_eff+1)*N, d_h]
4265 | 217 | 
4266 | 218 |             # 聚合：msg_h = probs @ V_cat
4267 | 219 |             msg_h = torch.einsum("hni,hid->hnd", probs, V_cat)  # [H,N,d_h]
4268 | 220 | 
4269 | 221 |             # 合并头并写入输出
4270 | 222 |             M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
4271 | 223 |             M_out[t] = M_t
4272 | 224 | 
4273 | 225 |         return M_out  # [T,N,d]
4274 | ```
4275 | 
4276 | ## File: F:\SomeProjects\CSGNN\spikenet_x\sta_sparse.py
4277 | 
4278 | - Extension: .py
4279 | - Language: python
4280 | - Size: 12853 bytes
4281 | - Created: 2025-08-22 23:21:28
4282 | - Modified: 2025-08-23 00:17:47
4283 | 
4284 | ### Code
4285 | 
4286 | ```python
4287 |   1 | # -*- coding: utf-8 -*-
4288 |   2 | """
4289 |   3 | Sparse Spiking Temporal Attention（STA）——O(E) 稀疏实现（大图可用）
4290 |   4 | 
4291 |   5 | 设计要点
4292 |   6 | --------
4293 |   7 | - 不构造 [N,N] 稠密矩阵，完全基于 edge_index = (src,dst) 按边计算；
4294 |   8 | - 时间因果窗口 W：对每个 (t,i) 仅聚合 t' ∈ [t-W, t] 的消息；
4295 |   9 | - 源端脉冲门控：在 logit 上加 log(S[t',src] + eps)，等价于概率缩放；
4296 |  10 | - 两遍 segment-softmax（数值稳定）：
4297 |  11 |     Pass-1：按接收端 dst 计算各头的 segment-wise amax（log-sum-exp 的 max 项）；
4298 |  12 |     Pass-2：重新计算 exp(score - amax(dst))，用 scatter_add 聚合分母/分子得到消息；
4299 |  13 | - 相对时间编码与可学习偏置 b[Δt] 由 RelativeTimeEncoding 复用；
4300 |  14 | - 注意：本稀疏版本当前不实现 Top-K 截断（dense 版本支持），必要时后续可加入“每 dst 流式 Top-K”。
4301 |  15 | 
4302 |  16 | 复杂度
4303 |  17 | ------
4304 |  18 | O(T * H * W * E)，显存主要为按 E 规模的临时张量。适用于大图/子图批训练。
4305 |  19 | 
4306 |  20 | 接口
4307 |  21 | ----
4308 |  22 | forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E], time_idx:Long[T]) -> M:[T,N,d]
4309 |  23 | """
4310 |  24 | 
4311 |  25 | from __future__ import annotations
4312 |  26 | 
4313 |  27 | from typing import Optional, Tuple
4314 |  28 | 
4315 |  29 | import math
4316 |  30 | import torch
4317 |  31 | import torch.nn as nn
4318 |  32 | import torch.nn.functional as F
4319 |  33 | 
4320 |  34 | from .rel_time import RelativeTimeEncoding
4321 |  35 | 
4322 |  36 | 
4323 |  37 | def _has_scatter_reduce_tensor() -> bool:
4324 |  38 |     # PyTorch 1.12+ 提供 Tensor.scatter_reduce_
4325 |  39 |     return hasattr(torch.Tensor, "scatter_reduce_")
4326 |  40 | 
4327 |  41 | 
4328 |  42 | def _try_import_torch_scatter():
4329 |  43 |     try:
4330 |  44 |         import torch_scatter  # type: ignore
4331 |  45 |         return torch_scatter
4332 |  46 |     except Exception:
4333 |  47 |         return None
4334 |  48 | 
4335 |  49 | 
4336 |  50 | _TORCH_SCATTER = _try_import_torch_scatter()
4337 |  51 | _HAS_TSR = _has_scatter_reduce_tensor()
4338 |  52 | 
4339 |  53 | 
4340 |  54 | def _segment_amax_1d(x: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
4341 |  55 |     """
4342 |  56 |     计算 out[j] = max_{i: index[i]==j} x[i]，其中 j ∈ [0, K)
4343 |  57 |     优先使用 Tensor.scatter_reduce_('amax')；其次 torch_scatter.scatter_max；最终回退到排序段法。
4344 |  58 |     """
4345 |  59 |     device, dtype = x.device, x.dtype
4346 |  60 |     neg_inf = torch.tensor(-1e30, dtype=dtype, device=device)
4347 |  61 | 
4348 |  62 |     if _HAS_TSR:
4349 |  63 |         # 使用非 in-place 版本 scatter_reduce 修复梯度计算问题
4350 |  64 |         # include_self=False 时，空段结果未定义，需手动填充
4351 |  65 |         init_val = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
4352 |  66 |         out = init_val.scatter_reduce(0, index, x, reduce="amax", include_self=False)
4353 |  67 |         return out
4354 |  68 | 
4355 |  69 |     if _TORCH_SCATTER is not None:
4356 |  70 |         # torch_scatter.scatter_max 返回 (out, argmax)
4357 |  71 |         out, _ = _TORCH_SCATTER.scatter_max(x, index, dim=0, dim_size=K)
4358 |  72 |         # 对于空段，scatter_max 会给出 0；为一致性，将空段填为 -inf：
4359 |  73 |         # 通过统计计数判断空段
4360 |  74 |         cnt = torch.zeros(K, device=device, dtype=torch.long)
4361 |  75 |         cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
4362 |  76 |         out = torch.where(cnt > 0, out, neg_inf)
4363 |  77 |         return out
4364 |  78 | 
4365 |  79 |     # 回退：排序段法（可能较慢，但不依赖扩展）
4366 |  80 |     perm = torch.argsort(index)
4367 |  81 |     idx_s = index[perm]
4368 |  82 |     x_s = x[perm]
4369 |  83 |     out = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
4370 |  84 |     if idx_s.numel() == 0:
4371 |  85 |         return out
4372 |  86 |     # 找段边界
4373 |  87 |     boundary = torch.ones_like(idx_s, dtype=torch.bool)
4374 |  88 |     boundary[1:] = idx_s[1:] != idx_s[:-1]
4375 |  89 |     # 段起点位置
4376 |  90 |     starts = torch.nonzero(boundary, as_tuple=False).flatten()
4377 |  91 |     # 段终点（含）位置
4378 |  92 |     ends = torch.empty_like(starts)
4379 |  93 |     ends[:-1] = starts[1:] - 1
4380 |  94 |     ends[-1] = idx_s.numel() - 1
4381 |  95 |     # 逐段计算最大（Python 循环，仅在无高阶算子时作为兜底）
4382 |  96 |     for s, e in zip(starts.tolist(), ends.tolist()):
4383 |  97 |         j = int(idx_s[s].item())
4384 |  98 |         out[j] = torch.maximum(out[j], x_s[s : e + 1].max())
4385 |  99 |     return out
4386 | 100 | 
4387 | 101 | 
4388 | 102 | class SparseSpikingTemporalAttention(nn.Module):
4389 | 103 |     def __init__(
4390 | 104 |         self,
4391 | 105 |         d_in: int,
4392 | 106 |         d: int,
4393 | 107 |         heads: int = 4,
4394 | 108 |         W: int = 64,
4395 | 109 |         use_rel_bias: bool = True,
4396 | 110 |         attn_drop: float = 0.1,
4397 | 111 |         temp: float = 1.0,
4398 | 112 |         # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8）
4399 | 113 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
4400 | 114 |         pe_n_freq: int = 3,
4401 | 115 |         pe_num_buckets: int = 0,
4402 | 116 |     ) -> None:
4403 | 117 |         """
4404 | 118 |         稀疏 STA，不支持 Top-K 截断（如需可后续加入流式 Top-K）。
4405 | 119 |         """
4406 | 120 |         super().__init__()
4407 | 121 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
4408 | 122 |         assert W >= 0, "W 必须 >= 0"
4409 | 123 | 
4410 | 124 |         self.d_in = int(d_in)
4411 | 125 |         self.d = int(d)
4412 | 126 |         self.heads = int(heads)
4413 | 127 |         self.d_head = self.d // self.heads
4414 | 128 |         self.W = int(W)
4415 | 129 |         self.use_rel_bias = bool(use_rel_bias)
4416 | 130 |         self.temp = float(temp)
4417 | 131 | 
4418 | 132 |         # 相对时间编码
4419 | 133 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq, num_buckets=pe_num_buckets)
4420 | 134 |         d_pe = self.rel_enc.d_pe
4421 | 135 | 
4422 | 136 |         # 线性投影（K 拼接相对时间编码）
4423 | 137 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
4424 | 138 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
4425 | 139 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
4426 | 140 | 
4427 | 141 |         # 注意力 dropout：实现为“边贡献丢弃再归一化”的无缩放 Bernoulli mask（训练时）
4428 | 142 |         self.p_drop = float(attn_drop)
4429 | 143 |         self.scale = self.d_head ** -0.5
4430 | 144 | 
4431 | 145 |         self.reset_parameters()
4432 | 146 | 
4433 | 147 |     def reset_parameters(self) -> None:
4434 | 148 |         nn.init.xavier_uniform_(self.W_q.weight)
4435 | 149 |         nn.init.xavier_uniform_(self.W_k.weight)
4436 | 150 |         nn.init.xavier_uniform_(self.W_v.weight)
4437 | 151 | 
4438 | 152 |     @staticmethod
4439 | 153 |     def _check_edges(edge_index: torch.Tensor, N: int) -> None:
4440 | 154 |         assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
4441 | 155 |         E = edge_index.size(1)
4442 | 156 |         if E == 0:
4443 | 157 |             return
4444 | 158 |         assert int(edge_index.min()) >= 0 and int(edge_index.max()) < N, "edge_index 越界"
4445 | 159 | 
4446 | 160 |     def _edge_arrays(self, edge_index: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
4447 | 161 |         # 拆分 src, dst 为长度 E 的向量
4448 | 162 |         src = edge_index[0].to(device=device, dtype=torch.long)
4449 | 163 |         dst = edge_index[1].to(device=device, dtype=torch.long)
4450 | 164 |         return src, dst
4451 | 165 | 
4452 | 166 |     @torch.no_grad()
4453 | 167 |     def _drop_mask(self, shape: torch.Size, device: torch.device) -> Optional[torch.Tensor]:
4454 | 168 |         if not self.training or self.p_drop <= 0.0:
4455 | 169 |             return None
4456 | 170 |         return (torch.rand(shape, device=device) > self.p_drop)
4457 | 171 | 
4458 | 172 |     def forward(
4459 | 173 |         self,
4460 | 174 |         H_tilde: torch.Tensor,            # [T, N, d_in]
4461 | 175 |         S: torch.Tensor,                  # [T, N] in [0,1]
4462 | 176 |         edge_index: torch.Tensor,         # [2, E]
4463 | 177 |         time_idx: torch.Tensor,           # [T]
4464 | 178 |         adj_mask: Optional[torch.Tensor] = None,  # 兼容签名；稀疏实现忽略
4465 | 179 |     ) -> torch.Tensor:
4466 | 180 |         assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
4467 | 181 |         assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
4468 | 182 |         assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"
4469 | 183 | 
4470 | 184 |         T, N, Din = H_tilde.shape
4471 | 185 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
4472 | 186 | 
4473 | 187 |         device = H_tilde.device
4474 | 188 |         dtype = H_tilde.dtype
4475 | 189 | 
4476 | 190 |         # 校验边并拆分
4477 | 191 |         self._check_edges(edge_index, N)
4478 | 192 |         src, dst = self._edge_arrays(edge_index, device)  # [E], [E]
4479 | 193 |         E = src.numel()
4480 | 194 | 
4481 | 195 |         # 相对时间编码与偏置（仅构造 0..W）
4482 | 196 |         pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
4483 | 197 |         if not self.use_rel_bias:
4484 | 198 |             rel_bias = torch.zeros_like(rel_bias)
4485 | 199 | 
4486 | 200 |         # 预计算 Q(t, ·) 与 V(t, ·)
4487 | 201 |         Q_all = self.W_q(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()  # [T,H,N,d_h]
4488 | 202 |         V_all = self.W_v(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()  # [T,H,N,d_h]
4489 | 203 | 
4490 | 204 |         # 输出
4491 | 205 |         M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
4492 | 206 | 
4493 | 207 |         eps_gate = 1.0e-6
4494 | 208 |         neg_inf = -1.0e30
4495 | 209 | 
4496 | 210 |         for t in range(T):
4497 | 211 |             W_eff = min(self.W, t)
4498 | 212 |             # Q_t: [H,N,d_h]
4499 | 213 |             Q_t = Q_all[t]  # [H,N,d_h]
4500 | 214 | 
4501 | 215 |             # -------- Pass-1：按 dst 计算 segment-wise amax（各头独立） --------
4502 | 216 |             max_dst_list = []
4503 | 217 |             for dt in range(W_eff + 1):
4504 | 218 |                 t_prime = t - dt
4505 | 219 |                 # 构造 K_{t'}（拼接相对时间编码）
4506 | 220 |                 pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
4507 | 221 |                 K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)  # [N, d_in+d_pe]
4508 | 222 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
4509 | 223 | 
4510 | 224 |                 # 门控（源端脉冲）
4511 | 225 |                 gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)  # [N]
4512 | 226 | 
4513 | 227 |                 # 对每个头计算边打分：scores[h,e] = <Q_t[h,dst[e]], K_tp[h,src[e]]> * scale + b[dt] + log S
4514 | 228 |                 # gather Q/K
4515 | 229 |                 Q_d = Q_t[:, dst, :]              # [H,E,d_h]
4516 | 230 |                 K_s = K_tp[:, src, :]             # [H,E,d_h]
4517 | 231 |                 # 点积
4518 | 232 |                 scores = (Q_d * K_s).sum(dim=-1) * self.scale  # [H,E]
4519 | 233 |                 # 相对偏置
4520 | 234 |                 scores = scores + float(rel_bias[dt])
4521 | 235 |                 # 源脉冲门控
4522 | 236 |                 scores = scores + gate_log[src]  # 广播到 [H,E]
4523 | 237 | 
4524 | 238 |                 # softmax 温度
4525 | 239 |                 if self.temp != 1.0:
4526 | 240 |                     scores = scores / float(self.temp)
4527 | 241 | 
4528 | 242 |                 # 对每头做 segment amax
4529 | 243 |                 m_h = torch.stack([_segment_amax_1d(scores[h], dst, N) for h in range(self.heads)])
4530 | 244 |                 max_dst_list.append(m_h)
4531 | 245 | 
4532 | 246 |             max_dst = torch.stack(max_dst_list, dim=0).max(dim=0)[0]
4533 | 247 | 
4534 | 248 |             # -------- Pass-2：exp(score - amax(dst)) 聚合分母/分子 --------
4535 | 249 |             denom = torch.zeros((self.heads, N), device=device, dtype=dtype)          # [H,N]
4536 | 250 |             numer = torch.zeros((self.heads, N, self.d_head), device=device, dtype=dtype)  # [H,N,d_h]
4537 | 251 | 
4538 | 252 |             for dt in range(W_eff + 1):
4539 | 253 |                 t_prime = t - dt
4540 | 254 |                 pe = pe_table[dt].to(dtype=dtype, device=device)
4541 | 255 |                 K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)  # [N, d_in+d_pe]
4542 | 256 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
4543 | 257 |                 V_tp = V_all[t_prime]  # [H,N,d_h]
4544 | 258 |                 gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)  # [N]
4545 | 259 | 
4546 | 260 |                 Q_d = Q_t[:, dst, :]          # [H,E,d_h]
4547 | 261 |                 K_s = K_tp[:, src, :]         # [H,E,d_h]
4548 | 262 |                 V_s = V_tp[:, src, :]         # [H,E,d_h]
4549 | 263 | 
4550 | 264 |                 scores = (Q_d * K_s).sum(dim=-1) * self.scale  # [H,E]
4551 | 265 |                 scores = scores + float(rel_bias[dt])
4552 | 266 |                 scores = scores + gate_log[src]
4553 | 267 |                 if self.temp != 1.0:
4554 | 268 |                     scores = scores / float(self.temp)
4555 | 269 | 
4556 | 270 |                 # exp(score - max_dst[dst])
4557 | 271 |                 # broadcast: max_dst[:, dst] -> [H,E]
4558 | 272 |                 max_g = max_dst[:, dst]
4559 | 273 |                 ex = torch.exp(scores - max_g)  # [H,E]
4560 | 274 | 
4561 | 275 |                 # attention dropout：训练时对边贡献做伯努利丢弃（不做 1/(1-p) 缩放，随后自动归一化）
4562 | 276 |                 mask = self._drop_mask(ex.shape, device=device)
4563 | 277 |                 if mask is not None:
4564 | 278 |                     ex = ex * mask.to(dtype=ex.dtype)
4565 | 279 | 
4566 | 280 |                 # 逐头 scatter_add 到 dst
4567 | 281 |                 for h in range(self.heads):
4568 | 282 |                     # 分母
4569 | 283 |                     denom[h].index_add_(0, dst, ex[h])  # [N]
4570 | 284 |                     # 分子：ex[h][:,None] * V_s[h] 累加到 dst
4571 | 285 |                     contrib = ex[h].unsqueeze(-1) * V_s[h]  # [E,d_h]
4572 | 286 |                     # 将 [E,d_h] 累加到 [N,d_h]：循环通道（d_h 小，循环成本可接受）
4573 | 287 |                     # 向量化 index_add_ 仅支持 1D，这里按通道展开
4574 | 288 |                     for c in range(self.d_head):
4575 | 289 |                         numer[h, :, c].index_add_(0, dst, contrib[:, c])
4576 | 290 | 
4577 | 291 |             # 得到各头消息并合并
4578 | 292 |             # 防零保护
4579 | 293 |             denom = torch.clamp(denom, min=1e-12)
4580 | 294 |             msg_h = numer / denom.unsqueeze(-1)  # [H,N,d_h]
4581 | 295 |             M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
4582 | 296 |             M_out[t] = M_t
4583 | 297 | 
4584 | 298 |         return M_out  # [T,N,d]
4585 | 299 | 
4586 | 300 | 
4587 | 301 | __all__ = ["SparseSpikingTemporalAttention"]
4588 | ```
4589 | 
4590 | ## File: F:\SomeProjects\CSGNN\spikenet_x\__init__.py
4591 | 
4592 | - Extension: .py
4593 | - Language: python
4594 | - Size: 973 bytes
4595 | - Created: 2025-08-22 13:06:01
4596 | - Modified: 2025-08-22 23:27:05
4597 | 
4598 | ### Code
4599 | 
4600 | ```python
4601 |  1 | # -*- coding: utf-8 -*-
4602 |  2 | """
4603 |  3 | SpikeNet-X package
4604 |  4 | 
4605 |  5 | Exports the core building blocks specified in `提示词.md`:
4606 |  6 | - LearnableDelayLine
4607 |  7 | - SpikingTemporalAttention (dense fallback implementation)
4608 |  8 | - LIFCell
4609 |  9 | - SpikeNetXLayer
4610 | 10 | - Masked ops helpers and RelativeTimeEncoding
4611 | 11 | """
4612 | 12 | 
4613 | 13 | from .masked_ops import (
4614 | 14 |     masked_softmax,
4615 | 15 |     masked_topk_softmax,
4616 | 16 |     topk_mask_logits,
4617 | 17 |     fill_masked_,
4618 | 18 |     NEG_INF,
4619 | 19 | )
4620 | 20 | from .rel_time import RelativeTimeEncoding
4621 | 21 | from .delayline import LearnableDelayLine
4622 | 22 | from .sta import SpikingTemporalAttention
4623 | 23 | from .sta_sparse import SparseSpikingTemporalAttention
4624 | 24 | from .lif_cell import LIFCell
4625 | 25 | from .spikenetx_layer import SpikeNetXLayer
4626 | 26 | 
4627 | 27 | __all__ = [
4628 | 28 |     "masked_softmax",
4629 | 29 |     "masked_topk_softmax",
4630 | 30 |     "topk_mask_logits",
4631 | 31 |     "fill_masked_",
4632 | 32 |     "NEG_INF",
4633 | 33 |     "RelativeTimeEncoding",
4634 | 34 |     "LearnableDelayLine",
4635 | 35 |     "SpikingTemporalAttention",
4636 | 36 |     "SparseSpikingTemporalAttention",
4637 | 37 |     "LIFCell",
4638 | 38 |     "SpikeNetXLayer",
4639 | 39 | ]
4640 | ```
4641 | 
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
- Size: 19403 bytes
- Created: 2025-08-21 17:29:04
- Modified: 2025-09-14 18:34:10

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
 20 |     以一批种子节点 nodes 抽 1-hop 子图，并返回：
 21 |       - subgraph_nodes: 子图包含的全局节点 id，形状 [N_sub]
 22 |       - subgraph_edge_index: 子图边(局部id)，形状 [2, E_sub]
 23 |       - nodes_local_index: 种子在子图里的局部索引，形状 [B]
 24 |     关键改动：保留子图内部的“所有边”（src/dst 都在子图内），解锁多层传播。
 25 |     """
 26 |     row, col = edge_index_full
 27 |     device = row.device
 28 |     nodes = nodes.to(device)
 29 | 
 30 |     # 1) 先收集邻居（全收或限量）
 31 |     if num_neighbors == -1:
 32 |         mask = torch.isin(row, nodes)
 33 |         neighbors = col[mask]
 34 |     else:
 35 |         # 简洁做法：对所有与种子相连的邻居做全局采样，期望规模 ≈ B * num_neighbors
 36 |         mask = torch.isin(row, nodes)
 37 |         neighbors_all = col[mask]
 38 |         target = nodes.numel() * int(num_neighbors)
 39 |         if neighbors_all.numel() > target > 0:
 40 |             perm = torch.randperm(neighbors_all.numel(), device=device)[:target]
 41 |             neighbors = neighbors_all[perm]
 42 |         else:
 43 |             neighbors = neighbors_all
 44 | 
 45 |     # 2) 子图节点集合：种子 ∪ 采样邻居
 46 |     subgraph_nodes = torch.unique(torch.cat([nodes, neighbors], dim=0))
 47 | 
 48 |     # 3) **关键修复**：仅保留子图内部的边（src/dst 都在 subgraph_nodes）
 49 |     mask_src = torch.isin(row, subgraph_nodes)
 50 |     mask_dst = torch.isin(col, subgraph_nodes)
 51 |     edge_mask = mask_src & mask_dst
 52 |     subgraph_edge_index_global = edge_index_full[:, edge_mask]  # 仍是“全局 id”
 53 | 
 54 |     # 4) 将全局 id 映射到局部 id（纯 Torch，避免 Python 循环/字典）
 55 |     subgraph_nodes_sorted, _ = torch.sort(subgraph_nodes)  # searchsorted 需要有序
 56 |     src_global = subgraph_edge_index_global[0]
 57 |     dst_global = subgraph_edge_index_global[1]
 58 |     src_local = torch.searchsorted(subgraph_nodes_sorted, src_global)
 59 |     dst_local = torch.searchsorted(subgraph_nodes_sorted, dst_global)
 60 |     subgraph_edge_index = torch.stack([src_local, dst_local], dim=0)
 61 | 
 62 |     # 5) 种子节点的局部索引
 63 |     nodes_local_index = torch.searchsorted(subgraph_nodes_sorted, nodes)
 64 | 
 65 |     return subgraph_nodes_sorted, subgraph_edge_index, nodes_local_index
 66 | 
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
180 | parser.add_argument("--model", nargs="?", default="spikenet",
181 |                     help="Model to use ('spikenet', 'spikenetx'). (default: spikenet)")
182 | parser.add_argument("--dataset", nargs="?", default="DBLP",
183 |                     help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
184 | parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
185 | parser.add_argument('--hids', type=int, nargs='+',
186 |                     default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
187 | parser.add_argument("--aggr", nargs="?", default="mean",
188 |                     help="Aggregate function ('mean', 'sum'). (default: 'mean')")
189 | parser.add_argument("--sampler", nargs="?", default="sage",
190 |                     help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
191 | parser.add_argument("--surrogate", nargs="?", default="sigmoid",
192 |                     help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
193 | parser.add_argument("--neuron", nargs="?", default="LIF",
194 |                     help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
195 | parser.add_argument('--batch_size', type=int, default=1024,
196 |                     help='Batch size for training. (default: 1024)')
197 | parser.add_argument('--lr', type=float, default=5e-3,
198 |                     help='Learning rate for training. (default: 5e-3)')
199 | parser.add_argument('--train_size', type=float, default=0.4,
200 |                     help='Ratio of nodes for training. (default: 0.4)')
201 | parser.add_argument('--alpha', type=float, default=1.0,
202 |                     help='Smooth factor for surrogate learning. (default: 1.0)')
203 | parser.add_argument('--p', type=float, default=0.5,
204 |                     help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
205 | parser.add_argument('--dropout', type=float, default=0.7,
206 |                     help='Dropout probability. (default: 0.7)')
207 | parser.add_argument('--epochs', type=int, default=100,
208 |                     help='Number of training epochs. (default: 100)')
209 | parser.add_argument('--concat', action='store_true',
210 |                     help='Whether to concat node representation and neighborhood representations. (default: False)')
211 | parser.add_argument('--seed', type=int, default=2022,
212 |                     help='Random seed for model. (default: 2022)')
213 | parser.add_argument('--datapath', type=str, default='./data',
214 |                     help='Wheres your data?, Default is ./data')
215 | 
216 | # SpikeNet-X specific args
217 | parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for SpikeNet-X. (default: 4)')
218 | parser.add_argument('--topk', type=int, default=8, help='Top-k neighbors for SpikeNet-X attention. (default: 8)')
219 | parser.add_argument('--W', type=int, default=8, help='Time window size for SpikeNet-X. (default: 8)')
220 | parser.add_argument('--attn_impl', type=str, default='dense', choices=['dense','sparse'],
221 |                     help='Attention kernel for SpikeNet-X: "dense" (fallback, supports top-k) or "sparse". (default: "dense")')
222 | 
223 | 
224 | # 新增：模型保存、加载与测试参数
225 | parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
226 |                     help='Directory to save model checkpoints. (default: checkpoints)')
227 | parser.add_argument('--resume_path', type=str, default=None,
228 |                     help='Path to a checkpoint file to resume training from. (default: None)')
229 | parser.add_argument('--test_model_path', type=str, default=None,
230 |                     help='Path to a model file to load for testing only. (default: None)')
231 | 
232 | 
233 | try:
234 |     args = parser.parse_args()
235 |     args.test_size = 1 - args.train_size
236 |     args.train_size = args.train_size - 0.05
237 |     args.val_size = 0.05
238 |     args.split_seed = 42
239 |     tab_printer(args)
240 | except:
241 |     parser.print_help()
242 |     exit(0)
243 | 
244 | assert len(args.hids) == len(args.sizes), "must be equal!"
245 | 
246 | if args.dataset.lower() == "dblp":
247 |     data = dataset.DBLP(root = args.datapath)
248 | elif args.dataset.lower() == "tmall":
249 |     data = dataset.Tmall(root = args.datapath)
250 | elif args.dataset.lower() == "patent":
251 |     data = dataset.Patent(root = args.datapath)
252 | else:
253 |     raise ValueError(
254 |         f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")
255 | 
256 | # train:val:test
257 | data.split_nodes(train_size=args.train_size, val_size=args.val_size,
258 |                  test_size=args.test_size, random_state=args.split_seed)
259 | 
260 | set_seed(args.seed)
261 | 
262 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
263 | 
264 | y = data.y.to(device)
265 | 
266 | train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
267 | val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
268 |                         pin_memory=False, batch_size=200000, shuffle=False)
269 | test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
270 | 
271 | if args.model == 'spikenetx':
272 |     
273 |     def train_spikenetx():
274 |         model.train()
275 |         total_loss = 0
276 |         # Let's use a fixed number of neighbors for now to control memory
277 |         # 现为25， -1 为全邻居
278 |         num_neighbors_to_sample = 25 
279 |         for nodes in tqdm(train_loader, desc='Training'):
280 |             nodes = nodes.to(device)
281 |             subgraph_nodes, subgraph_edge_index, nodes_local_index = sample_subgraph(nodes, edge_index_full, num_neighbors=num_neighbors_to_sample)
282 |             
283 |             H0_subgraph = H0_full[:, subgraph_nodes, :]
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
- Size: 4560 bytes
- Created: 2025-08-22 12:50:15
- Modified: 2025-08-22 12:50:55

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
 92 | def topk_mask_logits(
 93 |     logits: torch.Tensor,
 94 |     k: int,
 95 |     dim: int = -1,
 96 |     inplace: bool = False,
 97 | ) -> Tuple[torch.Tensor, torch.Tensor]:
 98 |     """
 99 |     在维度 dim 上选出前 k 的元素，其余位置置为 -inf（或近似值）。
100 | 
101 |     注意：
102 |     - 该函数只在 logits 上执行 Top-k 筛选，不做 softmax。
103 |     - 返回 (new_logits, keep_mask)
104 | 
105 |     参数
106 |     ----
107 |     logits : Float[...]
108 |     k : int
109 |         k >= 1
110 |     dim : int
111 |     inplace : bool
112 |         是否原地写回
113 | 
114 |     返回
115 |     ----
116 |     new_logits : Float[...]
117 |         仅保留 Top-k 的 logits；其余位置为 -inf
118 |     keep_mask : Bool[...]
119 |         True 表示该位置被保留
120 |     """
121 |     assert k >= 1, "topk must be >= 1"
122 |     # 取 Top-k 的阈值
123 |     topk_vals, topk_idx = torch.topk(logits, k=k, dim=dim)
124 |     # 构造保留 mask
125 |     keep_mask = torch.zeros_like(logits, dtype=torch.bool)
126 |     keep_mask.scatter_(dim, topk_idx, True)
127 | 
128 |     if inplace:
129 |         out = fill_masked_(logits, keep_mask, NEG_INF)
130 |         return out, keep_mask
131 |     else:
132 |         new_logits = torch.where(keep_mask, logits, torch.full_like(logits, NEG_INF))
133 |         return new_logits, keep_mask
134 | 
135 | 
136 | def masked_topk_softmax(
137 |     logits: torch.Tensor,
138 |     mask: Optional[torch.Tensor],
139 |     k: int,
140 |     dim: int = -1,
141 |     temperature: float = 1.0,
142 | ) -> torch.Tensor:
143 |     """
144 |     组合操作：先对 logits 进行掩码，随后 Top-k 截断，再做 softmax。
145 | 
146 |     等价步骤：
147 |       1) logits[~mask] = -inf
148 |       2) 仅保留维度 dim 上的 Top-k，其余 = -inf
149 |       3) softmax(dim)
150 | 
151 |     参数
152 |     ----
153 |     logits : Float[...]
154 |     mask : Optional[Bool/Byte[...] ]
155 |     k : int
156 |     dim : int
157 |     temperature : float
158 | 
159 |     返回
160 |     ----
161 |     probs : Float[...]
162 |     """
163 |     if mask is not None:
164 |         logits = logits.clone()
165 |         fill_masked_(logits, mask, NEG_INF)
166 |     logits, _ = topk_mask_logits(logits, k=k, dim=dim, inplace=False)
167 |     return masked_softmax(logits, mask=None, dim=dim, temperature=temperature)
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
- Size: 9234 bytes
- Created: 2025-08-22 12:59:28
- Modified: 2025-08-22 16:32:53

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
 36 |     """
 37 |     将 edge_index (2,E), 其中列为 (src, dst)，转换为稠密邻接掩码 adj[i,j]：
 38 |     True 表示存在 j -> i 的边（即 dst=i, src=j）。
 39 |     """
 40 |     assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
 41 |     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
 42 |     if edge_index.numel() == 0:
 43 |         return adj
 44 |     src = edge_index[0].to(torch.long)
 45 |     dst = edge_index[1].to(torch.long)
 46 |     adj[dst, src] = True
 47 |     return adj
 48 | 
 49 | 
 50 | class SpikingTemporalAttention(nn.Module):
 51 |     def __init__(
 52 |         self,
 53 |         d_in: int,
 54 |         d: int,
 55 |         heads: int = 4,
 56 |         topk: int = 16,
 57 |         W: int = 64,
 58 |         use_rel_bias: bool = True,
 59 |         attn_drop: float = 0.1,
 60 |         temp: float = 1.0,
 61 |         # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8，符合提示词推荐）
 62 |         pe_taus: Tuple[float, float] = (4.0, 16.0),
 63 |         pe_n_freq: int = 3,
 64 |         pe_num_buckets: int = 0,
 65 |     ) -> None:
 66 |         super().__init__()
 67 |         assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
 68 |         assert topk >= 1, "topk 必须 >= 1"
 69 |         assert W >= 0, "W 必须 >= 0"
 70 | 
 71 |         self.d_in = int(d_in)
 72 |         self.d = int(d)
 73 |         self.heads = int(heads)
 74 |         self.d_head = self.d // self.heads
 75 |         self.topk = int(topk)
 76 |         self.W = int(W)
 77 |         self.use_rel_bias = bool(use_rel_bias)
 78 |         self.temp = float(temp)
 79 | 
 80 |         # 相对时间编码
 81 |         self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq, num_buckets=pe_num_buckets)
 82 |         d_pe = self.rel_enc.d_pe
 83 | 
 84 |         # 线性投影（K 拼接相对时间编码）
 85 |         self.W_q = nn.Linear(d_in, self.d, bias=False)
 86 |         self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
 87 |         self.W_v = nn.Linear(d_in, self.d, bias=False)
 88 | 
 89 |         self.attn_drop = nn.Dropout(attn_drop) if attn_drop and attn_drop > 0 else nn.Identity()
 90 |         self.scale = self.d_head ** -0.5
 91 | 
 92 |         self.reset_parameters()
 93 | 
 94 |     def reset_parameters(self) -> None:
 95 |         nn.init.xavier_uniform_(self.W_q.weight)
 96 |         nn.init.xavier_uniform_(self.W_k.weight)
 97 |         nn.init.xavier_uniform_(self.W_v.weight)
 98 | 
 99 |     @staticmethod
100 |     def _build_adj_mask(
101 |         N: int,
102 |         edge_index: Optional[torch.Tensor],
103 |         adj_mask: Optional[torch.Tensor],
104 |         device: torch.device,
105 |     ) -> torch.Tensor:
106 |         """
107 |         构造邻接掩码 A: Bool[N,N]，A[i,j]=True 表示 j->i 存在边（j 属于 i 的邻居）
108 |         """
109 |         if adj_mask is not None:
110 |             A = adj_mask.to(device=device, dtype=torch.bool)
111 |             assert A.shape == (N, N), f"adj_mask 形状应为 [{N},{N}]"
112 |             return A
113 |         assert edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 2, \
114 |             "未提供 adj_mask 时必须提供 edge_index，形状 [2,E]"
115 |         return _edge_index_to_dense_adj(edge_index.to(device), N, device=device)
116 | 
117 |     def forward(
118 |         self,
119 |         H_tilde: torch.Tensor,            # [T, N, d_in]
120 |         S: torch.Tensor,                  # [T, N] in [0,1]
121 |         edge_index: Optional[torch.Tensor],  # [2, E] 或 None
122 |         time_idx: torch.Tensor,           # [T]
123 |         adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool 或 None
124 |     ) -> torch.Tensor:
125 |         assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
126 |         assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
127 |         assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"
128 | 
129 |         T, N, Din = H_tilde.shape
130 |         assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"
131 | 
132 |         device = H_tilde.device
133 |         dtype = H_tilde.dtype
134 | 
135 |         # 邻接掩码（稠密回退）
136 |         A = self._build_adj_mask(N, edge_index, adj_mask, device)  # [N,N] Bool
137 | 
138 |         # 相对时间编码与（可选）偏置（仅构造 0..W）
139 |         pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
140 |         if not self.use_rel_bias:
141 |             rel_bias = torch.zeros_like(rel_bias)
142 | 
143 |         # 预计算所有时刻的 Q、V
144 |         Q_all = self.W_q(H_tilde)  # [T, N, d]
145 |         V_all = self.W_v(H_tilde)  # [T, N, d]
146 | 
147 |         # 输出容器
148 |         M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
149 | 
150 |         eps_gate = 1.0e-6
151 | 
152 |         for t in range(T):
153 |             W_eff = min(self.W, t)
154 | 
155 |             # 多头视图
156 |             # Q_t: [N, H, d_h] -> 转为 [H, N, d_h] 便于后续计算
157 |             Q_t = Q_all[t].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
158 | 
159 |             logits_chunks = []   # 每个块 [H,N,N]
160 |             mask_chunks = []     # 每个块 [H,N,N]
161 |             V_chunks = []        # 每个块 [H,N,d_h]
162 |             gate_chunks = []     # 每个块 [1,1,N]（用于 log-domain 加法；在拼接后按候选展开）
163 | 
164 |             for dt in range(W_eff + 1):
165 |                 t_prime = t - dt
166 | 
167 |                 # 相对时间编码拼接到 K 输入
168 |                 pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
169 |                 pe_expand = pe.view(1, 1, -1).expand(N, -1, -1)   # [N,1,d_pe] -> 与 H_tilde[t'] 拼接
170 |                 K_in = torch.cat([H_tilde[t_prime], pe_expand.squeeze(1)], dim=-1)  # [N, d_in + d_pe]
171 | 
172 |                 # 线性映射并切分多头
173 |                 K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
174 |                 V_tp = V_all[t_prime].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
175 | 
176 |                 # 注意力 logits（缩放点积）
177 |                 # scores[h,i,j] = <Q_t[h,i,:], K_tp[h,j,:]>
178 |                 scores = torch.einsum("hid,hjd->hij", Q_t, K_tp) * self.scale  # [H,N,N]
179 | 
180 |                 # 相对偏置共享到 (i,j)
181 |                 if rel_bias is not None:
182 |                     scores = scores + float(rel_bias[dt])
183 | 
184 |                 # 源端脉冲门控（log-domain 加法）
185 |                 gate_j = torch.clamp(S[t_prime], 0.0, 1.0).to(dtype=dtype)  # [N]
186 |                 gate_chunks.append(torch.log(gate_j + eps_gate).view(1, 1, N))  # [1,1,N]
187 | 
188 |                 # 邻接掩码广播到各头
189 |                 mask_hij = A.view(1, N, N).expand(self.heads, -1, -1)  # [H,N,N]
190 | 
191 |                 logits_chunks.append(scores)
192 |                 mask_chunks.append(mask_hij)
193 |                 V_chunks.append(V_tp)
194 | 
195 |             if not logits_chunks:
196 |                 # 该 t 无可用键（仅 t=0 且 W=0 时可能发生）
197 |                 continue
198 | 
199 |             # 拼接候选维（按 dt 依次拼接）
200 |             # logits_flat: [H,N,(W_eff+1)*N]
201 |             logits_flat = torch.cat(logits_chunks, dim=2)
202 |             mask_flat = torch.cat(mask_chunks, dim=2)  # [H,N,(W_eff+1)*N]
203 |             gate_log_flat = torch.cat(
204 |                 [g.expand(self.heads, N, -1) for g in gate_chunks], dim=2
205 |             )  # [H,N,(W_eff+1)*N]
206 |             logits_flat = logits_flat + gate_log_flat
207 | 
208 |             # Top-k + masked softmax（温度缩放在函数内部）
209 |             k_eff = min(self.topk, logits_flat.size(-1))
210 |             probs = masked_topk_softmax(
211 |                 logits_flat, mask_flat, k=k_eff, dim=-1, temperature=self.temp
212 |             )  # [H,N,(W_eff+1)*N]
213 |             probs = self.attn_drop(probs)
214 | 
215 |             # 构造对应的值向量拼接：V_cat: [H,(W_eff+1)*N,d_h]
216 |             V_cat = torch.cat(V_chunks, dim=1)  # [H, (W_eff+1)*N, d_h]
217 | 
218 |             # 聚合：msg_h = probs @ V_cat
219 |             msg_h = torch.einsum("hni,hid->hnd", probs, V_cat)  # [H,N,d_h]
220 | 
221 |             # 合并头并写入输出
222 |             M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
223 |             M_out[t] = M_t
224 | 
225 |         return M_out  # [T,N,d]
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


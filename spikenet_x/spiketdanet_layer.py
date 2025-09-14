import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .new_modules import SpatialGNNWrapper, DelayLine, STAGNNAggregator
from .lif_cell import LIFCell

class MLP(nn.Module):
    def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d * hidden_mult)
        self.fc2 = nn.Linear(d * hidden_mult, d)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class SpikeTDANetLayer(nn.Module):
    def __init__(self, channels: int, heads: int, W: int, delay_kernel: int = 5, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 定义LIFCell专属的参数键
        lif_keys = ['lif_tau_theta', 'lif_gamma', 'lif_beta']
        lif_kwargs = {}
        # 从kwargs中弹出这些键，放入lif_kwargs字典
        for key in lif_keys:
            if key in kwargs:
                lif_kwargs[key] = kwargs.pop(key)
        # ---------------------------

        # 1. 空间GNN预处理
        self.spatial_gnn = SpatialGNNWrapper(channels, channels)
        self.norm1 = nn.LayerNorm(channels)

        # 2. 时间延迟建模
        self.delay_line = DelayLine(channels, kernel_size=delay_kernel)
        self.norm2 = nn.LayerNorm(channels)

        # 3. 时空信息聚合
        # 现在kwargs中已经没有LIF参数了，可以安全传递
        self.aggregator = STAGNNAggregator(d_in=channels, d=channels, heads=heads, W=W, **kwargs)
        
        # 4. 脉冲发放
        self.msg_proj = nn.Linear(channels, 1) # 投影聚合消息到标量电流
        # 使用分离出来的lif_kwargs进行初始化
        self.lif_cell = LIFCell(**lif_kwargs)
        
        # 5. FFN 和最终输出处理
        self.ffn = MLP(channels)
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, spikes: Optional[torch.Tensor], edge_index: torch.Tensor, time_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N, C = x.shape
        initial_input = x
        
        if spikes is None:
            spikes = torch.ones((T, N), device=x.device, dtype=x.dtype)

        # 1. 空间GNN预处理
        x_spatial = self.spatial_gnn(x, edge_index)
        x = self.norm1(x + x_spatial)

        # 2. 时间延迟建模
        x_delayed = self.delay_line(x)
        x = self.norm2(x + x_delayed)

        # 3. 时空聚合
        aggregated_message = self.aggregator(x, spikes, edge_index, time_idx)

        # 4. 脉冲发放
        # [T, N, C] -> [T, N, 1] -> [T, N]
        input_current = self.msg_proj(aggregated_message).squeeze(-1)
        new_spikes, new_v, _ = self.lif_cell(input_current)

        # 5. 最终输出 (FFN + 宏观残差)
        # FFN作用在聚合后的消息上，这是最富含信息的张量
        ffn_out = self.ffn(aggregated_message)
        layer_output_features = self.final_norm(initial_input + ffn_out)

        return layer_output_features, new_spikes

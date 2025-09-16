# 文件: spikenet_x/spiketdanet_layer.py

import torch
import torch.nn as nn
from typing import Tuple, Optional

# from .new_modules import SpatialGNNWrapper, DelayLine, STAGNNAggregator
from .new_modules import SpatialGNNWrapper, DelayLine
from .new_modules.sta_gnn_agg_optimized import STAGNNAggregator_Optimized as STAGNNAggregator

from .surrogate_lif_cell import SurrogateLIFCell # [MODIFIED] 导入新的LIF单元

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
    # [MODIFIED] 更新__init__签名以接收LIF超参数
    def __init__(self, channels: int, heads: int, W: int, delay_kernel: int = 5, 
                 lif_tau=0.95, lif_v_threshold=1.0, lif_alpha=1.0, lif_surrogate='sigmoid', **kwargs):
        super().__init__()
        self.channels = channels
        
        # 1. 空间GNN预处理
        self.spatial_gnn = SpatialGNNWrapper(channels, channels)
        self.norm1 = nn.LayerNorm(channels)

        # 2. 时间延迟建模
        self.delay_line = DelayLine(channels, kernel_size=delay_kernel)
        self.norm2 = nn.LayerNorm(channels)

        # 3. 时空信息聚合
        self.aggregator = STAGNNAggregator(d_in=channels, d=channels, heads=heads, W=W, **kwargs)
        
        # 4. 脉冲发放 (使用新的LIF单元)
        # [REMOVED] self.msg_proj = nn.Linear(channels, 1) # 移除信息瓶颈
        self.lif_cell = SurrogateLIFCell(
            channels=channels,
            tau=lif_tau,
            v_threshold=lif_v_threshold,
            alpha=lif_alpha,
            surrogate=lif_surrogate
        )
        
        # 5. FFN 和最终输出处理
        self.ffn = MLP(channels)
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, spikes: Optional[torch.Tensor], edge_index: torch.Tensor, time_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N, C = x.shape
        
        # --- [NEW] START: Handle None spikes for the first layer ---
        if spikes is None:
            # For the first layer, assume all-one spikes (no gating)
            # Shape should be [T, N, C] to match features for the aggregator
            spikes = torch.ones((T, N, C), device=x.device, dtype=x.dtype)
        # --- [NEW] END ---
        
        # [MODIFIED] 如果上一层脉冲是[T,N]，需要扩展以匹配特征维度
        # This part is now a fallback, but good to keep.
        elif spikes.dim() == 2:
            spikes = spikes.unsqueeze(-1).expand(-1, -1, C)

        # 1. 空间GNN预处理
        x_spatial = self.spatial_gnn(x, edge_index)
        x = self.norm1(x + x_spatial)

        # 2. 时间延迟建模
        x_delayed = self.delay_line(x)
        x_processed = self.norm2(x + x_delayed)

        # 3. 时空聚合
        aggregated_message = self.aggregator(x_processed, spikes, edge_index, time_idx)

        # 4. 脉冲发放 (处理高维特征)
        # [MODIFIED] 直接将高维消息送入LIF
        spikes_hd, _ = self.lif_cell(aggregated_message) # spikes_hd shape: [T, N, C]

        # 5. 最终输出 (FFN + 宏观残差)
        # [MODIFIED] 使用脉冲作为门控信号
        ffn_input = aggregated_message * spikes_hd
        ffn_out = self.ffn(ffn_input)
        layer_output_features = self.final_norm(x_processed + ffn_out)

        # [MODIFIED] 为下一层准备脉冲信号 (通过平均降维)
        new_spikes_for_next_layer = spikes_hd.mean(dim=-1)

        return layer_output_features, new_spikes_for_next_layer


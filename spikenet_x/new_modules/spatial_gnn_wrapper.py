# spikenet_x/new_modules/spatial_gnn_wrapper.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class SpatialGNNWrapper(nn.Module):
    """
    在时序图的每个时间步上高效地应用SAGEConv。
    输入格式: [T, N, d]
    """
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean'):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels, aggr=aggr)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 节点特征, 形状 [T, N, d_in]
            edge_index (torch.Tensor): 图的边索引, 形状 [2, E]

        Returns:
            torch.Tensor: 空间聚合后的节点特征, 形状 [T, N, d_out]
        """
        T, N, d = x.shape
        # 将 x 重塑为 [T*N, d] 以便进行批处理GNN卷积
        x_reshaped = x.reshape(T * N, d)

        # 扩展 edge_index 以匹配 T 个图快照
        # PyG的SAGEConv期望节点索引是全局的，所以我们需要为每个时间步的节点创建偏移
        edge_indices = [edge_index + t * N for t in range(T)]
        edge_index_expanded = torch.cat(edge_indices, dim=1)

        out_reshaped = self.conv(x_reshaped, edge_index_expanded)
        out_reshaped = self.activation(out_reshaped)

        # 将输出恢复为 [T, N, d_out]
        return out_reshaped.reshape(T, N, -1)

# spikenet_x/new_modules/delay_line.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DelayLine(nn.Module):
    """
    使用因果深度可分离1D卷积，低成本地建模多种时间延迟。
    输入格式: [T, N, d]
    """
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # 因果填充，只在左侧（过去）填充
        self.padding = kernel_size - 1
        
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            padding=self.padding, 
            groups=channels
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=1
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征, 形状 [T, N, d]

        Returns:
            torch.Tensor: 经过延迟建模后的特征, 形状 [T, N, d]
        """
        # [T, N, d] -> [N, d, T]
        x_permuted = x.permute(1, 2, 0)
        
        out = self.depthwise_conv(x_permuted)
        out = self.pointwise_conv(out)
        
        # 切片以保持输出长度为T，实现因果性
        out = out[:, :, :x_permuted.size(2)]
        
        out = self.activation(out)
        
        # [N, d, T] -> [T, N, d]
        return out.permute(2, 0, 1)

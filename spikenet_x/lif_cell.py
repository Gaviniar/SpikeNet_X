import torch
import torch.nn as nn
from typing import Tuple

class LIFCell(nn.Module):
    """
    一个批处理的、基于循环的 Leaky Integrate-and-Fire (LIF) 神经元单元。
    它接收一个形状为 [T, N] 的输入电流，并按时间步进行处理。
    """
    def __init__(self, lif_tau_theta: float = 1.0, lif_gamma: float = 0.95, lif_beta: float = 0.95):
        """
        初始化LIF神经元。

        Args:
            lif_tau_theta (float): 膜电位阈值 (V_th)。
            lif_gamma (float): 脉冲衰减因子，用于脉冲后的电位重置。
            lif_beta (float): 膜电位泄漏/衰减因子。
        """
        super().__init__()
        # 使用 register_buffer 将这些张量注册为模型的持久状态，但不是模型参数（不会被优化器更新）
        self.register_buffer("tau_theta", torch.tensor(lif_tau_theta))
        self.register_buffer("gamma", torch.tensor(lif_gamma))
        self.register_buffer("beta", torch.tensor(lif_beta))

    def forward(self, I_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LIF神经元的前向传播。

        Args:
            I_in (torch.Tensor): 输入电流，形状为 [T, N]，T是时间步数，N是节点数。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - spikes (torch.Tensor): 输出脉冲序列，形状 [T, N]。
            - v_mem (torch.Tensor): 膜电位历史记录，形状 [T, N]。
            - spike_history (torch.Tensor): 脉冲历史记录（与spikes相同，用于可能的内部调试）。
        """
        T, N = I_in.shape
        device = I_in.device

        # 初始化膜电位和脉冲历史记录
        v = torch.zeros(N, device=device)
        s = torch.zeros(N, device=device)

        # 用于存储每个时间步结果的列表
        v_mem_list = []
        spike_list = []

        # 按时间步循环处理
        for t in range(T):
            # 计算新的膜电位
            # v_new = v_old * beta (泄漏) + I_in (积分) - s_old * gamma (重置)
            v = self.beta * v + I_in[t] - self.gamma * s
            
            # 检查是否超过阈值，产生脉冲
            s = (v > self.tau_theta).float()
            
            # 脉冲发放后，重置膜电位 (硬重置)
            v = v * (1.0 - s)

            # 存储当前时间步的结果
            v_mem_list.append(v)
            spike_list.append(s)

        # 将列表堆叠成张量
        v_mem_out = torch.stack(v_mem_list, dim=0)
        spikes_out = torch.stack(spike_list, dim=0)
        
        return spikes_out, v_mem_out, spikes_out

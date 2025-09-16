# F:\SomeProjects\CSGNN\spikenet_x\surrogate_lif_cell.py
import torch
import torch.nn as nn
from typing import Tuple

# --- 拷贝自 spikenet/neuron.py 的替代梯度函数 ---
class BaseSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

class SigmoidSpike(BaseSpike):
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1. - sgax) * sgax * alpha
        return grad_input * sg, None

class TriangleSpike(BaseSpike):
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None

SURROGATE_MAP = {
    'sigmoid': SigmoidSpike.apply,
    'triangle': TriangleSpike.apply
}
# --- 替代梯度函数结束 ---


class SurrogateLIFCell(nn.Module):
    """
    一个支持高维特征并使用替代梯度进行训练的LIF神经元。
    它以 unrolled 方式处理 [T, N, D] 的输入。
    """
    def __init__(self, channels: int, v_threshold=1.0, v_reset=0.0, tau=0.95, alpha=1.0, surrogate='sigmoid'):
        super().__init__()
        self.channels = channels
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # 确保 tau 是一个可训练或固定的缓冲区
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        
        if surrogate not in SURROGATE_MAP:
            raise ValueError(f"Surrogate function '{surrogate}' is not supported. Available: {list(SURROGATE_MAP.keys())}")
        self.surrogate_fn = SURROGATE_MAP[surrogate]

    def forward(self, I_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LIF神经元的前向传播。

        Args:
            I_in (torch.Tensor): 输入电流/特征，形状为 [T, N, D]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - spikes (torch.Tensor): 输出脉冲序列，形状 [T, N, D]。
            - v_mem (torch.Tensor): 膜电位历史记录，形状 [T, N, D]。
        """
        T, N, D = I_in.shape
        device = I_in.device

        # 初始化膜电位
        v = torch.zeros(N, D, device=device)

        v_mem_list = []
        spike_list = []

        # 按时间步循环处理
        for t in range(T):
            # 1. 膜电位更新 (leaky integration)
            # v_new = v_old * tau (leak) + I_in (integrate)
            # 注意：原版Spike的LIF公式是 v = v + (dv - (v-v_reset))/tau
            # 这里简化为 v = v*tau + I_in，更常见
            v = v * self.tau + I_in[t]
            
            # 2. 发放脉冲 (使用替代梯度)
            # spike = surrogate(v - v_threshold)
            spike = self.surrogate_fn(v - self.v_threshold, self.alpha)
            
            # 3. 膜电位重置 (reset by subtraction)
            v = v - spike * self.v_threshold

            v_mem_list.append(v)
            spike_list.append(spike)

        v_mem_out = torch.stack(v_mem_list, dim=0)
        spikes_out = torch.stack(spike_list, dim=0)
        
        return spikes_out, v_mem_out


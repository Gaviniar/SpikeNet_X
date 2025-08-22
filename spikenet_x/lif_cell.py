# -*- coding: utf-8 -*-
"""
LIFCell: 脉冲神经元单元（支持自适应阈值与 fast-tanh 代理梯度）

接口
----
forward(M: Float[T, N, d]) -> Tuple[S: Float[T, N], V: Float[T, N], aux: Dict]
- M 为从聚合器得到的消息（电流输入）
- 先用线性投影 U: R^d -> R 将通道聚合为标量电流 I_tn
- 递推更新膜电位与阈值，产生脉冲

参考公式（提示词）
----------------
V_{i,t} = λ V_{i,t-1} + U m_{i,t} - θ_{i,t-1} R_{i,t-1}
S_{i,t} = 𝟙[V_{i,t} > θ_{i,t}]
V_{i,t} ← V_{i,t} - S_{i,t} · θ_{i,t}          (重置)
θ_{i,t} = τ_θ θ_{i,t-1} + γ S_{i,t-1}          (自适应阈值，可选)

训练
----
- 使用 fast-tanh 代理梯度:
  y = H(x) + (tanh(βx) - tanh(βx).detach())
  其中 H(x) 为硬阶跃 (x>0)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _fast_tanh_surrogate(x: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """
    硬触发 + 平滑梯度的 STE 实现:
      forward: step(x)
      backward: tanh(βx) 的导数 (≈ β * (1 - tanh^2(βx)))
    """
    hard = (x > 0).to(x.dtype)
    soft = torch.tanh(beta * x)
    return hard + (soft - soft.detach())


class LIFCell(nn.Module):
    def __init__(
        self,
        d: int,
        lambda_mem: float = 0.95,
        tau_theta: float = 0.99,
        gamma: float = 0.10,
        adaptive: bool = True,
        surrogate: str = "fast_tanh",
        beta: float = 2.0,
    ) -> None:
        super().__init__()
        assert 0.0 <= lambda_mem <= 1.0
        assert 0.0 <= tau_theta <= 1.0
        assert gamma >= 0.0

        self.d = int(d)
        self.adaptive = bool(adaptive)
        self.surrogate = str(surrogate)
        self.beta = float(beta)

        # U: R^d -> R（共享于所有节点），无偏置避免电流漂移
        self.proj = nn.Linear(d, 1, bias=False)

        # 将标量参数注册为 buffer，便于脚本化与移动设备
        self.register_buffer("lambda_mem", torch.as_tensor(lambda_mem, dtype=torch.float32))
        self.register_buffer("tau_theta", torch.as_tensor(tau_theta, dtype=torch.float32))
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float32))

    def _spike(self, x: torch.Tensor) -> torch.Tensor:
        if self.surrogate == "fast_tanh":
            return _fast_tanh_surrogate(x, beta=self.beta)
        # 兜底：纯硬阈值（无代理梯度）
        return (x > 0).to(x.dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        M: torch.Tensor,                # [T, N, d]
        state0: Optional[Dict] = None,  # 可选: {"V": [N], "theta": [N], "S": [N]}
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        assert M.dim() == 3, "M 形状应为 [T, N, d]"
        T, N, d = M.shape
        assert d == self.d, f"d 不匹配: 期望 {self.d}, 实得 {d}"

        device = M.device
        dtype = M.dtype

        # 初始状态
        if state0 is None:
            V = torch.zeros(N, device=device, dtype=dtype)
            theta = torch.ones(N, device=device, dtype=dtype)  # 初始阈值 1.0
            S_prev = torch.zeros(N, device=device, dtype=dtype)
        else:
            V = state0.get("V", torch.zeros(N, device=device, dtype=dtype)).to(dtype)
            theta = state0.get("theta", torch.ones(N, device=device, dtype=dtype)).to(dtype)
            S_prev = state0.get("S", torch.zeros(N, device=device, dtype=dtype)).to(dtype)

        S_seq = []
        V_seq = []
        theta_seq = []

        lam = self.lambda_mem
        tau = self.tau_theta
        gam = self.gamma

        for t in range(T):
            # 投影到标量电流 I_tn: [N]
            I = self.proj(M[t]).squeeze(-1)  # [N]

            # 记忆衰减 + 输入累积
            V = lam * V + I - (theta * S_prev)  # 包含上一步的 Refractory 抑制项

            # 触发条件与代理梯度
            x = V - theta
            S = self._spike(x)  # [N] in [0,1]

            # 重置：发放处扣除阈值
            V = V - S * theta

            # 自适应阈值
            if self.adaptive:
                theta = tau * theta + gam * S_prev

            # 记录
            S_seq.append(S)
            V_seq.append(V)
            theta_seq.append(theta)

            # 更新上一时刻的发放
            S_prev = S

        S_out = torch.stack(S_seq, dim=0)  # [T, N]
        V_out = torch.stack(V_seq, dim=0)  # [T, N]

        aux = {
            "theta": torch.stack(theta_seq, dim=0),   # [T, N]
            "spike_rate": S_out.mean().detach(),      # 标量，便于监控
        }
        return S_out, V_out, aux

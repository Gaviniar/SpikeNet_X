# -*- coding: utf-8 -*-
"""
LearnableDelayLine: 可学习多延迟通路（因果深度可分离 1D 卷积）

- 输入输出形状:
  forward(H: Float[T, N, d_in]) -> Float[T, N, d_in]

- 设计:
  对每个通道 c, 时间步 t:
      H_tilde[t, :, c] = sum_{k=0..K-1} w_c[k] * H[t-k, :, c]
  其中 w_c[k] = softplus(u_c[k]) * rho^k / sum_r softplus(u_c[r]) * rho^r
  并在 t-k < 0 使用因果左填充 0。

- 实现:
  将输入重排为 [N, d_in, T]，使用 groups=d_in 的 Conv1d 执行深度可分离因果卷积。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDelayLine(nn.Module):
    def __init__(
        self,
        d_in: int,
        K: int = 5,
        rho: float = 0.85,
        per_channel: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        参数
        ----
        d_in : 输入通道数（特征维）
        K : 延迟 tap 数 (>=1)
        rho : 指数折扣因子，(0,1)
        per_channel : True 表示逐通道独立权重；False 表示所有通道共享一组权重
        eps : 归一化时的数值稳定项
        """
        super().__init__()
        assert d_in >= 1, "d_in 必须 >= 1"
        assert K >= 1, "K 必须 >= 1"
        assert 0.0 < rho < 1.0, "rho 必须在 (0,1)"

        self.d_in = int(d_in)
        self.K = int(K)
        self.rho = float(rho)
        self.per_channel = bool(per_channel)
        self.eps = float(eps)

        # 原始可学习参数 u，经 softplus 后非负
        if self.per_channel:
            self.u = nn.Parameter(torch.zeros(self.d_in, self.K))
        else:
            self.u = nn.Parameter(torch.zeros(self.K))

        # 预先缓存 rho 的幂 [K]
        rho_pow = torch.tensor([self.rho ** k for k in range(self.K)], dtype=torch.float32)
        self.register_buffer("rho_pow", rho_pow, persistent=True)

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, K={self.K}, rho={self.rho}, per_channel={self.per_channel}"

    @torch.no_grad()
    def get_delay_weights(self) -> torch.Tensor:
        """
        返回当前归一化后的延迟权重 w，形状:
          - per_channel=True: [d_in, K]
          - per_channel=False: [K]
        便于监控/可视化。
        """
        if self.per_channel:
            # [d_in, K]
            sp = F.softplus(self.u)
            num = sp * self.rho_pow  # 广播到 [d_in, K]
            den = num.sum(dim=1, keepdim=True).clamp_min(self.eps)
            w = num / den
            return w
        else:
            # [K]
            sp = F.softplus(self.u)
            num = sp * self.rho_pow
            den = num.sum().clamp_min(self.eps)
            w = num / den
            return w

    def _build_depthwise_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        构造 Conv1d 的 depthwise 卷积核:
          - 形状 [d_in, 1, K]
          - 每个通道自有一条核（若 per_channel=True），否则共享一条核并复制
        """
        if self.per_channel:
            # [d_in, K] -> [d_in, 1, K]
            w = self.get_delay_weights().to(device=device, dtype=dtype).unsqueeze(1)
        else:
            # 共享核 [K] -> [d_in, 1, K]
            w_shared = self.get_delay_weights().to(device=device, dtype=dtype).view(1, 1, self.K)
            w = w_shared.expand(self.d_in, 1, self.K).contiguous()
        return w  # Float[d_in, 1, K]

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        H : Float[T, N, d_in]

        返回
        ----
        H_tilde : Float[T, N, d_in]
        """
        assert H.dim() == 3, "输入 H 形状应为 [T, N, d_in]"
        T, N, Din = H.shape
        assert Din == self.d_in, f"d_in 不匹配: 期望 {self.d_in}, 实得 {Din}"

        # [T, N, d] -> [N, d, T]
        x = H.permute(1, 2, 0).contiguous()

        # 因果左填充 K-1
        pad_left = self.K - 1
        if pad_left > 0:
            x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)  # 在时间维左侧填充

        # 深度可分离卷积 (groups=d_in)
        weight = self._build_depthwise_kernel(H.device, H.dtype)  # [d, 1, K]
        y = F.conv1d(x, weight=weight, bias=None, stride=1, padding=0, groups=self.d_in)
        # y: [N, d, T]

        # 回到 [T, N, d]
        H_tilde = y.permute(2, 0, 1).contiguous()
        return H_tilde

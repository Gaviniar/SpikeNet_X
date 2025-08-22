# -*- coding: utf-8 -*-
"""
相对时间编码与相对偏置

- 提供 RelativeTimeEncoding(nn.Module)
  * forward(time_idx: Long[T], W:int) -> (pe_table: Float[W+1, d_pe], rel_bias: Float[W+1])
  * pe_table[k] 表示 Δt = k 的编码向量（仅用于 0..W）
  * rel_bias 为可学习标量偏置 b[Δt]，长度 W+1

- 设计：
  * 指数衰减对（tau_m, tau_s）
  * 对数间隔频率的 sin/cos 对（n_freq 个频率）
  * 可选 log-bucket one-hot（num_buckets=0 表示关闭）
  * 输出维度 d_pe = 2 + 2*n_freq + num_buckets

- 数值与工程：
  * 仅构造 0..W 的 Δt 子表，避免构建完整 [T,T] 矩阵
  * Δt 使用 float 计算后堆叠为编码
"""

from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn


class RelativeTimeEncoding(nn.Module):
    """
    相对时间编码（仅依赖 Δt），并带可学习相对偏置表 b[0..W]。

    参数
    ----
    taus : Tuple[float, float]
        指数衰减的两个时间常数 (tau_m, tau_s)
    n_freq : int
        正弦/余弦的对数间隔频率个数
    num_buckets : int
        log-bucket one-hot 的桶数（0 表示关闭）
    """

    def __init__(
        self,
        taus: Tuple[float, float] = (4.0, 16.0),
        n_freq: int = 3,
        num_buckets: int = 0,
    ) -> None:
        super().__init__()
        assert len(taus) == 2 and taus[0] > 0 and taus[1] > 0
        assert n_freq >= 0
        assert num_buckets >= 0

        self.tau_m = float(taus[0])
        self.tau_s = float(taus[1])
        self.n_freq = int(n_freq)
        self.num_buckets = int(num_buckets)

        # 缓存最近一次构造的 W，以便重用不同 batch 的同一窗口
        self._cached_W = None
        self.register_buffer("_cached_pe", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        # 注意：rel_bias 的长度依赖于 W，故在 forward 时按需创建/扩展为 Parameter
        self.register_parameter("_rel_bias", None)

    @property
    def d_pe(self) -> int:
        # 2 (双指数) + 2*n_freq (sin/cos 对) + num_buckets (one-hot)
        return 2 + 2 * self.n_freq + self.num_buckets

    @staticmethod
    def _log_spaced_frequencies(n_freq: int, W: int) -> torch.Tensor:
        """
        生成对数间隔频率（角频率 ω），范围大致覆盖 [1/(2W), 1/2]（经验值）。
        """
        if n_freq <= 0:
            return torch.empty(0)
        f_min = 1.0 / max(2.0 * W, 1.0)
        f_max = 0.5
        freqs = torch.logspace(
            start=math.log10(f_min),
            end=math.log10(f_max),
            steps=n_freq,
        )
        # 转为角频率
        return 2.0 * math.pi * freqs

    @staticmethod
    def _log_bucketize(delta: torch.Tensor, num_buckets: int) -> torch.Tensor:
        """
        将 Δt 做对数分桶并返回 one-hot；delta >= 0 的整数张量。
        """
        if num_buckets <= 0:
            return torch.empty(delta.shape + (0,), device=delta.device, dtype=delta.dtype)

        # +1 防止 log(0)
        logv = torch.log2(delta.to(torch.float32) + 1.0)
        # 线性映射到桶 [0, num_buckets-1]
        idx = torch.clamp((logv / torch.clamp(logv.max(), min=1.0e-6)) * (num_buckets - 1), 0, num_buckets - 1)
        idx = idx.round().to(torch.long)

        one_hot = torch.zeros(delta.shape + (num_buckets,), device=delta.device, dtype=delta.dtype)
        one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return one_hot

    def _build_pe_for_window(self, W: int, device: torch.device) -> torch.Tensor:
        """
        构建长度 W+1 的相对时间编码表：pe[k] = phi(Δt=k)，形状 [W+1, d_pe]
        """
        delta = torch.arange(0, W + 1, device=device, dtype=torch.float32)  # [0..W]

        # 双指数衰减
        exp_m = torch.exp(-delta / self.tau_m)  # [W+1]
        exp_s = torch.exp(-delta / self.tau_s)  # [W+1]

        # 正弦/余弦
        omegas = self._log_spaced_frequencies(self.n_freq, W).to(device)
        if omegas.numel() > 0:
            # [W+1, n_freq]
            arg = delta.unsqueeze(-1) * omegas.unsqueeze(0)
            sinv = torch.sin(arg)
            cosv = torch.cos(arg)
            sincos = torch.cat([sinv, cosv], dim=-1)  # [W+1, 2*n_freq]
        else:
            sincos = torch.empty((W + 1, 0), device=device, dtype=torch.float32)

        # log-bucket one-hot
        if self.num_buckets > 0:
            buckets = self._log_bucketize(delta.to(torch.long), self.num_buckets).to(torch.float32)  # [W+1, B]
        else:
            buckets = torch.empty((W + 1, 0), device=device, dtype=torch.float32)

        pe = torch.cat(
            [
                exp_m.unsqueeze(-1),
                exp_s.unsqueeze(-1),
                sincos,
                buckets,
            ],
            dim=-1,
        )
        return pe  # [W+1, d_pe]

    def _ensure_bias(self, W: int, device: torch.device) -> torch.Tensor:
        """
        确保存在长度 W+1 的可学习偏置表；如果已有更短表，做扩展并保留已学部分。
        """
        if self._rel_bias is None:
            self._rel_bias = nn.Parameter(torch.zeros(W + 1, device=device))
        elif self._rel_bias.numel() < (W + 1):
            old = self._rel_bias.data
            new = torch.zeros(W + 1, device=device)
            new[: old.numel()] = old
            self._rel_bias = nn.Parameter(new)
        return self._rel_bias

    def forward(
        self,
        time_idx: torch.Tensor,  # Long[T]，通常为 arange(T)
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建窗口 0..W 的相对时间编码子表与相对偏置。

        返回
        ----
        pe_table : Float[W+1, d_pe]
        rel_bias : Float[W+1]
        """
        assert time_idx.dim() == 1, "time_idx 应为一维 LongTensor [T]"
        assert W >= 0, "W >= 0"

        device = time_idx.device
        # 缓存与重用
        if self._cached_W == W and self._cached_pe is not None and self._cached_pe.device == device:
            pe = self._cached_pe
        else:
            pe = self._build_pe_for_window(W, device)
            self._cached_W = W
            self._cached_pe = pe

        rel_bias = self._ensure_bias(W, device)
        return pe, rel_bias

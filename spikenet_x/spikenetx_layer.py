# -*- coding: utf-8 -*-
"""
SpikeNetXLayer: DelayLine + Spiking Temporal Attention (STA) + LIF 单层封装

根据《提示词.md》的签名：
    class SpikeNetXLayer(nn.Module):
        def __init__(...):
            ...
        def forward(self, H, S_prev, edge_index, time_idx, adj_mask=None, batch=None):
            H̃ = self.delay(H)                                   # [T,N,d_in]
            M  = self.sta(H̃, S_prev, edge_index, time_idx, adj_mask)  # [T,N,d]
            S, V, aux = self.neuron(M)                           # [T,N], [T,N]
            Y = self.norm(M + self.ffn(M))                       # 残差 + 归一
            return S, V, Y, {"M": M, **aux}

说明
----
- 稠密 STA 回退实现在 spikenet_x/sta.py 中，适合小图或功能验证；
- 稀疏边 rolling-window 版本可在后续新增（接口保持一致）；
- 本层不改变时间长度 T 与节点数 N，仅改变通道维（d_in -> d）。

"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .delayline import LearnableDelayLine
from .sta import SpikingTemporalAttention
from .sta_sparse import SparseSpikingTemporalAttention
from .lif_cell import LIFCell


class MLP(nn.Module):
    def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = d * hidden_mult
        self.fc1 = nn.Linear(d, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d, bias=True)
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SpikeNetXLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int,
        heads: int = 4,
        topk: int = 16,
        W: int = 64,
        K: int = 5,
        rho: float = 0.85,
        use_rel_bias: bool = True,
        attn_drop: float = 0.1,
        temp: float = 1.0,
        attn_impl: str = "dense",
        per_channel: bool = True,
        ffn_hidden_mult: int = 4,
        ffn_drop: float = 0.1,
        lif_lambda_mem: float = 0.95,
        lif_tau_theta: float = 0.99,
        lif_gamma: float = 0.10,
        lif_adaptive: bool = True,
        lif_surrogate: str = "fast_tanh",
        lif_beta: float = 2.0,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d = int(d)
        self.attn_impl = str(attn_impl)
        assert self.attn_impl in ("dense", "sparse"), "attn_impl must be 'dense' or 'sparse'"

        # 1) DelayLine（因果深度可分离 1D 卷积）
        self.delay = LearnableDelayLine(d_in=d_in, K=K, rho=rho, per_channel=per_channel)

        # 2) STA 聚合：根据 attn_impl 选择稠密/稀疏实现
        if self.attn_impl == "sparse":
            # 稀疏版本不支持 topk（后续可流式实现）
            self.sta = SparseSpikingTemporalAttention(
                d_in=d_in,
                d=d,
                heads=heads,
                W=W,
                use_rel_bias=use_rel_bias,
                attn_drop=attn_drop,
                temp=temp,
            )
        else:
            self.sta = SpikingTemporalAttention(
                d_in=d_in,
                d=d,
                heads=heads,
                topk=topk,
                W=W,
                use_rel_bias=use_rel_bias,
                attn_drop=attn_drop,
                temp=temp,
            )

        # 3) 脉冲单元
        self.neuron = LIFCell(
            d=d,
            lambda_mem=lif_lambda_mem,
            tau_theta=lif_tau_theta,
            gamma=lif_gamma,
            adaptive=lif_adaptive,
            surrogate=lif_surrogate,
            beta=lif_beta,
        )

        # 归一与 FFN（残差）
        self.norm = nn.LayerNorm(d)
        self.ffn = MLP(d=d, hidden_mult=ffn_hidden_mult, dropout=ffn_drop)

    def forward(
        self,
        H: torch.Tensor,                       # [T, N, d_in]
        S_prev: Optional[torch.Tensor],        # [T, N] 或 None（None 时采用全 1 门控）
        edge_index: Optional[torch.Tensor],    # [2, E] 或 None（若提供 adj_mask 可为 None）
        time_idx: torch.Tensor,                # [T]
        adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool
        batch: Optional[torch.Tensor] = None,     # 预留；当前未使用
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        assert H.dim() == 3, "H 形状应为 [T, N, d_in]"
        T, N, Din = H.shape
        assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"

        device = H.device
        dtype = H.dtype

        # 若上一层脉冲缺省，则使用全 1 门控（不抑制注意力）
        if S_prev is None:
            S_gate = torch.ones((T, N), device=device, dtype=dtype)
        else:
            assert S_prev.shape == (T, N), "S_prev 形状应为 [T, N]"
            S_gate = S_prev.to(device=device, dtype=dtype)

        # 1) DelayLine
        H_tilde = self.delay(H)  # [T, N, d_in]

        # 2) STA 聚合为 d 维消息
        M = self.sta(H_tilde, S_gate, edge_index=edge_index, time_idx=time_idx, adj_mask=adj_mask)  # [T, N, d]

        # 3) LIF 发放
        S, V, aux = self.neuron(M)  # S:[T,N], V:[T,N]
        aux = {"M": M, **aux}

        # 4) 残差 + 归一（Pre-LN 的一个简化变体）
        Y = self.norm(M + self.ffn(M))  # [T, N, d]

        return S, V, Y, aux

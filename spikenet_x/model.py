# -*- coding: utf-8 -*-
"""
SpikeNet-X: multi-layer network composed of (DelayLine -> STA -> LIF) blocks.

This module provides a thin, task-agnostic backbone `SpikeNetX` that stacks
`SpikeNetXLayer` L times and (optionally) a lightweight readout head.

Key ideas follow `提示词.md`:
- Time-first tensors: H: [T, N, d_in], S: [T, N]
- Event-driven STA with causal window W and Top-k sparsification
- Learnable DelayLine in front of STA to model propagation delay
- LIF cell generates spikes that can gate attention in the next layer

Typical usage
-------------
>>> import torch
>>> from spikenet_x import SpikeNetX
>>> T, N, d_in, d, Hs, L = 32, 128, 64, 128, 4, 2
>>> H0 = torch.randn(T, N, d_in)
>>> edge_index = torch.randint(0, N, (2, 4*N))  # toy edges
>>> time_idx = torch.arange(T)
>>> model = SpikeNetX(d_in=d_in, d=d, layers=L, heads=Hs, topk=16, W=32, out_dim=10)
>>> out = model(H0, edge_index=edge_index, time_idx=time_idx)
>>> out["logits"].shape  # [N, out_dim] by default (last-time readout)
torch.Size([128, 10])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .spikenetx_layer import SpikeNetXLayer


class SpikeNetX(nn.Module):
    """
    A stack of SpikeNetXLayer blocks with optional classifier head.

    Args
    ----
    d_in: int
        Input feature dimension.
    d: int
        Hidden/STA output dimension per layer.
    layers: int
        Number of stacked layers.
    heads: int
        Number of attention heads per layer.
    topk: int
        Top-k candidates kept per (i,t) in STA.
    W: int
        Causal attention time window.
    K: int
        DelayLine taps.
    rho: float
        DelayLine exponential discount.
    use_rel_bias: bool
        Whether to use learnable relative bias b[Δt].
    attn_drop: float
        Attention dropout prob.
    temp: float
        Softmax temperature for attention logits.
    per_channel: bool
        Per-channel DelayLine weights if True (recommended).
    ffn_hidden_mult: int
        Multiplier of FFN hidden width inside each layer.
    ffn_drop: float
        Dropout inside layer FFN.
    lif_*: see LIFCell.
    out_dim: Optional[int]
        If set, attach a linear head to produce logits for node-level tasks.
    readout: str
        'last' (default): use last time-step T-1 for logits,
        'mean': temporal mean pooling over T.
    """

    def __init__(
        self,
        d_in: int,
        d: int,
        layers: int = 2,
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
        lif_tau_theta: float = 0.95,
        lif_gamma: float = 0.20,
        lif_adaptive: bool = True,
        lif_surrogate: str = "fast_tanh",
        lif_beta: float = 2.0,
        out_dim: Optional[int] = None,
        readout: str = "last",
    ) -> None:
        super().__init__()
        assert layers >= 1, "layers must be >= 1"
        assert readout in ("last", "mean"), "readout must be 'last' or 'mean'"

        self.layers = int(layers)
        self.readout = readout
        self.out_dim = out_dim
        self.attn_impl = attn_impl
        assert self.attn_impl in ("dense", "sparse"), "attn_impl must be 'dense' or 'sparse'"

        mods: List[SpikeNetXLayer] = []
        for l in range(layers):
            in_dim = d_in if l == 0 else d
            mods.append(
                SpikeNetXLayer(
                    d_in=in_dim,
                    d=d,
                    heads=heads,
                    topk=topk,
                    W=W,
                    K=K,
                    rho=rho,
                    use_rel_bias=use_rel_bias,
                    attn_drop=attn_drop,
                    temp=temp,
                    attn_impl=attn_impl,
                    per_channel=per_channel,
                    ffn_hidden_mult=ffn_hidden_mult,
                    ffn_drop=ffn_drop,
                    lif_lambda_mem=lif_lambda_mem,
                    lif_tau_theta=lif_tau_theta,
                    lif_gamma=lif_gamma,
                    lif_adaptive=lif_adaptive,
                    lif_surrogate=lif_surrogate,
                    lif_beta=lif_beta,
                )
            )
        self.blocks = nn.ModuleList(mods)

        self.head = nn.Linear(d, out_dim, bias=True) if out_dim is not None else None
        if self.head is not None:
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        H: torch.Tensor,                        # [T, N, d_in]
        edge_index: Optional[torch.Tensor],     # [2, E] or None (if adj_mask provided)
        time_idx: torch.Tensor,                 # [T]
        adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool or None
        S0: Optional[torch.Tensor] = None,        # initial spikes for layer-0 gating [T, N] (optional)
    ) -> Dict[str, torch.Tensor]:
        assert H.dim() == 3, "H should be [T, N, d_in]"
        T, N, _ = H.shape
        assert time_idx.dim() == 1 and time_idx.numel() == T, "time_idx must be [T]"

        S_prev = S0  # first layer gating; None -> all-ones gating inside block
        Y = None
        S_list: List[torch.Tensor] = []
        V_list: List[torch.Tensor] = []

        X = H
        aux_last: Dict[str, torch.Tensor] = {}
        for blk in self.blocks:
            S, V, Y, aux = blk(
                H=X,
                S_prev=S_prev,
                edge_index=edge_index,
                time_idx=time_idx,
                adj_mask=adj_mask,
            )
            S_list.append(S)   # each: [T, N]
            V_list.append(V)
            S_prev = S         # spikes feed-forward as gate for next layer
            X = Y              # features for next layer
            aux_last = aux

        # Readout
        if self.readout == "last":
            z = Y[-1]  # [N, d]
        else:  # "mean"
            z = Y.mean(dim=0)  # [N, d]

        logits = self.head(z) if self.head is not None else None

        out: Dict[str, torch.Tensor] = {
            "repr": z,                       # [N, d]
            "Y_last": Y,                     # [T, N, d]
            "S_list": torch.stack(S_list),   # [L, T, N]
            "V_list": torch.stack(V_list),   # [L, T, N]
        }
        if logits is not None:
            out["logits"] = logits           # [N, out_dim]

        # for convenience: expose a few internals when available
        if "M" in aux_last:
            out["M_last"] = aux_last["M"]    # [T, N, d]

        return out


def shape_check_demo() -> Tuple[torch.Size, Optional[torch.Size]]:
    """
    Minimal shape check (no training). Returns (Y_last_shape, logits_shape).
    """
    T, N, d_in, d, Hs, L = 16, 64, 32, 64, 4, 2
    E = N * 4
    H0 = torch.randn(T, N, d_in)
    edge_index = torch.randint(0, N, (2, E))
    time_idx = torch.arange(T)

    model = SpikeNetX(d_in=d_in, d=d, layers=L, heads=Hs, topk=8, W=8, out_dim=5)
    out = model(H0, edge_index=edge_index, time_idx=time_idx)
    return out["Y_last"].shape, out.get("logits", None).shape if "logits" in out else None

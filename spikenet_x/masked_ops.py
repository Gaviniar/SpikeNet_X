# -*- coding: utf-8 -*-
"""
掩码与 Top-k 相关的张量操作（纯 PyTorch 实现）

函数约定
--------
- 所有 logits/score 相关的函数在被掩蔽位置填充为 -inf（或非常负的数），
  再做 softmax，以确保数值与归一化正确。
- 所有张量形状均保持与输入一致，除非特别说明。

作者: Cline
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


NEG_INF = -1e30  # 作为 -inf 的数值近似，避免部分设备对 -inf 的不一致处理


def fill_masked_(logits: torch.Tensor, mask: torch.Tensor, value: float = NEG_INF) -> torch.Tensor:
    """
    原地将 mask==0 的位置填充为 value（默认近似 -inf）。

    参数
    ----
    logits : Float[...]
        任意形状的分数张量
    mask : Bool/Byte[...]
        与 logits 同形，True/1 表示可用，False/0 表示被掩蔽
    value : float
        被掩蔽位置写入的值

    返回
    ----
    logits : Float[...]
        与输入同一引用的张量（原地修改）
    """
    if mask.dtype != torch.bool:
        mask = mask != 0
    logits.masked_fill_(~mask, value)
    return logits


def masked_softmax(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    在给定维度上对带掩码的 logits 进行 softmax。

    - 先在掩蔽位置写入 -inf，再做 softmax，避免“先置零后归一”导致的数值偏差。
    - 支持温度缩放：logits / temperature

    参数
    ----
    logits : Float[...]
    mask : Bool/Byte[...] or None
        与 logits 广播兼容；为 None 时等价于全 True
    dim : int
    temperature : float

    返回
    ----
    probs : Float[...]
    """
    if temperature != 1.0:
        logits = logits / float(temperature)

    if mask is not None:
        # 为避免修改外部张量，做一份拷贝
        logits = logits.clone()
        fill_masked_(logits, mask, NEG_INF)

    # 数值稳定 softmax
    max_val, _ = torch.max(logits, dim=dim, keepdim=True)
    shifted = logits - max_val
    exp = torch.exp(shifted)
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask != 0
        exp = exp * mask.to(dtype=exp.dtype)

    denom = torch.clamp(exp.sum(dim=dim, keepdim=True), min=1e-12)
    return exp / denom


def topk_mask_logits(
    logits: torch.Tensor,
    k: int,
    dim: int = -1,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在维度 dim 上选出前 k 的元素，其余位置置为 -inf（或近似值）。

    注意：
    - 该函数只在 logits 上执行 Top-k 筛选，不做 softmax。
    - 返回 (new_logits, keep_mask)

    参数
    ----
    logits : Float[...]
    k : int
        k >= 1
    dim : int
    inplace : bool
        是否原地写回

    返回
    ----
    new_logits : Float[...]
        仅保留 Top-k 的 logits；其余位置为 -inf
    keep_mask : Bool[...]
        True 表示该位置被保留
    """
    assert k >= 1, "topk must be >= 1"
    # 取 Top-k 的阈值
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=dim)
    # 构造保留 mask
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask.scatter_(dim, topk_idx, True)

    if inplace:
        out = fill_masked_(logits, keep_mask, NEG_INF)
        return out, keep_mask
    else:
        new_logits = torch.where(keep_mask, logits, torch.full_like(logits, NEG_INF))
        return new_logits, keep_mask


def masked_topk_softmax(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    dim: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    组合操作：先对 logits 进行掩码，随后 Top-k 截断，再做 softmax。

    等价步骤：
      1) logits[~mask] = -inf
      2) 仅保留维度 dim 上的 Top-k，其余 = -inf
      3) softmax(dim)

    参数
    ----
    logits : Float[...]
    mask : Optional[Bool/Byte[...] ]
    k : int
    dim : int
    temperature : float

    返回
    ----
    probs : Float[...]
    """
    if mask is not None:
        logits = logits.clone()
        fill_masked_(logits, mask, NEG_INF)
    logits, _ = topk_mask_logits(logits, k=k, dim=dim, inplace=False)
    return masked_softmax(logits, mask=None, dim=dim, temperature=temperature)

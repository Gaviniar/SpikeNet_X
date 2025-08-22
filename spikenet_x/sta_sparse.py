# -*- coding: utf-8 -*-
"""
Sparse Spiking Temporal Attention（STA）——O(E) 稀疏实现（大图可用）

设计要点
--------
- 不构造 [N,N] 稠密矩阵，完全基于 edge_index = (src,dst) 按边计算；
- 时间因果窗口 W：对每个 (t,i) 仅聚合 t' ∈ [t-W, t] 的消息；
- 源端脉冲门控：在 logit 上加 log(S[t',src] + eps)，等价于概率缩放；
- 两遍 segment-softmax（数值稳定）：
    Pass-1：按接收端 dst 计算各头的 segment-wise amax（log-sum-exp 的 max 项）；
    Pass-2：重新计算 exp(score - amax(dst))，用 scatter_add 聚合分母/分子得到消息；
- 相对时间编码与可学习偏置 b[Δt] 由 RelativeTimeEncoding 复用；
- 注意：本稀疏版本当前不实现 Top-K 截断（dense 版本支持），必要时后续可加入“每 dst 流式 Top-K”。

复杂度
------
O(T * H * W * E)，显存主要为按 E 规模的临时张量。适用于大图/子图批训练。

接口
----
forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E], time_idx:Long[T]) -> M:[T,N,d]
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rel_time import RelativeTimeEncoding


def _has_scatter_reduce_tensor() -> bool:
    # PyTorch 1.12+ 提供 Tensor.scatter_reduce_
    return hasattr(torch.Tensor, "scatter_reduce_")


def _try_import_torch_scatter():
    try:
        import torch_scatter  # type: ignore
        return torch_scatter
    except Exception:
        return None


_TORCH_SCATTER = _try_import_torch_scatter()
_HAS_TSR = _has_scatter_reduce_tensor()


def _segment_amax_1d(x: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
    """
    计算 out[j] = max_{i: index[i]==j} x[i]，其中 j ∈ [0, K)
    优先使用 Tensor.scatter_reduce_('amax')；其次 torch_scatter.scatter_max；最终回退到排序段法。
    """
    device, dtype = x.device, x.dtype
    neg_inf = torch.tensor(-1e30, dtype=dtype, device=device)

    if _HAS_TSR:
        # 使用非 in-place 版本 scatter_reduce 修复梯度计算问题
        # include_self=False 时，空段结果未定义，需手动填充
        init_val = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
        out = init_val.scatter_reduce(0, index, x, reduce="amax", include_self=False)
        return out

    if _TORCH_SCATTER is not None:
        # torch_scatter.scatter_max 返回 (out, argmax)
        out, _ = _TORCH_SCATTER.scatter_max(x, index, dim=0, dim_size=K)
        # 对于空段，scatter_max 会给出 0；为一致性，将空段填为 -inf：
        # 通过统计计数判断空段
        cnt = torch.zeros(K, device=device, dtype=torch.long)
        cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
        out = torch.where(cnt > 0, out, neg_inf)
        return out

    # 回退：排序段法（可能较慢，但不依赖扩展）
    perm = torch.argsort(index)
    idx_s = index[perm]
    x_s = x[perm]
    out = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
    if idx_s.numel() == 0:
        return out
    # 找段边界
    boundary = torch.ones_like(idx_s, dtype=torch.bool)
    boundary[1:] = idx_s[1:] != idx_s[:-1]
    # 段起点位置
    starts = torch.nonzero(boundary, as_tuple=False).flatten()
    # 段终点（含）位置
    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = idx_s.numel() - 1
    # 逐段计算最大（Python 循环，仅在无高阶算子时作为兜底）
    for s, e in zip(starts.tolist(), ends.tolist()):
        j = int(idx_s[s].item())
        out[j] = torch.maximum(out[j], x_s[s : e + 1].max())
    return out


class SparseSpikingTemporalAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int,
        heads: int = 4,
        W: int = 64,
        use_rel_bias: bool = True,
        attn_drop: float = 0.1,
        temp: float = 1.0,
        # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8）
        pe_taus: Tuple[float, float] = (4.0, 16.0),
        pe_n_freq: int = 3,
        pe_num_buckets: int = 0,
    ) -> None:
        """
        稀疏 STA，不支持 Top-K 截断（如需可后续加入流式 Top-K）。
        """
        super().__init__()
        assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
        assert W >= 0, "W 必须 >= 0"

        self.d_in = int(d_in)
        self.d = int(d)
        self.heads = int(heads)
        self.d_head = self.d // self.heads
        self.W = int(W)
        self.use_rel_bias = bool(use_rel_bias)
        self.temp = float(temp)

        # 相对时间编码
        self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq, num_buckets=pe_num_buckets)
        d_pe = self.rel_enc.d_pe

        # 线性投影（K 拼接相对时间编码）
        self.W_q = nn.Linear(d_in, self.d, bias=False)
        self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
        self.W_v = nn.Linear(d_in, self.d, bias=False)

        # 注意力 dropout：实现为“边贡献丢弃再归一化”的无缩放 Bernoulli mask（训练时）
        self.p_drop = float(attn_drop)
        self.scale = self.d_head ** -0.5

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    @staticmethod
    def _check_edges(edge_index: torch.Tensor, N: int) -> None:
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
        E = edge_index.size(1)
        if E == 0:
            return
        assert int(edge_index.min()) >= 0 and int(edge_index.max()) < N, "edge_index 越界"

    def _edge_arrays(self, edge_index: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # 拆分 src, dst 为长度 E 的向量
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        return src, dst

    @torch.no_grad()
    def _drop_mask(self, shape: torch.Size, device: torch.device) -> Optional[torch.Tensor]:
        if not self.training or self.p_drop <= 0.0:
            return None
        return (torch.rand(shape, device=device) > self.p_drop)

    def forward(
        self,
        H_tilde: torch.Tensor,            # [T, N, d_in]
        S: torch.Tensor,                  # [T, N] in [0,1]
        edge_index: torch.Tensor,         # [2, E]
        time_idx: torch.Tensor,           # [T]
        adj_mask: Optional[torch.Tensor] = None,  # 兼容签名；稀疏实现忽略
    ) -> torch.Tensor:
        assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
        assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
        assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"

        T, N, Din = H_tilde.shape
        assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"

        device = H_tilde.device
        dtype = H_tilde.dtype

        # 校验边并拆分
        self._check_edges(edge_index, N)
        src, dst = self._edge_arrays(edge_index, device)  # [E], [E]
        E = src.numel()

        # 相对时间编码与偏置（仅构造 0..W）
        pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
        if not self.use_rel_bias:
            rel_bias = torch.zeros_like(rel_bias)

        # 预计算 Q(t, ·) 与 V(t, ·)
        Q_all = self.W_q(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()  # [T,H,N,d_h]
        V_all = self.W_v(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()  # [T,H,N,d_h]

        # 输出
        M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)

        eps_gate = 1.0e-6
        neg_inf = -1.0e30

        for t in range(T):
            W_eff = min(self.W, t)
            # Q_t: [H,N,d_h]
            Q_t = Q_all[t]  # [H,N,d_h]

            # -------- Pass-1：按 dst 计算 segment-wise amax（各头独立） --------
            max_dst_list = []
            for dt in range(W_eff + 1):
                t_prime = t - dt
                # 构造 K_{t'}（拼接相对时间编码）
                pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
                K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)  # [N, d_in+d_pe]
                K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]

                # 门控（源端脉冲）
                gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)  # [N]

                # 对每个头计算边打分：scores[h,e] = <Q_t[h,dst[e]], K_tp[h,src[e]]> * scale + b[dt] + log S
                # gather Q/K
                Q_d = Q_t[:, dst, :]              # [H,E,d_h]
                K_s = K_tp[:, src, :]             # [H,E,d_h]
                # 点积
                scores = (Q_d * K_s).sum(dim=-1) * self.scale  # [H,E]
                # 相对偏置
                scores = scores + float(rel_bias[dt])
                # 源脉冲门控
                scores = scores + gate_log[src]  # 广播到 [H,E]

                # softmax 温度
                if self.temp != 1.0:
                    scores = scores / float(self.temp)

                # 对每头做 segment amax
                m_h = torch.stack([_segment_amax_1d(scores[h], dst, N) for h in range(self.heads)])
                max_dst_list.append(m_h)

            max_dst = torch.stack(max_dst_list, dim=0).max(dim=0)[0]

            # -------- Pass-2：exp(score - amax(dst)) 聚合分母/分子 --------
            denom = torch.zeros((self.heads, N), device=device, dtype=dtype)          # [H,N]
            numer = torch.zeros((self.heads, N, self.d_head), device=device, dtype=dtype)  # [H,N,d_h]

            for dt in range(W_eff + 1):
                t_prime = t - dt
                pe = pe_table[dt].to(dtype=dtype, device=device)
                K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)  # [N, d_in+d_pe]
                K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
                V_tp = V_all[t_prime]  # [H,N,d_h]
                gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)  # [N]

                Q_d = Q_t[:, dst, :]          # [H,E,d_h]
                K_s = K_tp[:, src, :]         # [H,E,d_h]
                V_s = V_tp[:, src, :]         # [H,E,d_h]

                scores = (Q_d * K_s).sum(dim=-1) * self.scale  # [H,E]
                scores = scores + float(rel_bias[dt])
                scores = scores + gate_log[src]
                if self.temp != 1.0:
                    scores = scores / float(self.temp)

                # exp(score - max_dst[dst])
                # broadcast: max_dst[:, dst] -> [H,E]
                max_g = max_dst[:, dst]
                ex = torch.exp(scores - max_g)  # [H,E]

                # attention dropout：训练时对边贡献做伯努利丢弃（不做 1/(1-p) 缩放，随后自动归一化）
                mask = self._drop_mask(ex.shape, device=device)
                if mask is not None:
                    ex = ex * mask.to(dtype=ex.dtype)

                # 逐头 scatter_add 到 dst
                for h in range(self.heads):
                    # 分母
                    denom[h].index_add_(0, dst, ex[h])  # [N]
                    # 分子：ex[h][:,None] * V_s[h] 累加到 dst
                    contrib = ex[h].unsqueeze(-1) * V_s[h]  # [E,d_h]
                    # 将 [E,d_h] 累加到 [N,d_h]：循环通道（d_h 小，循环成本可接受）
                    # 向量化 index_add_ 仅支持 1D，这里按通道展开
                    for c in range(self.d_head):
                        numer[h, :, c].index_add_(0, dst, contrib[:, c])

            # 得到各头消息并合并
            # 防零保护
            denom = torch.clamp(denom, min=1e-12)
            msg_h = numer / denom.unsqueeze(-1)  # [H,N,d_h]
            M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
            M_out[t] = M_t

        return M_out  # [T,N,d]


__all__ = ["SparseSpikingTemporalAttention"]

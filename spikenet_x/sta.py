# -*- coding: utf-8 -*-
"""
SpikingTemporalAttention（STA）——稠密回退实现（小图/验证用）

功能
----
- 在因果与邻接掩码下，对 (邻居 j, 过去时间 t') 的键值进行多头注意力；
- 将源端脉冲 S[j,t'] 作为门控（在 logit 上加 log(S+eps) 等价于概率缩放）；
- 支持 Top-k 稀疏化（在 (j,t') 的联合候选维度上执行）；
- 支持相对时间编码与可学习相对偏置 b[Δt]；
- 输出每个 (t,i) 的聚合消息 M[t,i,:]，形状 [T, N, d]。

复杂度
------
Dense 回退：O(T * (W+1) * H * N^2)。在大图上请实现/切换稀疏边版本。

接口（与《提示词.md》一致）
------------------------
forward(H_tilde:[T,N,d_in], S:[T,N], edge_index:Long[2,E] 或 adj_mask:Bool[N,N],
        time_idx:Long[T]) -> M:[T,N,d]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_ops import masked_topk_softmax
from .rel_time import RelativeTimeEncoding


def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    # 形状检查
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 应为 [2, E]"
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    # --- 越界保护：确保传入的是“局部 id（0..num_nodes-1）” ---
    max_id = int(edge_index.max().item())
    min_id = int(edge_index.min().item())
    if max_id >= num_nodes or min_id < 0:
        raise RuntimeError(
            f"[dense adj] edge_index 越界：min={min_id}, max={max_id}, 但 num_nodes={num_nodes}。"
            "请确认子图边已映射为局部 id（0..N_sub-1），且 H0_subgraph 的 N 与之匹配。"
        )

    # 用 GPU 写稠密邻接（安全、快速）
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    adj[dst, src] = True
    return adj



class SpikingTemporalAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int,
        heads: int = 4,
        topk: int = 16,
        W: int = 64,
        use_rel_bias: bool = True,
        attn_drop: float = 0.1,
        temp: float = 1.0,
        # 相对时间编码配置（默认 d_pe = 2 + 2*3 = 8，符合提示词推荐）
        pe_taus: Tuple[float, float] = (4.0, 16.0),
        pe_n_freq: int = 3,
        pe_num_buckets: int = 0,
    ) -> None:
        super().__init__()
        assert heads >= 1 and d % heads == 0, "heads*d_head 必须等于 d"
        assert topk >= 1, "topk 必须 >= 1"
        assert W >= 0, "W 必须 >= 0"

        self.d_in = int(d_in)
        self.d = int(d)
        self.heads = int(heads)
        self.d_head = self.d // self.heads
        self.topk = int(topk)
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

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop and attn_drop > 0 else nn.Identity()
        self.scale = self.d_head ** -0.5

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    @staticmethod
    def _build_adj_mask(
        N: int,
        edge_index: Optional[torch.Tensor],
        adj_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        构造邻接掩码 A: Bool[N,N]，A[i,j]=True 表示 j->i 存在边（j 属于 i 的邻居）
        """
        if adj_mask is not None:
            A = adj_mask.to(device=device, dtype=torch.bool)
            assert A.shape == (N, N), f"adj_mask 形状应为 [{N},{N}]"
            return A
        assert edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 2, \
            "未提供 adj_mask 时必须提供 edge_index，形状 [2,E]"
        return _edge_index_to_dense_adj(edge_index.to(device), N, device=device)

    def forward(
        self,
        H_tilde: torch.Tensor,            # [T, N, d_in]
        S: torch.Tensor,                  # [T, N] in [0,1]
        edge_index: Optional[torch.Tensor],  # [2, E] 或 None
        time_idx: torch.Tensor,           # [T]
        adj_mask: Optional[torch.Tensor] = None,  # [N, N] Bool 或 None
    ) -> torch.Tensor:
        assert H_tilde.dim() == 3, "H_tilde 形状应为 [T, N, d_in]"
        assert S.dim() == 2 and S.shape[:2] == H_tilde.shape[:2], "S 与 H_tilde 的 [T,N] 必须一致"
        assert time_idx.dim() == 1 and time_idx.numel() == H_tilde.size(0), "time_idx 形状应为 [T] 且与 T 一致"

        T, N, Din = H_tilde.shape
        assert Din == self.d_in, f"d_in 不匹配：期望 {self.d_in}, 实得 {Din}"

        device = H_tilde.device
        dtype = H_tilde.dtype

        # 邻接掩码（稠密回退）
        A = self._build_adj_mask(N, edge_index, adj_mask, device)  # [N,N] Bool

        # 相对时间编码与（可选）偏置（仅构造 0..W）
        pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)  # pe:[W+1,d_pe], bias:[W+1]
        if not self.use_rel_bias:
            rel_bias = torch.zeros_like(rel_bias)

        # 预计算所有时刻的 Q、V
        Q_all = self.W_q(H_tilde)  # [T, N, d]
        V_all = self.W_v(H_tilde)  # [T, N, d]

        # 输出容器
        M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)

        eps_gate = 1.0e-6

        for t in range(T):
            W_eff = min(self.W, t)

            # 多头视图
            # Q_t: [N, H, d_h] -> 转为 [H, N, d_h] 便于后续计算
            Q_t = Q_all[t].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]

            logits_chunks = []   # 每个块 [H,N,N]
            mask_chunks = []     # 每个块 [H,N,N]
            V_chunks = []        # 每个块 [H,N,d_h]
            gate_chunks = []     # 每个块 [1,1,N]（用于 log-domain 加法；在拼接后按候选展开）

            for dt in range(W_eff + 1):
                t_prime = t - dt

                # 相对时间编码拼接到 K 输入
                pe = pe_table[dt].to(dtype=dtype, device=device)  # [d_pe]
                pe_expand = pe.view(1, 1, -1).expand(N, -1, -1)   # [N,1,d_pe] -> 与 H_tilde[t'] 拼接
                K_in = torch.cat([H_tilde[t_prime], pe_expand.squeeze(1)], dim=-1)  # [N, d_in + d_pe]

                # 线性映射并切分多头
                K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]
                V_tp = V_all[t_prime].view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()  # [H,N,d_h]

                # 注意力 logits（缩放点积）
                # scores[h,i,j] = <Q_t[h,i,:], K_tp[h,j,:]>
                scores = torch.einsum("hid,hjd->hij", Q_t, K_tp) * self.scale  # [H,N,N]

                # 相对偏置共享到 (i,j)
                if rel_bias is not None:
                    scores = scores + float(rel_bias[dt])

                # 源端脉冲门控（log-domain 加法）
                gate_j = torch.clamp(S[t_prime], 0.0, 1.0).to(dtype=dtype)  # [N]
                gate_chunks.append(torch.log(gate_j + eps_gate).view(1, 1, N))  # [1,1,N]

                # 邻接掩码广播到各头
                mask_hij = A.view(1, N, N).expand(self.heads, -1, -1)  # [H,N,N]

                logits_chunks.append(scores)
                mask_chunks.append(mask_hij)
                V_chunks.append(V_tp)

            if not logits_chunks:
                # 该 t 无可用键（仅 t=0 且 W=0 时可能发生）
                continue

            # 拼接候选维（按 dt 依次拼接）
            # logits_flat: [H,N,(W_eff+1)*N]
            logits_flat = torch.cat(logits_chunks, dim=2)
            mask_flat = torch.cat(mask_chunks, dim=2)  # [H,N,(W_eff+1)*N]
            gate_log_flat = torch.cat(
                [g.expand(self.heads, N, -1) for g in gate_chunks], dim=2
            )  # [H,N,(W_eff+1)*N]
            logits_flat = logits_flat + gate_log_flat

            # Top-k + masked softmax（温度缩放在函数内部）
            k_eff = min(self.topk, logits_flat.size(-1))
            probs = masked_topk_softmax(
                logits_flat, mask_flat, k=k_eff, dim=-1, temperature=self.temp
            )  # [H,N,(W_eff+1)*N]
            probs = self.attn_drop(probs)
            
            torch.nan_to_num_(probs, nan=0.0)
            
            # 构造对应的值向量拼接：V_cat: [H,(W_eff+1)*N,d_h]
            V_cat = torch.cat(V_chunks, dim=1)  # [H, (W_eff+1)*N, d_h]

            # 聚合：msg_h = probs @ V_cat
            msg_h = torch.einsum("hni,hid->hnd", probs, V_cat)  # [H,N,d_h]

            # 合并头并写入输出
            M_t = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)  # [N,d]
            M_out[t] = M_t

        return M_out  # [T,N,d]

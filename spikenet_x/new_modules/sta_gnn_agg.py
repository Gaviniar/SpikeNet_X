# spikenet_x/new_modules/sta_gnn_agg.py
# 内容基本迁移自 sta_sparse.py，并重命名类

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..rel_time import RelativeTimeEncoding

# Helper functions from sta_sparse.py (for segment_amax)
def _has_scatter_reduce_tensor() -> bool:
    return hasattr(torch.Tensor, "scatter_reduce_")

def _try_import_torch_scatter():
    try:
        import torch_scatter
        return torch_scatter
    except Exception:
        return None

_TORCH_SCATTER = _try_import_torch_scatter()
_HAS_TSR = _has_scatter_reduce_tensor()

def _segment_amax_1d(x: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    neg_inf = torch.tensor(-1e30, dtype=dtype, device=device)
    if _HAS_TSR:
        init_val = torch.full((K,), neg_inf.item(), device=device, dtype=dtype)
        out = init_val.scatter_reduce(0, index, x, reduce="amax", include_self=False)
        return out
    if _TORCH_SCATTER is not None:
        out, _ = _TORCH_SCATTER.scatter_max(x, index, dim=0, dim_size=K)
        cnt = torch.zeros(K, device=device, dtype=torch.long)
        cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
        out = torch.where(cnt > 0, out, neg_inf)
        return out
    raise ImportError("STAGNNAggregator requires either PyTorch >= 1.12 or torch_scatter.")


class STAGNNAggregator(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int,
        heads: int = 4,
        W: int = 32,
        use_rel_bias: bool = True,
        attn_drop: float = 0.1,
        temp: float = 1.0,
        pe_taus: Tuple[float, float] = (4.0, 16.0),
        pe_n_freq: int = 3,
    ) -> None:
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
        self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq)
        d_pe = self.rel_enc.d_pe
        self.W_q = nn.Linear(d_in, self.d, bias=False)
        self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
        self.W_v = nn.Linear(d_in, self.d, bias=False)
        self.p_drop = float(attn_drop)
        self.scale = self.d_head ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    @staticmethod
    def _check_edges(edge_index: torch.Tensor, N: int) -> None:
        if edge_index.numel() > 0:
            assert int(edge_index.min()) >= 0 and int(edge_index.max()) < N, "edge_index 越界"

    def _edge_arrays(self, edge_index: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        H_tilde: torch.Tensor,
        S: torch.Tensor,
        edge_index: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        T, N, Din = H_tilde.shape
        device, dtype = H_tilde.device, H_tilde.dtype
        self._check_edges(edge_index, N)
        src, dst = self._edge_arrays(edge_index, device)
        pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)
        if not self.use_rel_bias:
            rel_bias = torch.zeros_like(rel_bias)

        Q_all = self.W_q(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()
        V_all = self.W_v(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3).contiguous()
        M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
        eps_gate = 1.0e-6

        for t in range(T):
            W_eff = min(self.W, t)
            Q_t = Q_all[t]
            
            max_dst_list = []
            for dt in range(W_eff + 1):
                t_prime = t - dt
                pe = pe_table[dt].to(dtype=dtype, device=device)
                K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)
                K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()
                gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)
                Q_d, K_s = Q_t[:, dst, :], K_tp[:, src, :]
                scores = (Q_d * K_s).sum(dim=-1) * self.scale + float(rel_bias[dt]) + gate_log[src]
                if self.temp != 1.0: scores = scores / float(self.temp)
                m_h = torch.stack([_segment_amax_1d(scores[h], dst, N) for h in range(self.heads)])
                max_dst_list.append(m_h)
            
            max_dst = torch.stack(max_dst_list, dim=0).max(dim=0)[0] if max_dst_list else torch.full((self.heads, N), -1e30, device=device, dtype=dtype)

            denom = torch.zeros((self.heads, N), device=device, dtype=dtype)
            numer = torch.zeros((self.heads, N, self.d_head), device=device, dtype=dtype)

            for dt in range(W_eff + 1):
                t_prime = t - dt
                pe = pe_table[dt].to(dtype=dtype, device=device)
                K_in = torch.cat([H_tilde[t_prime], pe.view(1, -1).expand(N, -1)], dim=-1)
                K_tp = self.W_k(K_in).view(N, self.heads, self.d_head).permute(1, 0, 2).contiguous()
                V_tp = V_all[t_prime]
                gate_log = torch.log(torch.clamp(S[t_prime], 0.0, 1.0) + eps_gate).to(dtype=dtype)
                Q_d, K_s, V_s = Q_t[:, dst, :], K_tp[:, src, :], V_tp[:, src, :]
                scores = (Q_d * K_s).sum(dim=-1) * self.scale + float(rel_bias[dt]) + gate_log[src]
                if self.temp != 1.0: scores = scores / float(self.temp)
                
                max_g = max_dst[:, dst]
                ex = torch.exp(scores - max_g)
                mask = self._drop_mask(ex.shape, device=device)
                if mask is not None: ex = ex * mask.to(dtype=ex.dtype)

                for h in range(self.heads):
                    denom[h].index_add_(0, dst, ex[h])
                    contrib = ex[h].unsqueeze(-1) * V_s[h]
                    for c in range(self.d_head):
                        numer[h, :, c].index_add_(0, dst, contrib[:, c])

            denom = torch.clamp(denom, min=1e-12)
            msg_h = numer / denom.unsqueeze(-1)
            M_out[t] = msg_h.permute(1, 0, 2).contiguous().view(N, self.d)
            
        return M_out

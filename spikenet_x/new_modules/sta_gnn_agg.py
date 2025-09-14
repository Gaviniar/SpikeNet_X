# spikenet_x/new_modules/sta_gnn_agg.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..rel_time import RelativeTimeEncoding

# Helper functions for segment_amax 
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
        self.d_in, self.d, self.heads, self.d_head, self.W = d_in, d, heads, d // heads, W
        self.use_rel_bias, self.temp, self.p_drop = use_rel_bias, temp, attn_drop
        
        self.rel_enc = RelativeTimeEncoding(taus=pe_taus, n_freq=pe_n_freq)
        d_pe = self.rel_enc.d_pe
        self.W_q = nn.Linear(d_in, self.d, bias=False)
        self.W_k = nn.Linear(d_in + d_pe, self.d, bias=False)
        self.W_v = nn.Linear(d_in, self.d, bias=False)
        self.scale = self.d_head ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(
        self,
        H_tilde: torch.Tensor,
        S: torch.Tensor,
        edge_index: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        T, N, _ = H_tilde.shape
        device, dtype = H_tilde.device, H_tilde.dtype
        src, dst = edge_index[0].to(device), edge_index[1].to(device)
        E = src.numel()

        pe_table, rel_bias = self.rel_enc(time_idx.to(device), W=self.W)
        if not self.use_rel_bias:
            rel_bias = torch.zeros_like(rel_bias)

        Q_all = self.W_q(H_tilde).view(T, N, self.heads, self.d_head).permute(0, 2, 1, 3)
        
        M_out = torch.zeros((T, N, self.d), device=device, dtype=dtype)
        eps_gate = 1.0e-6

        for t in range(T):
            if E == 0: continue
            
            W_eff = min(self.W, t)
            dt_range = torch.arange(W_eff + 1, device=device)
            t_prime_range = t - dt_range

            Q_d_t = Q_all[t, :, dst, :]
            
            H_src_window = H_tilde[t_prime_range][:, src, :]
            pe_window = pe_table[:W_eff+1].unsqueeze(1).expand(-1, E, -1)
            
            K_in_flat = torch.cat([H_src_window, pe_window], dim=-1).view(-1, self.d_in + self.rel_enc.d_pe)
            K_s_window = self.W_k(K_in_flat).view(W_eff + 1, E, self.heads, self.d_head).permute(2, 0, 1, 3)
            
            V_s_window = self.W_v(H_src_window).view(W_eff + 1, E, self.heads, self.d_head).permute(2, 0, 1, 3)

            gate_log_window = torch.log(S[t_prime_range][:, src] + eps_gate)
            rel_bias_window = rel_bias[:W_eff+1]

            scores = torch.einsum('hed,hwed->hwe', Q_d_t, K_s_window) * self.scale
            
            ### --- MODIFICATION START: Replaced in-place additions --- ###
            scores = scores + rel_bias_window.view(1, -1, 1)
            scores = scores + gate_log_window.view(1, -1, E)
            ### --- MODIFICATION END --- ###
            
            if self.temp != 1.0: scores /= self.temp
            
            scores_flat = scores.reshape(self.heads, -1)
            dst_expanded = dst.repeat(W_eff + 1)

            max_scores_per_dst = torch.stack([_segment_amax_1d(s, dst_expanded, N) for s in scores_flat])
            max_g = max_scores_per_dst[:, dst_expanded]
            ex = torch.exp(scores_flat - max_g)

            if self.training and self.p_drop > 0:
                drop_mask = (torch.rand_like(ex) > self.p_drop).to(dtype)
                ### --- MODIFICATION START: Replaced in-place multiplication --- ###
                ex = ex * drop_mask
                ### --- MODIFICATION END --- ###

            V_flat = V_s_window.reshape(self.heads, -1, self.d_head)
            
            numer_flat = ex.unsqueeze(-1) * V_flat
            numer = torch.zeros((self.heads, N, self.d_head), device=device, dtype=dtype)
            for h in range(self.heads):
                for c in range(self.d_head):
                    numer[h, :, c].index_add_(0, dst_expanded, numer_flat[h, :, c])

            denom = torch.zeros((self.heads, N), device=device, dtype=dtype)
            for h in range(self.heads):
                denom[h].index_add_(0, dst_expanded, ex[h])
            
            denom = torch.clamp(denom, min=1e-12).unsqueeze(-1)
            msg_h = numer / denom
            M_out[t] = msg_h.permute(1, 0, 2).reshape(N, self.d)

        return M_out

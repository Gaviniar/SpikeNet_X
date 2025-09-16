# spikenet_x/new_modules/sta_gnn_agg_optimized.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from ..rel_time import RelativeTimeEncoding

# Helper function to handle vectorized scatter_add for multi-dimensional tensors
def _segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Vectorized segment sum using scatter_add_."""
    result_shape = (num_segments,) + data.shape[1:]
    result = torch.zeros(result_shape, dtype=data.dtype, device=data.device)
    # expand segment_ids to match data's dimensions for scatter_add_
    view_shape = (segment_ids.shape[0],) + (1,) * (data.dim() - 1)
    segment_ids = segment_ids.view(view_shape).expand_as(data)
    result.scatter_add_(0, segment_ids, data)
    return result

class STAGNNAggregator_Optimized(nn.Module):
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
        if E == 0:
            return torch.zeros((T, N, self.d), device=device, dtype=dtype)

        pe_table, rel_bias = self.rel_enc(time_idx, W=self.W)
        if not self.use_rel_bias:
            rel_bias = torch.zeros_like(rel_bias)

        # --- Vectorized Index Generation ---
        t_coords = torch.arange(T, device=device).view(-1, 1)
        dt_coords = torch.arange(self.W + 1, device=device).view(1, -1)
        
        t_prime_matrix = t_coords - dt_coords
        valid_mask = t_prime_matrix >= 0
        
        t_indices, dt_indices = valid_mask.nonzero(as_tuple=True)
        t_prime_indices = t_prime_matrix[t_indices, dt_indices]
        
        num_interactions = len(t_indices)
        
        t_indices_exp = t_indices.repeat_interleave(E)
        dt_indices_exp = dt_indices.repeat_interleave(E)
        t_prime_indices_exp = t_prime_indices.repeat_interleave(E)
        
        src_exp = src.repeat(num_interactions)
        dst_exp = dst.repeat(num_interactions)

        # --- Vectorized Attention Calculation ---
        Q = self.W_q(H_tilde).view(T, N, self.heads, self.d_head)
        V = self.W_v(H_tilde).view(T, N, self.heads, self.d_head)

        Q_gathered = Q[t_indices_exp, dst_exp]
        
        H_k_gathered = H_tilde[t_prime_indices_exp, src_exp]
        pe_gathered = pe_table[dt_indices_exp]
        K_in = torch.cat([H_k_gathered, pe_gathered], dim=-1)
        K_gathered = self.W_k(K_in).view(-1, self.heads, self.d_head)

        V_gathered = V[t_prime_indices_exp, src_exp]

        scores = torch.einsum('ehd,ehd->eh', Q_gathered, K_gathered) * self.scale
        
        # 使用非原地操作 (out-of-place)
        scores = scores + rel_bias[dt_indices_exp].unsqueeze(-1)
        
        eps_gate = 1e-6
        spike_gate = S[t_prime_indices_exp, src_exp]
        
        spike_gate_per_head = spike_gate.view(-1, self.heads, self.d_head)
        scalar_gate_per_head = spike_gate_per_head.mean(dim=-1)
        scores = scores + torch.log(scalar_gate_per_head + eps_gate)
        
        if self.temp != 1.0:
            scores = scores / self.temp
            
        # 5. Numerically stable softmax (segment-wise)
        segment_ids = t_indices_exp * N + dst_exp
        num_segments = T * N
        
        max_scores = torch.full((num_segments, self.heads), -1e30, device=device, dtype=dtype)
        max_scores.scatter_reduce_(0, segment_ids.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=False)

        scores_normalized = torch.exp(scores - max_scores[segment_ids])
        
        if self.training and self.p_drop > 0:

            scores_normalized = scores_normalized * (torch.rand_like(scores_normalized) > self.p_drop)

        denom = _segment_sum(scores_normalized, segment_ids, num_segments)
        
        numer_contrib = scores_normalized.unsqueeze(-1) * V_gathered
        numer = _segment_sum(numer_contrib, segment_ids, num_segments)
        
        M_flat = numer / torch.clamp(denom, min=1e-12).unsqueeze(-1)
        M_out = M_flat.reshape(T, N, self.d)
        
        return M_out

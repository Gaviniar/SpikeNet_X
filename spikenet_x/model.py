# spikenet_x/model.py
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .spiketdanet_layer import SpikeTDANetLayer

class SpikeTDANet(nn.Module):
    """
    Spike-TDANet: A Spiking Temporal Delay Attention Network.
    This model stacks SpikeTDANetLayer blocks.
    """
    # [MODIFIED] 更新__init__以传递LIF参数
    def __init__(
        self,
        d_in: int,
        d: int,
        layers: int = 2,
        heads: int = 4,
        W: int = 32,
        out_dim: Optional[int] = None,
        readout: str = "mean",
        # LIF Hyperparameters
        lif_tau: float = 0.95,
        lif_v_threshold: float = 1.0,
        lif_alpha: float = 1.0,
        lif_surrogate: str = 'sigmoid',
        **kwargs
    ) -> None:
        super().__init__()
        assert layers >= 1
        assert readout in ("last", "mean")

        self.readout = readout
        self.d_in = d_in
        self.d = d
        
        self.input_proj = nn.Linear(d_in, d)
        
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(SpikeTDANetLayer(
                channels=d, heads=heads, W=W, 
                lif_tau=lif_tau, lif_v_threshold=lif_v_threshold,
                lif_alpha=lif_alpha, lif_surrogate=lif_surrogate,
                **kwargs
            ))

        self.head = nn.Linear(d, out_dim) if out_dim is not None else None

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        features = self.input_proj(H)
        spikes = None  # 第一层没有前序脉冲
        
        spike_list = []

        for layer in self.layers:
            # [MODIFIED] spikes的形状在层间传递时是[T, N]
            features, spikes = layer(features, spikes, edge_index, time_idx)
            spike_list.append(spikes)

        if self.readout == "last":
            z = features[-1]
        else:
            z = features.mean(dim=0)

        logits = self.head(z) if self.head is not None else None

        out: Dict[str, torch.Tensor] = {
            "repr": z,
            "Y_last": features,
            "S_list": torch.stack(spike_list) if spike_list else torch.empty(0),
        }
        if logits is not None:
            out["logits"] = logits

        return out

# -*- coding: utf-8 -*-
"""
SpikeNet-X 最小可运行示例

运行方法：
    python -m spikenet_x.minimal_example
"""

import torch

# 动态地添加 spikenet_x 包的父目录到 sys.path
# 以便在 spikenet_x 目录外也能运行此脚本（例如从项目根目录）
import os
import sys
if __package__ is None or __package__ == '':
    # a bit of a hack to get relative imports working when running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from spikenet_x.model import SpikeNetX
else:
    from .model import SpikeNetX


def erdos_renyi_edge_index(num_nodes: int, p: float = 0.1, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand((num_nodes, num_nodes), generator=g) < p
    # 去除自环
    mask.fill_diagonal_(False)
    src, dst = mask.nonzero(as_tuple=True)
    # edge_index: [2, E]，列为 (src, dst)
    return torch.stack([src, dst], dim=0).to(torch.long)


def main():
    print("--- SpikeNet-X Minimal Example & Shape Check ---")
    
    T, N, d_in, d, Hs, L = 16, 64, 32, 64, 4, 2
    E = N * 5

    print(f"Params: T={T}, N={N}, d_in={d_in}, d={d}, heads={Hs}, layers={L}")

    H0 = torch.randn(T, N, d_in)
    edge_index = erdos_renyi_edge_index(N, p=0.05, seed=1)
    time_idx = torch.arange(T)

    print(f"Input shapes: H={H0.shape}, edge_index={edge_index.shape}, time_idx={time_idx.shape}")

    model = SpikeNetX(
        d_in=d_in,
        d=d,
        layers=L,
        heads=Hs,
        topk=8,
        W=8,
        attn_impl="sparse",
        out_dim=5,
    )
    model.eval() # 禁用 dropout
    print(f"\nModel:\n{model}\n")

    with torch.no_grad():
        out = model(H0, edge_index=edge_index, time_idx=time_idx)

    print("--- Output Shape Check ---")
    print(f"Y_last (final features): {out['Y_last'].shape}")
    print(f"S_list (spikes per layer): {out['S_list'].shape}")
    print(f"V_list (voltages per layer): {out['V_list'].shape}")
    print(f"logits (readout): {out['logits'].shape if out.get('logits') is not None else 'N/A'}")
    print(f"M_last (last layer msg): {out['M_last'].shape}")

    # 检查形状是否符合预期
    assert out["Y_last"].shape == (T, N, d)
    assert out["S_list"].shape == (L, T, N)
    assert out["V_list"].shape == (L, T, N)
    if out.get("logits") is not None:
        assert out["logits"].shape == (N, 5)

    print("\n✅ All shapes are correct.")


if __name__ == "__main__":
    main()

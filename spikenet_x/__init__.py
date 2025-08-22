# -*- coding: utf-8 -*-
"""
SpikeNet-X package

Exports the core building blocks specified in `提示词.md`:
- LearnableDelayLine
- SpikingTemporalAttention (dense fallback implementation)
- LIFCell
- SpikeNetXLayer
- Masked ops helpers and RelativeTimeEncoding
"""

from .masked_ops import (
    masked_softmax,
    masked_topk_softmax,
    topk_mask_logits,
    fill_masked_,
    NEG_INF,
)
from .rel_time import RelativeTimeEncoding
from .delayline import LearnableDelayLine
from .sta import SpikingTemporalAttention
from .sta_sparse import SparseSpikingTemporalAttention
from .lif_cell import LIFCell
from .spikenetx_layer import SpikeNetXLayer

__all__ = [
    "masked_softmax",
    "masked_topk_softmax",
    "topk_mask_logits",
    "fill_masked_",
    "NEG_INF",
    "RelativeTimeEncoding",
    "LearnableDelayLine",
    "SpikingTemporalAttention",
    "SparseSpikingTemporalAttention",
    "LIFCell",
    "SpikeNetXLayer",
]

"""GNN-based subgraph completion utilities for LightRAG.

This package is intentionally optional: importing it does not require PyTorch
or PyG. The training path raises a clear error if either dependency is
unavailable.
"""

from .config import SubgraphCompletionConfig
from .miner import LightRAGSubgraphCompletionMiner, augment_lightrag_with_subgraph_completion

__all__ = [
    "SubgraphCompletionConfig",
    "LightRAGSubgraphCompletionMiner",
    "augment_lightrag_with_subgraph_completion",
]

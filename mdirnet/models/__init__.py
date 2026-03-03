"""
Model modules for MDIRNET
- PPM: Patch Partitioning Module (content-aware patch sampling)
- DRAM: Dynamic Rank Allocation Module (adaptive rank prediction)
- DUOVPCA: Deep Unfolded OVPCA core (iterative restoration)
- SAM: Supervised Attention Module (refinement)
- MDIRNET: Main network combining all modules
"""

from .ppm import PatchPartitioningModule
from .dram import DynamicRankAllocationModule
from .du_ovpca import DUOVPCA, OVPCAIteration
from .sam import SupervisedAttentionModule
from .mdirnet import MDIRNET

__all__ = [
    'PatchPartitioningModule',
    'DynamicRankAllocationModule',
    'DUOVPCA',
    'OVPCAIteration',
    'SupervisedAttentionModule',
    'MDIRNET'
]
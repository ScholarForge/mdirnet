import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppm import PatchPartitioningModule
from .dram import DynamicRankAllocationModule
from .du_ovpca import DUOVPCA
from .sam import SupervisedAttentionModule


class MDIRNET(nn.Module):
    """
    Multi-Degradation Image Restoration Network (MDIRNET)
    
    Unified framework for restoring images affected by multiple degradations.
    Combines PPM, DRAM, DU-OVPCA, and SAM modules.
    """
    
    def __init__(self,
                 in_channels=3,
                 patch_size=64,
                 num_patch_groups=8,
                 patches_per_group=16,
                 num_ovpca_iterations=6,
                 dram_min_rank=4,
                 dram_max_rank=32,
                 use_ppm=True,
                 use_dram=True,
                 use_sam=True):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patch_groups = num_patch_groups
        self.patches_per_group = patches_per_group
        self.use_ppm = use_ppm
        self.use_dram = use_dram
        self.use_sam = use_sam
        
        # Patch Partitioning Module
        if use_ppm:
            self.ppm = PatchPartitioningModule(
                in_channels=in_channels,
                patch_size=patch_size
            )
        
        # Dynamic Rank Allocation Module
        if use_dram:
            self.dram = DynamicRankAllocationModule(
                in_channels=in_channels,
                r_min=dram_min_rank,
                r_max=dram_max_rank,
                patch_size=patch_size
            )
        
        # Deep Unfolded OVPCA Core
        self.du_ovpca = DUOVPCA(
            num_iterations=num_ovpca_iterations,
            max_rank=dram_max_rank
        )
        
        # Supervised Attention Module
        if use_sam:
            self.sam = SupervisedAttentionModule(
                in_channels=in_channels
            )
        
        # Patch aggregation weights
        self.aggregation_weights = nn.Parameter(
            torch.ones(patches_per_group) / patches_per_group
        )
        

        self.global_refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input degraded image [B, C, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            restored_img: Restored image [B, C, H, W]

        """
        B, C, H, W = x.shape
        device = x.device
        
        # Step 1: Adaptive patch extraction (PPM)
        if self.use_ppm:
            patches, flow_field, deformed_grid = self.ppm(x)
            # Reshape patches into groups
            patch_groups = self._group_patches(patches)
        else:
            # Fixed grid patch extraction
            patches = self._extract_fixed_patches(x)
            patch_groups = self._group_patches(patches)
        
        # Step 2: Dynamic rank allocation (DRAM)
        ranks = []
        if self.use_dram:
            for i, group in enumerate(patch_groups):
                rank, _ = self.dram(group.unsqueeze(0))
                ranks.append(rank)
        else:
            # Fixed rank
            ranks = [torch.tensor(16, device=device) for _ in patch_groups]
        
        # Step 3: Low-rank restoration via DU-OVPCA
        restored_groups = []
        for i, (group, rank) in enumerate(zip(patch_groups, ranks)):
            # group: [1, T, C, H, W]
            restored = self.du_ovpca(group, rank)
            restored_groups.append(restored)
        
        # Step 4: Patch aggregation
        initial_restored = self._aggregate_patches(restored_groups, (H, W))
        
        # Step 5: Supervised attention refinement (SAM)
        if self.use_sam:
            refined_img, attention_map = self.sam(x, initial_restored)
        else:
            refined_img = initial_restored
            attention_map = None
        
        final_img = self.global_refine(refined_img) + refined_img
        
        if return_attention:
            return final_img, {
                'attention_map': attention_map,
                'ranks': ranks,
                'flow_field': flow_field if self.use_ppm else None
            }
        
        return final_img
    
    def _extract_fixed_patches(self, x):
        """Extract patches using fixed grid"""
        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size // 2,
            padding=self.patch_size // 2
        )
        # Reshape to [B, num_patches, C, patch_size, patch_size]
        B, C_flat, num_patches = patches.shape
        patches = patches.view(
            B, -1, self.patch_size, self.patch_size, num_patches
        ).permute(0, 4, 1, 2, 3)
        return patches
    
    def _group_patches(self, patches):
        """Group patches based on similarity"""
        B, num_patches, C, H, W = patches.shape
        
        groups = []
        patches_per_group = num_patches // self.num_patch_groups
        
        for i in range(self.num_patch_groups):
            start_idx = i * patches_per_group
            end_idx = (i + 1) * patches_per_group
            group = patches[:, start_idx:end_idx]
            
            if group.shape[1] < patches_per_group:
                pad_size = patches_per_group - group.shape[1]
                padding = group[:, -1:].repeat(1, pad_size, 1, 1, 1)
                group = torch.cat([group, padding], dim=1)
            else:
                group = group[:, :patches_per_group]
            
            groups.append(group)
        
        return groups
    
    def _aggregate_patches(self, restored_groups, target_size):
        """Aggregate overlapping patches into full image"""
        B = restored_groups[0].shape[0]
        C = restored_groups[0].shape[2]
        H, W = target_size
        
        # Initialize accumulator and weight counter
        accumulator = torch.zeros(B, C, H, W, device=restored_groups[0].device)
        weight_map = torch.zeros(B, 1, H, W, device=restored_groups[0].device)
    
        for group_idx, group in enumerate(restored_groups):
            for patch_idx, patch in enumerate(group[0]):  # [C, H, W]
                y = (group_idx * self.patch_size) % H
                x = ((group_idx + patch_idx) * self.patch_size) % W
                
                if y + self.patch_size <= H and x + self.patch_size <= W:
                    accumulator[:, :, y:y+self.patch_size, x:x+self.patch_size] += patch
                    weight_map[:, :, y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Normalize
        weight_map = weight_map.clamp(min=1)
        aggregated = accumulator / weight_map
        
        return aggregated
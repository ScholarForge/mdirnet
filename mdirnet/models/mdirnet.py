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

    def __init__(
        self,
        in_channels=3,
        patch_size=64,
        num_patch_groups=8,
        patches_per_group=16,
        num_ovpca_iterations=6,
        dram_min_rank=4,
        dram_max_rank=32,
        use_ppm=True,
        use_dram=True,
        use_sam=True
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patch_groups = num_patch_groups
        self.patches_per_group = patches_per_group
        self.use_ppm = use_ppm
        self.use_dram = use_dram
        self.use_sam = use_sam
        self.in_channels = in_channels

        # Patch Partitioning Module
        if self.use_ppm:
            self.ppm = PatchPartitioningModule(
                in_channels=in_channels,
                patch_size=patch_size
            )

        # Dynamic Rank Allocation Module
        if self.use_dram:
            self.dram = DynamicRankAllocationModule(
                in_channels=in_channels,
                r_min=dram_min_rank,
                r_max=dram_max_rank,
                patch_size=patch_size
            )

        # Deep Unfolded OVPCA Core
        self.du_ovpca = DUOVPCA(
            num_iterations=num_ovpca_iterations,
            max_rank=dram_max_rank,
            out_channels=in_channels
        )

        # Supervised Attention Module
        if self.use_sam:
            self.sam = SupervisedAttentionModule(
                in_channels=in_channels
            )

        # Learnable patch aggregation weights
        self.aggregation_weights = nn.Parameter(
            torch.ones(patches_per_group, dtype=torch.float32) / max(patches_per_group, 1)
        )

        # Global refinement after SAM
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
            restored image [B, C, H, W]
        """
        assert x.dim() == 4, f"Expected input [B, C, H, W], got {x.shape}"

        B, C, H, W = x.shape
        device = x.device

        flow_field = None
        deformed_grid = None

        # Step 1: Patch extraction
        if self.use_ppm:
            patches, flow_field, deformed_grid = self.ppm(x)
        else:
            patches = self._extract_fixed_patches(x)

        # patches: [B, num_patches, C, patch_size, patch_size]
        patch_groups = self._group_patches(patches)

        # Step 2: Dynamic rank allocation
        ranks = []
        complexities = []

        if self.use_dram:
            for group in patch_groups:
                # group already has shape [B, T, C, H, W]
                rank, complexity = self.dram(group)
                ranks.append(rank)                  # [B]
                complexities.append(complexity)    # [B, 1]
        else:
            fixed_rank = min(16, self.du_ovpca.max_rank, self.patches_per_group, self.patch_size * self.patch_size * C)
            ranks = [
                torch.full((B,), fixed_rank, device=device, dtype=torch.long)
                for _ in patch_groups
            ]
            complexities = [None for _ in patch_groups]

        # Step 3: Low-rank restoration via DU-OVPCA
        restored_groups = []
        for group, rank in zip(patch_groups, ranks):
            restored = self.du_ovpca(group, rank)   # rank is [B]
            restored_groups.append(restored)

        # Step 4: Patch aggregation
        initial_restored = self._aggregate_patches(
            restored_groups=restored_groups,
            target_size=(H, W)
        )

        # Step 5: Supervised attention refinement
        if self.use_sam:
            refined_img, attention_map = self.sam(x, initial_restored)
        else:
            refined_img = initial_restored
            attention_map = None

        # Step 6: Global residual refinement
        final_img = self.global_refine(refined_img) + refined_img

        if return_attention:
            return final_img, {
                "attention_map": attention_map,
                "ranks": ranks,
                "complexities": complexities,
                "flow_field": flow_field,
                "deformed_grid": deformed_grid
            }

        return final_img

    def _extract_fixed_patches(self, x):
        """
        Extract patches using a fixed overlapping grid.

        Returns:
            patches: [B, num_patches, C, patch_size, patch_size]
        """
        B, C, H, W = x.shape
        stride = self.patch_size // 2

        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=stride,
            padding=self.patch_size // 2
        )  

        B, flat_dim, num_patches = patches.shape
        patches = patches.transpose(1, 2).contiguous()  # [B, num_patches, flat_dim]
        patches = patches.view(B, num_patches, C, self.patch_size, self.patch_size)

        return patches

    def _group_patches(self, patches):
        """
        Group patches into num_patch_groups groups.

        Args:
            patches: [B, num_patches, C, H, W]

        Returns:
            groups: list of tensors, each [B, T, C, H, W]
        """
        assert patches.dim() == 5, f"Expected patches [B, N, C, H, W], got {patches.shape}"

        B, num_patches, C, H, W = patches.shape
        groups = []

       
        group_size = self.patches_per_group

 
        total_needed = self.num_patch_groups * group_size
        if num_patches < total_needed:
            pad_count = total_needed - num_patches
            pad_patch = patches[:, -1:, :, :, :].repeat(1, pad_count, 1, 1, 1)
            patches = torch.cat([patches, pad_patch], dim=1)
            num_patches = patches.shape[1]

     
        patches = patches[:, :total_needed, :, :, :]

        for i in range(self.num_patch_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group = patches[:, start_idx:end_idx, :, :, :]  # [B, T, C, H, W]
            groups.append(group)

        return groups

    def _aggregate_patches(self, restored_groups, target_size):
        """
        Aggregate restored overlapping patches into full image.

        Args:
            restored_groups: list of tensors, each [B, T, C, patch_size, patch_size]
            target_size: (H, W)

        Returns:
            aggregated: [B, C, H, W]
        """
        assert len(restored_groups) > 0,

        B = restored_groups[0].shape[0]
        C = restored_groups[0].shape[2]
        H, W = target_size
        device = restored_groups[0].device

        accumulator = torch.zeros(B, C, H, W, device=device)
        weight_map = torch.zeros(B, 1, H, W, device=device)

        patch_h = self.patch_size
        patch_w = self.patch_size
        stride = self.patch_size // 2

     
        max_y = max(H - patch_h, 0)
        max_x = max(W - patch_w, 0)

        for group_idx, group in enumerate(restored_groups):
            # group: [B, T, C, patch_h, patch_w]
            _, T, _, _, _ = group.shape

            for patch_idx in range(T):
                patch = group[:, patch_idx]  # [B, C, patch_h, patch_w]

                # Learnable scalar weight per patch position
                weight = self.aggregation_weights[patch_idx % self.patches_per_group]

            
                y = min(group_idx * stride, max_y)
                x = min(patch_idx * stride, max_x)

                accumulator[:, :, y:y + patch_h, x:x + patch_w] += patch * weight
                weight_map[:, :, y:y + patch_h, x:x + patch_w] += weight

        weight_map = weight_map.clamp(min=1e-6)
        aggregated = accumulator / weight_map

        return aggregated
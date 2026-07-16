import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppm import PatchPartitioningModule
from .dram import DynamicRankAllocationModule
from .du_ovpca import DUOVPCA
from .sam import SupervisedAttentionModule


class MDIRNET(nn.Module):

    def __init__(
        self,
        in_channels=3,
        patch_size=8,
        num_patch_groups=512,
        patches_per_group=8,
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

        if self.use_ppm:
            self.ppm = PatchPartitioningModule(
                in_channels=in_channels,
                patch_size=patch_size
            )

        if self.use_dram:
            self.dram = DynamicRankAllocationModule(
                in_channels=in_channels,
                r_min=dram_min_rank,
                r_max=dram_max_rank
            )

        self.du_ovpca = DUOVPCA(
            num_iterations=num_ovpca_iterations,
            max_rank=dram_max_rank
        )

        if self.use_sam:
            self.sam = SupervisedAttentionModule(
                in_channels=in_channels
            )

    def forward(self, x, return_intermediates=False):
        B, C, H, W = x.shape
        device = x.device

        flow_field = None
        deformed_grid = None

        if self.use_ppm:
            patches, flow_field, deformed_grid = self.ppm(x)
        else:
            patches = self._extract_fixed_patches(x)

        patch_groups, group_centers = self._group_patches_knn(patches)

        ranks = []
        if self.use_dram:
            for group in patch_groups:
                rank = self.dram(group)
                ranks.append(rank)
        else:
            fixed_rank = min(
                16,
                self.du_ovpca.max_rank,
                self.patches_per_group
            )

            ranks = [
                torch.full(
                    (B,),
                    fixed_rank,
                    device=device,
                    dtype=torch.long
                )
                for _ in patch_groups
            ]

        restored_groups = []

        for group, rank in zip(patch_groups, ranks):
            restored = self.du_ovpca(group, rank)
            restored_groups.append(restored)

        d_hat = self._aggregate_patches(
            restored_groups,
            group_centers,
            (H, W)
        )

        if self.use_sam:
            d_tilde, gate_map = self.sam(x, d_hat)
        else:
            d_tilde = d_hat
            gate_map = None

        if return_intermediates:
            return d_tilde, {
                "gate_map": gate_map,
                "ranks": ranks,
                "flow_field": flow_field,
                "deformed_grid": deformed_grid,
                "d_hat": d_hat,
                "restored_groups": restored_groups
            }

        return d_tilde

    def _extract_fixed_patches(self, x):

        B, C, H, W = x.shape

        m = self.patch_size
        stride = m

        patches = x.unfold(2, m, stride).unfold(3, m, stride)

        nH, nW = patches.shape[2], patches.shape[3]

        patches = patches.contiguous().view(
            B,
            C,
            nH * nW,
            m,
            m
        )

        patches = patches.permute(0, 2, 1, 3, 4)

        centers = []

        for i in range(nH):
            for j in range(nW):

                cy = i * stride + m // 2
                cx = j * stride + m // 2

                centers.append((cy, cx))

        return patches, centers

    def _group_patches_knn(self, patches_input):

        if isinstance(patches_input, tuple):

            patches, centers = patches_input

        else:

            patches = patches_input

            B, N, C, H, W = patches.shape

            m = self.patch_size

            side = int(N ** 0.5) if N > 0 else 1

            centers = []

            for i in range(side):
                for j in range(side):

                    centers.append(
                        (
                            i * m + m // 2,
                            j * m + m // 2
                        )
                    )

            centers = centers[:N]

        B, N, C, pH, pW = patches.shape

        T = self.patches_per_group
        G = min(self.num_patch_groups, N)

        device = patches.device

        centers_tensor = torch.tensor(
            centers[:N],
            dtype=torch.float32,
            device=device
        )

        dist_matrix = torch.cdist(
            centers_tensor,
            centers_tensor,
            p=2
        )

        if N > G:
            reference_indices = torch.randperm(
                N,
                device=device
            )[:G]
        else:
            reference_indices = torch.arange(
                N,
                device=device
            )

        groups = []
        group_centers_list = []

        for ref_idx in reference_indices:

            dists = dist_matrix[ref_idx]

            nn_indices = torch.topk(
                dists,
                k=min(T, N),
                largest=False
            ).indices

            groups.append(
                patches[:, nn_indices]
            )

            group_centers_list.append(
                [centers[i] for i in nn_indices.cpu().tolist()]
            )

        return groups, group_centers_list

    def _aggregate_patches(
        self,
        restored_groups,
        group_centers,
        target_size
    ):

        B = restored_groups[0].shape[0]
        C = restored_groups[0].shape[2]

        H, W = target_size

        device = restored_groups[0].device

        m = self.patch_size

        accumulator = torch.zeros(
            B,
            C,
            H,
            W,
            device=device
        )

        weight_map = torch.zeros(
            B,
            1,
            H,
            W,
            device=device
        )

        for group, centers in zip(
            restored_groups,
            group_centers
        ):

            T = group.shape[1]

            for patch_idx in range(T):

                patch = group[:, patch_idx]

                cy, cx = centers[patch_idx]

                y = max(0, cy - m // 2)
                x = max(0, cx - m // 2)

                if H >= m:
                    y = min(y, H - m)
                else:
                    y = 0

                if W >= m:
                    x = min(x, W - m)
                else:
                    x = 0

                ph = min(m, H - y)
                pw = min(m, W - x)

                accumulator[:, :, y:y + ph, x:x + pw] += patch[:, :, :ph, :pw]

                weight_map[:, :, y:y + ph, x:x + pw] += 1.0

        weight_map = weight_map.clamp(min=1e-6)

        return accumulator / weight_map
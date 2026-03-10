import torch
import torch.nn as nn
import torch.nn.functional as F


class OVPCAIteration(nn.Module):
    """
    Single learnable iteration of OVPCA.
    Operates on tensors of shape [B, r_max].
    """

    def __init__(self, max_rank=32):
        super().__init__()
        self.max_rank = max_rank

        self.G_A = nn.Sequential(
            nn.Linear(max_rank * 2, max_rank * 4),
            nn.ReLU(inplace=True),
            nn.Linear(max_rank * 4, max_rank),
            nn.Tanh()
        )

        self.G_X = nn.Sequential(
            nn.Linear(max_rank * 2, max_rank * 4),
            nn.ReLU(inplace=True),
            nn.Linear(max_rank * 4, max_rank),
            nn.Tanh()
        )

        self.shrinkage = nn.Sequential(
            nn.Linear(max_rank, max_rank),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, D, U_r, V_r, l_r, k_A, k_X, l_hat, omega):
        """
        Args:
            D:     [B, M, T]
            U_r:   [B, M, r_max]
            V_r:   [B, T, r_max]
            l_r:   [B, r_max]
            k_A:   [B, r_max]
            k_X:   [B, r_max]
            l_hat: [B, r_max]
            omega: [B]

        Returns:
            k_A_new, k_X_new, l_hat_new, omega_new
        """
        B = D.shape[0]

       
        omega_col = omega.unsqueeze(-1)

        # Update basis scaling factors
        k_A_input = torch.cat([
            k_X * l_r * l_hat,
            omega_col.expand(B, l_r.shape[1])
        ], dim=-1)
        k_A_new = self.G_A(k_A_input)

        k_X_input = torch.cat([
            k_A * l_r * l_hat,
            omega_col.expand(B, l_r.shape[1])
        ], dim=-1)
        k_X_new = self.G_X(k_X_input)

        # Update singular values
        m_l = k_X_new * l_r * k_A_new
        s = torch.rsqrt(omega + 1e-8).unsqueeze(-1)

        psi = self.shrinkage(m_l)
        l_hat_new = m_l + s * psi

        # Update omega
        l_D_norm = torch.sum(l_r ** 2, dim=-1)                     # [B]
        l_hat_norm = torch.sum(l_hat_new ** 2, dim=-1)             # [B]
        inner_prod = torch.sum(k_X_new * l_hat_new * k_A_new * l_r, dim=-1)  # [B]

        residual = l_D_norm - 2.0 * inner_prod + l_hat_norm
        omega_new = (D.shape[-2] * D.shape[-1]) / (residual + 1e-8)

        return k_A_new, k_X_new, l_hat_new, omega_new


class DUOVPCA(nn.Module):
    """
    Deep Unfolded OVPCA Network

    Input:
        patch_group: [B, T, C, H, W]
        rank: int, scalar tensor, or tensor [B]

    Output:
        restored_patches: [B, T, C, H, W]
    """

    def __init__(self, num_iterations=6, max_rank=32, out_channels=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.max_rank = max_rank
        self.out_channels = out_channels

        self.iterations = nn.ModuleList([
            OVPCAIteration(max_rank) for _ in range(num_iterations)
        ])

        self.init_k = nn.Parameter(torch.ones(1, max_rank) * 0.1)
        self.init_omega = nn.Parameter(torch.tensor(1.0))

   
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def _svd(self, D, r):
        """
        Compute truncated batched low-rank SVD.

        Args:
            D: [B, M, T], where M = C*H*W
            r: int

        Returns:
            U: [B, M, r]
            S: [B, r]
            V: [B, T, r]
        """
        U, S, V = torch.svd_lowrank(D, q=r)
        return U, S, V

    def forward(self, patch_group, rank):
        """
        Args:
            patch_group: [B, T, C, H, W]
            rank: int, scalar tensor, or tensor [B]

        Returns:
            refined: [B, T, C, H, W]
        """
        assert patch_group.dim() == 5, f"Expected [B, T, C, H, W], got {patch_group.shape}"

        B, T, C, H, W = patch_group.shape
        device = patch_group.device

        
        if isinstance(rank, int):
            rank = torch.full((B,), rank, device=device, dtype=torch.long)
        elif isinstance(rank, torch.Tensor):
            rank = rank.to(device).long().view(-1)
            if rank.numel() == 1:
                rank = rank.expand(B)
            elif rank.numel() != B:
                raise ValueError(f"rank must have shape [B] or be scalar, got {rank.shape}")
        else:
            raise TypeError(f"Unsupported rank type: {type(rank)}")

        max_valid_rank = min(self.max_rank, T, C * H * W)
        rank = rank.clamp(min=1, max=max_valid_rank)

        r_max = int(rank.max().item())

        # Build patch-group matrix D: [B, M, T], M = C*H*W
        D = patch_group.reshape(B, T, C * H * W).permute(0, 2, 1).contiguous()

   
        U_r, S_r, V_r = self._svd(D, r_max)   # U_r:[B,M,r_max], S_r:[B,r_max], V_r:[B,T,r_max]
        l_r = S_r[:, :r_max]

        # Rank mask: [B, r_max]
        rank_mask = (
            torch.arange(r_max, device=device).unsqueeze(0) < rank.unsqueeze(1)
        ).float()

   
        k_A = self.init_k[:, :r_max].expand(B, -1).clone()
        k_X = self.init_k[:, :r_max].expand(B, -1).clone()
        l_hat = l_r.clone()
        omega = self.init_omega.expand(B).clone()

        k_A = k_A * rank_mask
        k_X = k_X * rank_mask
        l_hat = l_hat * rank_mask

        # Iterative unfolding
        for iteration in self.iterations:
            k_A, k_X, l_hat, omega = iteration(
                D, U_r, V_r, l_r, k_A, k_X, l_hat, omega
            )

            # Enforce per-sample active rank after every iteration
            k_A = k_A * rank_mask
            k_X = k_X * rank_mask
            l_hat = l_hat * rank_mask

      
        k_A = k_A * rank_mask
        k_X = k_X * rank_mask
        l_hat = l_hat * rank_mask

        # Reconstruction:
        # D_rec = U diag(k_A) diag(l_hat) diag(k_X) V^T
       
        diag_vals = k_A * l_hat * k_X   # [B, r_max]

        # Scale U by diagonal values
        U_scaled = U_r * diag_vals.unsqueeze(1)   # [B, M, r_max]
        D_rec = torch.bmm(U_scaled, V_r.transpose(1, 2))  # [B, M, T]

        # Back to [B, T, C, H, W]
        restored = D_rec.permute(0, 2, 1).contiguous().reshape(B, T, C, H, W)

        # Patch-wise refinement in image space
        restored_reshaped = restored.reshape(B * T, C, H, W)
        refined = self.refine(restored_reshaped)
        refined = refined.reshape(B, T, C, H, W)

        return refined
import torch
import torch.nn as nn


class OVPCAIteration(nn.Module):

    def __init__(self, max_rank=32):
        super().__init__()
        self.max_rank = max_rank
        self.shrinkage_scale = nn.Parameter(torch.ones(1, max_rank))

    def forward(self, l_r, k_A, k_X, l_hat, omega, p, n, D_fro_sq):
        omega_col = omega.unsqueeze(-1)
        r = l_r.shape[-1]

        k_A_new = torch.rsqrt(
            1.0 + omega_col * (l_r * k_X * l_hat) ** 2 + 1e-8
        )
        k_X_new = torch.rsqrt(
            1.0 + omega_col * (l_r * k_A * l_hat) ** 2 + 1e-8
        )

        m_l = k_X_new * l_r * k_A_new
        s = torch.rsqrt(omega + 1e-8).unsqueeze(-1) * self.shrinkage_scale[:, :r]

        l_hat_new = torch.sign(m_l) * torch.relu(torch.abs(m_l) - s)

        recon_vals = k_A_new * l_hat_new * k_X_new
        residual = (
            D_fro_sq
            - 2.0 * torch.sum(recon_vals * l_r, dim=-1)
            + torch.sum(recon_vals ** 2, dim=-1)
        )
        omega_new = (p * n) / (residual + 1e-8)

        return k_A_new, k_X_new, l_hat_new, omega_new


class DUOVPCA(nn.Module):

    def __init__(self, num_iterations=6, max_rank=32):
        super().__init__()
        self.num_iterations = num_iterations
        self.max_rank = max_rank

        self.iterations = nn.ModuleList([
            OVPCAIteration(max_rank) for _ in range(num_iterations)
        ])

    def forward(self, patch_group, rank):
        B, T, C, H, W = patch_group.shape
        device = patch_group.device
        M = C * H * W

        if isinstance(rank, int):
            rank = torch.full((B,), rank, device=device, dtype=torch.long)
        elif isinstance(rank, torch.Tensor):
            rank = rank.to(device).long().view(-1)
            if rank.numel() == 1:
                rank = rank.expand(B)

        max_valid_rank = min(self.max_rank, T, M)
        rank = rank.clamp(min=1, max=max_valid_rank)
        r_max = int(rank.max().item())

        D = patch_group.reshape(B, T, M).permute(0, 2, 1).contiguous()

        D_fro_sq = torch.sum(D ** 2, dim=(1, 2))

        U_r, S_r, V_r = torch.svd_lowrank(D, q=r_max)
        l_r = S_r[:, :r_max]

        rank_mask = (
            torch.arange(r_max, device=device).unsqueeze(0) < rank.unsqueeze(1)
        ).float()

        k_A = torch.ones(B, r_max, device=device) * rank_mask
        k_X = torch.ones(B, r_max, device=device) * rank_mask
        l_hat = l_r.clone() * rank_mask

        tail_energy = D_fro_sq - torch.sum(l_r ** 2, dim=-1)
        omega = (M * T) / (tail_energy + 1e-8)

        for iteration in self.iterations:
            k_A, k_X, l_hat, omega = iteration(
                l_r, k_A, k_X, l_hat, omega, M, T, D_fro_sq
            )
            k_A = k_A * rank_mask
            k_X = k_X * rank_mask
            l_hat = l_hat * rank_mask

        diag_vals = k_A * l_hat * k_X
        U_scaled = U_r * diag_vals.unsqueeze(1)
        D_rec = torch.bmm(U_scaled, V_r.transpose(1, 2))

        restored = D_rec.permute(0, 2, 1).contiguous().reshape(B, T, C, H, W)
        return restored

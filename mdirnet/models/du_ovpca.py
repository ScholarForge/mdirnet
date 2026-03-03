import torch
import torch.nn as nn
import torch.nn.functional as F

class OVPCAIteration(nn.Module):
    """
    Single iteration of OVPCA algorithm as a learnable layer
    """
    
    def __init__(self, max_rank=32):
        super().__init__()
        self.max_rank = max_rank
        
        # Learnable parameters for each update step
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
        
        # Initialize as identity-like
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, D, U_r, V_r, l_r, k_A, k_X, l_hat, omega, r):
        """
        Single OVPCA iteration
        """
        B = D.shape[0]
        device = D.device
        
        # Update k_A and k_X (basis scaling factors)
        k_A_input = torch.cat([
            k_X * l_r.unsqueeze(0) * l_hat.unsqueeze(0),
            omega.expand(B, -1)
        ], dim=-1)
        k_A_new = self.G_A(k_A_input)
        
        k_X_input = torch.cat([
            k_A * l_r.unsqueeze(0) * l_hat.unsqueeze(0),
            omega.expand(B, -1)
        ], dim=-1)
        k_X_new = self.G_X(k_X_input)
        
        # Compute m_l (mean of singular values)
        m_l = k_X_new * l_r.unsqueeze(0) * k_A_new
        
        # Compute s (scale parameter)
        s = torch.rsqrt(omega).unsqueeze(-1)
        

        psi = self.shrinkage(m_l)
        l_hat_new = m_l + s * psi
        
        # Update omega
        l_D_norm = torch.norm(l_r) ** 2
        l_hat_norm = torch.norm(l_hat_new, dim=-1) ** 2
        inner_prod = (k_X_new * l_hat_new * k_A_new * l_r.unsqueeze(0)).sum(dim=-1)
        
        residual = l_D_norm - 2 * inner_prod + l_hat_norm
        omega_new = (D.shape[-2] * D.shape[-1]) / (residual + 1e-8)
        
        return k_A_new, k_X_new, l_hat_new, omega_new


class DUOVPCA(nn.Module):
    """
    Deep Unfolded OVPCA Network
    
    Unrolls iterative OVPCA optimization into learnable layers.
    """
    
    def __init__(self, num_iterations=6, max_rank=32):
        super().__init__()
        self.num_iterations = num_iterations
        self.max_rank = max_rank
        
        # Learnable OVPCA iterations
        self.iterations = nn.ModuleList([
            OVPCAIteration(max_rank) for _ in range(num_iterations)
        ])
        
        # Optional: learnable initializations
        self.init_k = nn.Parameter(torch.ones(1, max_rank) * 0.1)
        self.init_omega = nn.Parameter(torch.tensor(1.0))
        
        # Final reconstruction layers
        self.refine = nn.Sequential(
            nn.Conv2d(max_rank, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def _svd(self, x, r):
        """Compute truncated SVD"""
        B, C, H, W = x.shape
        
        # Reshape to [B, C*H, W] or similar based on patch structure
      
        x_flat = x.view(B, C*H, W)
        
        U, S, V = torch.svd_lowrank(x_flat, q=r)
        return U, S, V
    
    def forward(self, patch_group, rank):
        """
        Args:
            patch_group: [B, T, C, H, W] patch group
            rank: Integer rank for this group
            
        Returns:
            restored_patches: Restored patches [B, T, C, H, W]
        """
        B, T, C, H, W = patch_group.shape
        device = patch_group.device
        r = rank.item() if isinstance(rank, torch.Tensor) else rank
        
        # Prepare data matrix [B, C*H*W, T]
        D = patch_group.view(B, -1, T)
        
        # Compute SVD
        U_r, S_r, V_r = self._svd(D.view(B, C, H*W, T), r)
        l_r = S_r[:, :r]
        
        # Initialize
        k_A = self.init_k[:, :r].expand(B, -1)
        k_X = self.init_k[:, :r].expand(B, -1)
        l_hat = l_r.clone()
        omega = self.init_omega.expand(B)
        
        # Iterative refinement
        for i, iteration in enumerate(self.iterations):
            k_A, k_X, l_hat, omega = iteration(
                D, U_r, V_r, l_r, k_A, k_X, l_hat, omega, r
            )
        
        # Reconstruct
        A_rec = U_r[:, :, :r] @ torch.diag_embed(k_A)
        X_rec = V_r[:, :, :r] @ torch.diag_embed(k_X)
        L_rec = torch.diag_embed(l_hat)
        
        D_rec = A_rec @ L_rec @ X_rec.transpose(-2, -1)
        
        # Reshape back to patches
        restored = D_rec.view(B, T, C, H, W)
        
        restored_reshaped = restored.view(B*T, C, H, W)
        refined = self.refine(restored_reshaped)
        refined = refined.view(B, T, C, H, W)
        
        return refined
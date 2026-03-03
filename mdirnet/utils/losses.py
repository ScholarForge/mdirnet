import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridReconstructionLoss(nn.Module):
    """
    Hybrid L1 + L2 reconstruction loss
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        
        return l1_loss + self.alpha * l2_loss


class LatentConsistencyLoss(nn.Module):
    """
    Latent feature uncertainty regularizer

    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, patch_groups):
        """
        Args:
            patch_groups: List of restored patch groups
                          each shape [B, T, C, H, W]
        """
        consistency_loss = 0.0
        num_groups = len(patch_groups)
        
        for group in patch_groups:
            # Compute variance across patches in group
            mean_patch = group.mean(dim=1, keepdim=True)
            variance = ((group - mean_patch) ** 2).mean()
            
            # Log-scaled penalty
            consistency_loss += torch.log(1 + variance)
        
        return consistency_loss / num_groups


class RankRegularizationLoss(nn.Module):
    """
    Encourages efficient rank allocation
    Penalizes unnecessarily high ranks
    """
    
    def __init__(self, target_rank_ratio=0.5, max_rank=32):
        super().__init__()
        self.target_rank_ratio = target_rank_ratio
        self.max_rank = max_rank
    
    def forward(self, ranks):
        """
        Args:
            ranks: Predicted ranks [num_groups]
        """
        # Normalize ranks to [0, 1]
        norm_ranks = ranks.float() / self.max_rank
        
        # Encourage ranks near target ratio
        target = torch.ones_like(norm_ranks) * self.target_rank_ratio
        loss = F.mse_loss(norm_ranks, target)
        
        return loss


class MDIRNETLoss(nn.Module):
    """
    Combined loss for MDIRNET training
    """
    
    def __init__(self,
                 lambda_rec=1.0,
                 lambda_unc=0.1,
                 lambda_rank=0.05,
                 alpha_l2=0.1):
        super().__init__()
        
        self.lambda_rec = lambda_rec
        self.lambda_unc = lambda_unc
        self.lambda_rank = lambda_rank
        
        self.rec_loss = HybridReconstructionLoss(alpha=alpha_l2)
        self.unc_loss = LatentConsistencyLoss()
        self.rank_loss = RankRegularizationLoss()
    
    def forward(self,
                pred,
                target,
                patch_groups=None,
                ranks=None,
                return_components=False):
        """
        Compute total loss
        """
        # Reconstruction loss
        loss_rec = self.rec_loss(pred, target)
        
        # Latent consistency loss
        loss_unc = 0.0
        if patch_groups is not None and self.lambda_unc > 0:
            loss_unc = self.unc_loss(patch_groups)
        
        # Rank regularization loss
        loss_rank = 0.0
        if ranks is not None and self.lambda_rank > 0:
            loss_rank = self.rank_loss(ranks)
        
        total_loss = (self.lambda_rec * loss_rec +
                      self.lambda_unc * loss_unc +
                      self.lambda_rank * loss_rank)
        
        if return_components:
            return total_loss, {
                'rec_loss': loss_rec.item(),
                'unc_loss': loss_unc.item() if isinstance(loss_unc, torch.Tensor) else 0,
                'rank_loss': loss_rank.item() if isinstance(loss_rank, torch.Tensor) else 0
            }
        
        return total_loss
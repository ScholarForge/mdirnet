import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridReconstructionLoss(nn.Module):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        return l1_loss + self.alpha * l2_loss


class LatentConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, patch_groups):
        consistency_loss = 0.0
        num_groups = len(patch_groups)

        for group in patch_groups:
            mean_patch = group.mean(dim=1, keepdim=True)
            variance = ((group - mean_patch) ** 2).mean()
            consistency_loss += torch.log(1.0 + variance)

        return consistency_loss / max(num_groups, 1)


class MDIRNETLoss(nn.Module):

    def __init__(self, lambda_rec=1.0, lambda_unc=0.1, alpha_l2=0.5):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_unc = lambda_unc

        self.rec_loss = HybridReconstructionLoss(alpha=alpha_l2)
        self.unc_loss = LatentConsistencyLoss()

    def forward(self, pred, target, patch_groups=None):
        loss_rec = self.rec_loss(pred, target)

        loss_unc = torch.tensor(0.0, device=pred.device)
        if patch_groups is not None and self.lambda_unc > 0:
            loss_unc = self.unc_loss(patch_groups)

        total = self.lambda_rec * loss_rec + self.lambda_unc * loss_unc
        return total

import torch
import torch.nn as nn


class DynamicRankAllocationModule(nn.Module):

    def __init__(self, in_channels=3, r_min=4, r_max=32):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max

        self.feat_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.rank_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, patch_group):
        B, T, C, H, W = patch_group.shape

        mean_patch = patch_group.mean(dim=1)

        feat = self.feat_extractor(mean_patch).view(B, -1)
        r_hat = self.rank_head(feat).squeeze(-1)

        r_cont = self.r_min + 2.0 * (r_hat * (self.r_max - self.r_min) / 2.0)
        r_disc = self.r_min + 2.0 * torch.round(r_hat * (self.r_max - self.r_min) / 2.0)
        r_disc = r_disc.clamp(self.r_min, self.r_max)

        r_pred = r_cont + (r_disc - r_cont).detach()

        return r_pred.long()

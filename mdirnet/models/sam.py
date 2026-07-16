import torch
import torch.nn as nn


class SupervisedAttentionModule(nn.Module):

    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, degraded, restored):
        f_deg = self.conv1(degraded)
        f_res = self.conv2(restored)

        gate = self.conv3(torch.cat([f_deg, f_res], dim=1))

        refined = gate * degraded + (1 - gate) * restored

        return refined, gate

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchPartitioningModule(nn.Module):

    def __init__(self, in_channels=3, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 2, 3, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.flow_net:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        m = self.patch_size

        delta = self.flow_net(x)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        flow = delta.permute(0, 2, 3, 1)
        deformed_grid = base_grid + flow

        nH = H // m
        nW = W // m

        center_ys = torch.arange(nH, device=device).float() * m + m // 2
        center_xs = torch.arange(nW, device=device).float() * m + m // 2

        patches_list = []
        centers_list = []

        for b in range(B):
            batch_patches = []
            for i in range(nH):
                for j in range(nW):
                    cy_pixel = int(center_ys[i].item())
                    cx_pixel = int(center_xs[j].item())

                    cy_pixel = min(cy_pixel, H - 1)
                    cx_pixel = min(cx_pixel, W - 1)

                    offset = deformed_grid[b, cy_pixel, cx_pixel] - base_grid[0, cy_pixel, cx_pixel]

                    y_start = max(0, cy_pixel - m // 2)
                    x_start = max(0, cx_pixel - m // 2)
                    y_start = min(y_start, H - m) if H >= m else 0
                    x_start = min(x_start, W - m) if W >= m else 0

                    sub_grid_y = torch.linspace(-1, 1, m, device=device)
                    sub_grid_x = torch.linspace(-1, 1, m, device=device)
                    gy, gx = torch.meshgrid(sub_grid_y, sub_grid_x, indexing="ij")
                    sub_base = torch.stack([gx, gy], dim=-1)

                    scale_y = m / H
                    scale_x = m / W
                    center_norm_x = (x_start + m / 2) / W * 2 - 1
                    center_norm_y = (y_start + m / 2) / H * 2 - 1

                    sample_grid = torch.stack([
                        center_norm_x + sub_base[..., 0] * scale_x + offset[0],
                        center_norm_y + sub_base[..., 1] * scale_y + offset[1],
                    ], dim=-1).unsqueeze(0)

                    patch = F.grid_sample(
                        x[b:b+1], sample_grid,
                        mode="bilinear", padding_mode="zeros", align_corners=False
                    )
                    batch_patches.append(patch)

                    if b == 0:
                        shifted_cy = cy_pixel + int(offset[1].item() * H / 2)
                        shifted_cx = cx_pixel + int(offset[0].item() * W / 2)
                        shifted_cy = max(m // 2, min(shifted_cy, H - m // 2))
                        shifted_cx = max(m // 2, min(shifted_cx, W - m // 2))
                        centers_list.append((shifted_cy, shifted_cx))

            batch_patches = torch.cat(batch_patches, dim=0)
            patches_list.append(batch_patches)

        patches = torch.stack(patches_list)
        N = nH * nW
        patches = patches.view(B, N, C, m, m)

        return (patches, centers_list), delta, deformed_grid

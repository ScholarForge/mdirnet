import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchPartitioningModule(nn.Module):
    """
    Learnable Patch Partitioning Module (PPM)

    Extracts adaptive patches using a learned flow field.
    """

    def __init__(
        self,
        in_channels=3,
        patch_size=64,
        stride=None
    ):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2

        # Flow prediction network
        self.flow_net = nn.Sequential(

            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()
        )

        # controls magnitude of deformation
        self.sampling_temperature = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args
            x: [B, C, H, W]

        Returns
            patches: [B, N, C, patch_size, patch_size]
            flow_field: [B, 2, H, W]
            deformed_grid: [B, H, W, 2]
        """

        B, C, H, W = x.shape
        device = x.device

        # Step 1: predict deformation field
        flow_field = self.flow_net(x)

        # Step 2: base sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Step 3: deform grid
        flow = flow_field.permute(0, 2, 3, 1)
        deformed_grid = base_grid + flow * self.sampling_temperature

        # Step 4: extract patches
        patches = self._extract_patches_from_grid(x, deformed_grid)

        return patches, flow_field, deformed_grid

    def _extract_patches_from_grid(self, x, grid):

        B, C, H, W = x.shape
        device = x.device

        stride = self.stride

        grid_sampled = grid[:, ::stride, ::stride, :]
        B, Hp, Wp, _ = grid_sampled.shape

        num_patches = Hp * Wp

        grid_flat = grid_sampled.view(B, -1, 2)

        patches = []

        for b in range(B):

            batch_patches = []

            for i in range(num_patches):

                center = grid_flat[b, i]

                patch_grid = self._create_patch_grid(
                    center, self.patch_size, H, W, device
                )

                patch = F.grid_sample(
                    x[b:b+1],
                    patch_grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False
                )

                batch_patches.append(patch)

            batch_patches = torch.cat(batch_patches, dim=0)
            patches.append(batch_patches)

        patches = torch.stack(patches)
        patches = patches.view(B, num_patches, C, self.patch_size, self.patch_size)

        return patches

    def _create_patch_grid(self, center, patch_size, H, W, device):

        patch_y, patch_x = torch.meshgrid(
            torch.linspace(-1, 1, patch_size, device=device),
            torch.linspace(-1, 1, patch_size, device=device),
            indexing="ij"
        )

        scale_x = patch_size / W
        scale_y = patch_size / H

        patch_grid = torch.stack([
            center[0] + patch_x * scale_x,
            center[1] + patch_y * scale_y
        ], dim=-1)

        return patch_grid

    def visualize_sampling_density(self, x, flow_field):

        with torch.no_grad():

            B, C, H, W = x.shape

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing="ij"
            )

            base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

            deformed_grid = base_grid + flow_field.permute(0, 2, 3, 1)

            density = torch.zeros(B, H, W, device=x.device)

            sampled_points = deformed_grid[:, ::self.stride, ::self.stride, :]

            pixel_coords = (sampled_points + 1) / 2
            pixel_coords[..., 0] *= (W - 1)
            pixel_coords[..., 1] *= (H - 1)

            pixel_coords = pixel_coords.long()

            for b in range(B):
                for i in range(pixel_coords.shape[1]):
                    for j in range(pixel_coords.shape[2]):

                        x_coord = pixel_coords[b, i, j, 0].clamp(0, W - 1)
                        y_coord = pixel_coords[b, i, j, 1].clamp(0, H - 1)

                        density[b, y_coord, x_coord] += 1

            return density
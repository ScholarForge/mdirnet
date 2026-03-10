import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchPartitioningModule(nn.Module):
    """
    Learnable Patch Partitioning Module (PPM)

    """
    
    def __init__(self, 
                 in_channels=3,
                 patch_size=64,
                 stride=None,  # If None, uses patch_size//2 for overlap
                 num_neighbors=15):  # k-1 nearest neighbors for grouping
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2
        self.num_neighbors = num_neighbors
        
        # Small CNN for flow field prediction
        self.flow_net = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4
            
            # Decoder
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),  # Output 2-channel flow field
            nn.Tanh()  # Output normalized offsets in [-1, 1]
        )
        
    
        self.sampling_temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values to start with near-regular grid"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
              
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Degraded image [B, C, H, W]
            
        Returns:
            patches: List of extracted patches with their coordinates
            patch_groups: Grouped patches for collaborative filtering
            flow_field: Predicted flow field [B, 2, H, W]
            sampling_grid: Deformed sampling grid
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Step 1: Predict flow field 
        flow_field = self.flow_net(x)  # [B, 2, H, W]
        
        # Step 2: Create base grid G_base (normalized coordinates [-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Step 3: Create deformed grid G_deformed = G_base + Δ
        flow_field_permuted = flow_field.permute(0, 2, 3, 1)  # [B, H, W, 2]
        deformed_grid = base_grid + flow_field_permuted * self.sampling_temperature
        
        # Step 4: Extract patches using differentiable sampling
        # For each potential patch center, sample using grid
        patches = self._extract_patches_from_grid(x, deformed_grid)
        
        # Step 5: Group patches by spatial proximity (k-nearest neighbors)
        patch_groups = self._group_patches_by_proximity(patches, deformed_grid)
        
        return {
            'patches': patches,
            'patch_groups': patch_groups,
            'flow_field': flow_field,
            'sampling_grid': deformed_grid,
            'base_grid': base_grid
        }
    
    def _extract_patches_from_grid(self, x, grid):
        """
        Extract patches centered at deformed grid positions
    
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Number of patch centers (can be all pixels or sampled)
        # For efficiency, we sample a subset of grid points
        stride = self.stride
        grid_sampled = grid[:, ::stride, ::stride, :]  # [B, H', W', 2]
        
        B, Hp, Wp, _ = grid_sampled.shape
        num_patches = Hp * Wp
        
        # Reshape grid for grid_sample
        grid_flat = grid_sampled.view(B, -1, 1, 2)  # [B, N, 1, 2]
        
        # Extract patches using grid_sample
        # For each center, we need to sample a patch of size patch_size
        patches = []
        patch_coords = []
        
        for b in range(B):
            batch_patches = []
            batch_coords = []
            
            for i in range(num_patches):
                center = grid_flat[b, i, 0]  # [2]
                
                # Create local grid for patch around center
                patch_grid = self._create_patch_grid(center, self.patch_size, H, W, device)
                
                # Sample patch
                patch = F.grid_sample(
                    x[b:b+1], 
                    patch_grid.unsqueeze(0),
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )  # [1, C, patch_size, patch_size]
                
                batch_patches.append(patch)
                batch_coords.append(center)
            
            patches.append(torch.stack(batch_patches, dim=1))  # [1, N, C, P, P]
            patch_coords.append(torch.stack(batch_coords))
        
        patches = torch.cat(patches, dim=0)  # [B, N, C, P, P]
        patch_coords = torch.stack(patch_coords)  # [B, N, 2]
        
        return {
            'values': patches,
            'coordinates': patch_coords,
            'grid': grid_sampled
        }
    
    def _create_patch_grid(self, center, patch_size, H, W, device):
        """Create local grid for sampling a patch around center"""
        # Create normalized coordinates for patch
        patch_y, patch_x = torch.meshgrid(
            torch.linspace(-1, 1, patch_size, device=device),
            torch.linspace(-1, 1, patch_size, device=device),
            indexing='ij'
        )
        
        # Scale to be relative to patch size (not image size)
        # This ensures we sample a fixed-size patch regardless of image dimensions
        scale_x = 2.0 / W  # Convert normalized coordinates to pixel offsets
        scale_y = 2.0 / H
        
        patch_grid = torch.stack([
            center[0] + patch_x * scale_x * patch_size / 2,
            center[1] + patch_y * scale_y * patch_size / 2
        ], dim=-1)  # [patch_size, patch_size, 2]
        
        return patch_grid
    
    def _group_patches_by_proximity(self, patches_dict, deformed_grid):
        """
        Group patches by spatial proximity
        For each reference patch, collect k-1 nearest spatial neighbors
        
        As described in paper: "for each reference patch Pi, 
        collect its k-1 nearest neighboring patches in the spatial domain"
        """
        patches = patches_dict['values']  # [B, N, C, P, P]
        coords = patches_dict['coordinates']  # [B, N, 2]
        B, N, C, P, _ = patches.shape
        K = self.num_neighbors + 1  
        
        patch_groups = []
        
        for b in range(B):
            # Compute spatial distances between patch centers
            # coords[b]: [N, 2]
            coord_diff = coords[b].unsqueeze(1) - coords[b].unsqueeze(0)  # [N, N, 2]
            spatial_distances = torch.norm(coord_diff, dim=-1)  # [N, N]
            
            # For each patch, find k-1 nearest neighbors
            # We want groups of size K (including itself)
            _, neighbor_indices = torch.topk(spatial_distances, 
                                            k=K, 
                                            dim=-1, 
                                            largest=False)  # [N, K]
            
            # Form groups
            for i in range(N):
                group_indices = neighbor_indices[i]  # [K]
                group_patches = patches[b, group_indices]  # [K, C, P, P]
                
                # Ensure we have exactly K patches
                if len(group_indices) < K:
                    # Pad with nearest neighbors if needed
                    pad_size = K - len(group_indices)
                    padding = group_patches[-1:].repeat(pad_size, 1, 1, 1)
                    group_patches = torch.cat([group_patches, padding], dim=0)
                
                patch_groups.append(group_patches)
        
        # Stack all groups: [B*N, K, C, P, P]
        patch_groups = torch.stack(patch_groups, dim=0)
        
        # Reshape to [B, num_groups, K, C, P, P]
        patch_groups = patch_groups.view(B, -1, K, C, P, P)
        
        return patch_groups
    
    def visualize_sampling_density(self, x, flow_field):
        """
        Shows that more patches are allocated to degraded regions
        """
        with torch.no_grad():
            B, C, H, W = x.shape
            
            # Get deformed grid
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing='ij'
            )
            base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            deformed_grid = base_grid + flow_field.permute(0, 2, 3, 1)
            
            # Create density map
            density = torch.zeros(B, H, W, device=x.device)
            
            # Sample points from deformed grid
            sampled_points = deformed_grid[:, ::self.stride, ::self.stride, :]
            
            # Convert to pixel coordinates
            pixel_coords = (sampled_points + 1) / 2
            pixel_coords[..., 0] *= (W - 1)
            pixel_coords[..., 1] *= (H - 1)
            pixel_coords = pixel_coords.long()
            
            # Increment density
            for b in range(B):
                for i in range(pixel_coords.shape[1]):
                    for j in range(pixel_coords.shape[2]):
                        x_coord = pixel_coords[b, i, j, 0].clamp(0, W-1)
                        y_coord = pixel_coords[b, i, j, 1].clamp(0, H-1)
                        density[b, y_coord, x_coord] += 1
            

            return density

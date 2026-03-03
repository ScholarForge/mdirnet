import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRankAllocationModule(nn.Module):
    """
    Dynamic Rank Allocation Module (DRAM)
    
    Predicts optimal rank for each patch group
    Outputs rank value between r_min and r_max.
    """
    
    def __init__(self, 
                 in_channels=3, 
                 hidden_dim=128,
                 r_min=4, 
                 r_max=32,
                 patch_size=64):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.patch_size = patch_size
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Rank prediction head
        self.rank_predictor = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
       
        self.complexity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
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
        """
        Args:
            patch_group: [B, T, C, H, W] where T is number of patches in group
            
        Returns:
            predicted_rank: Integer rank for each group [B]
            complexity_score: Estimated complexity [B, 1]
        """
        B, T, C, H, W = patch_group.shape
        
        # Compute mean patch across group
        mean_patch = patch_group.mean(dim=1)  # [B, C, H, W]
        
        # Extract features
        features = self.feature_extractor(mean_patch)  # [B, 128, 1, 1]
        features = features.view(B, -1)  # [B, 128]
        
        # Predict normalized rank
        rank_norm = self.rank_predictor(features)  # [B, 1]
        
        # Predict complexity score
        complexity = self.complexity_head(features)  # [B, 1]
        
        # Convert to integer rank
        rank_scaled = self.r_min + rank_norm * (self.r_max - self.r_min)
        
        rank_pred = (rank_scaled / 2).round() * 2
        rank_pred = rank_pred.clamp(self.r_min, self.r_max)
        
        return rank_pred.squeeze(-1).long(), complexity
    
    def adaptive_forward(self, patch_groups):
        """
        Process multiple groups with batched rank prediction
        """
        ranks = []
        complexities = []
        
        for i, group in enumerate(patch_groups):
            rank, comp = self.forward(group.unsqueeze(0))
            ranks.append(rank)
            complexities.append(comp)
        
        return torch.stack(ranks), torch.stack(complexities)
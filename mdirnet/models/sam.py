import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedAttentionModule(nn.Module):
    """
    Supervised Attention Module (SAM)
    
    Refines restored image by comparing with original degraded input
    and generating attention maps.
    """
    
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        
        # Feature extractors
        self.degraded_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.restored_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Attention map generator
        self.attention_generator = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Refinement network
        self.refine_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, degraded_img, restored_img):
        """
        Args:
            degraded_img: Original degraded image [B, C, H, W]
            restored_img: Initially restored image [B, C, H, W]
            
        Returns:
            refined_img: Final refined image [B, C, H, W]
            attention_map: Generated attention map [B, 1, H, W]
        """
        # Extract features
        feat_degraded = self.degraded_conv(degraded_img)
        feat_restored = self.restored_conv(restored_img)
        
        # Generate attention map
        concat_feats = torch.cat([feat_degraded, feat_restored], dim=1)
        attention_map = self.attention_generator(concat_feats)
        
        # Apply attention for refinement
        refined = attention_map * restored_img + (1 - attention_map) * degraded_img
        
    
        concat_inputs = torch.cat([degraded_img, restored_img], dim=1)
        refine_offset = self.refine_net(concat_inputs)
        refined = refined + 0.1 * refine_offset  
        
        return refined, attention_map
    
    def get_attention_map(self, degraded_img, restored_img):
        
        feat_degraded = self.degraded_conv(degraded_img)
        feat_restored = self.restored_conv(restored_img)
        
        concat_feats = torch.cat([feat_degraded, feat_restored], dim=1)
        attention_map = self.attention_generator(concat_feats)
        
        return attention_map
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Keep feature layers only
        self.blocks = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[4:9],  # relu2_2
            vgg.features[9:16], # relu3_3
            vgg.features[16:23] # relu4_3
        ])
        
        # Weights for different layers
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
        
    def forward(self, x, y):
        """
        Calculate perceptual loss between x and y
        
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            
        Returns:
            torch.Tensor: Perceptual loss
        """
        # Normalize to match VGG input
        x = self._normalize(x)
        y = self._normalize(y)
        
        loss = 0.0
        x_features = [x]
        y_features = [y]
        
        # Extract features at different layers
        for block in self.blocks:
            x_features.append(block(x_features[-1]))
            y_features.append(block(y_features[-1]))
            
        # Calculate L2 loss at each layer
        for i in range(1, len(x_features)):
            loss += self.weights[i-1] * F.mse_loss(x_features[i], y_features[i])
            
        return loss
    
    def _normalize(self, x):
        """Normalize tensor to match VGG input range"""
        # Expects input in range [-1, 1]
        x = (x + 1.0) / 2.0  # Convert to [0, 1]
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std

class StyleLoss(nn.Module):
    """
    Style loss using Gram matrix
    """
    def __init__(self):
        super(StyleLoss, self).__init__()
        
    def gram_matrix(self, x):
        """
        Calculate Gram matrix for style representation
        
        Args:
            x (torch.Tensor): Feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Gram matrix
        """
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        # Normalize by feature map size
        return gram.div(C * H * W)
    
    def forward(self, x, y):
        """
        Calculate style loss between x and y
        
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            
        Returns:
            torch.Tensor: Style loss
        """
        x_gram = self.gram_matrix(x)
        y_gram = self.gram_matrix(y)
        return F.mse_loss(x_gram, y_gram)

class LVRLoss(nn.Module):
    """
    Layout Visual Refinement Loss
    
    Contains two components:
    - Perception Refinement (PR) Loss: Improves texture details
    - Style Refinement (SR) Loss: Enhances style consistency
    """
    def __init__(self):
        super(LVRLoss, self).__init__()
        self.perception_loss = VGGPerceptualLoss()
        self.style_loss = StyleLoss()
        
    def forward(self, gen_features, real_features, layout_mask=None):
        """
        Calculate LVR loss
        
        Args:
            gen_features (torch.Tensor): Generated image features
            real_features (torch.Tensor): Real image features
            layout_mask (torch.Tensor, optional): Mask to focus on layout regions
            
        Returns:
            torch.Tensor: Combined LVR loss
        """
        # Apply mask to focus on layout regions if provided
        if layout_mask is not None:
            gen_masked = gen_features * layout_mask
            real_masked = real_features * layout_mask
        else:
            gen_masked = gen_features
            real_masked = real_features
        
        # Calculate perception refinement loss
        pr_loss = self.perception_loss(gen_masked, real_masked)
        
        # Calculate style refinement loss
        sr_loss = self.style_loss(gen_masked, real_masked)
        
        # Combined loss (can be weighted differently if needed)
        return pr_loss + 0.5 * sr_loss

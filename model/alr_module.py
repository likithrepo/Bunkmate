import torch
import torch.nn as nn
import torch.nn.functional as F

class ALRModule(nn.Module):
    """
    Adaptive Layout Refinement Module for ALR-GAN
    
    Aligns the layout structure (object locations) of synthesized images 
    with real images using an adaptive loss function.
    """
    def __init__(self, feature_dim, text_dim):
        super(ALRModule, self).__init__()
        
        # Feature transformation for image features
        self.image_transform = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, 1, 0),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature transformation for text features
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive weight network
        self.adaptive_weight = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # Feature refinement network
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def calculate_ssm(self, text_features, image_features):
        """
        Calculate Semantics Similarity Matrix (SSM) between text and image
        
        Args:
            text_features (dict): Text embeddings with word-level features
            image_features (torch.Tensor): Image feature maps [B, C, H, W]
            
        Returns:
            torch.Tensor: Semantics Similarity Matrix
        """
        # Get word embeddings from text features
        word_embeddings = text_features['word_embeddings']  # [B, L, D]
        
        # Transform image features
        img_features = self.image_transform(image_features)  # [B, C, H, W]
        B, C, H, W = img_features.size()
        
        # Reshape image features to [B, C, H*W]
        img_features = img_features.view(B, C, -1)
        
        # Reshape word embeddings to [B, L, D]
        word_features = word_embeddings.permute(0, 2, 1)  # [B, D, L]
        
        # Calculate similarity matrix
        # SSM = softmax(img_features^T * word_features)
        similarity = torch.bmm(img_features.permute(0, 2, 1), word_features)  # [B, H*W, L]
        
        # Apply softmax to get attention weights
        ssm = F.softmax(similarity, dim=2)  # [B, H*W, L]
        
        # Reshape back to spatial dimensions
        ssm = ssm.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, L, H, W]
        
        return ssm
    
    def forward(self, text_features, image_features, real_image_features=None):
        """
        Forward pass through the ALR module
        
        Args:
            text_features (dict): Text embeddings 
            image_features (torch.Tensor): Feature maps from generator
            real_image_features (torch.Tensor, optional): Feature maps from real images
            
        Returns:
            torch.Tensor: Refined image features
            torch.Tensor: ALR loss (if in training mode)
        """
        # Calculate Semantics Similarity Matrix for generated image
        ssm = self.calculate_ssm(text_features, image_features)
        
        # Calculate adaptive weights
        adaptive_weights = self.adaptive_weight(image_features)
        
        # Apply feature refinement
        refined_features = self.refinement(image_features)
        
        # If in training mode (real image features provided)
        if real_image_features is not None:
            # Calculate SSM for real image
            real_ssm = self.calculate_ssm(text_features, real_image_features)
            
            # Calculate ALR loss
            hard_regions = 1.0 - adaptive_weights  # Hard regions have higher weight
            
            # MSE between generated and real SSM, weighted by hardness
            alr_loss = F.mse_loss(ssm * hard_regions, real_ssm * hard_regions)
            
            return refined_features, alr_loss
        
        # In inference mode
        return refined_features
    
    def alr_loss(self, gen_ssm, real_ssm, adaptive_weights=None):
        """
        Calculate Adaptive Layout Refinement loss
        
        Args:
            gen_ssm (torch.Tensor): SSM for generated image
            real_ssm (torch.Tensor): SSM for real image
            adaptive_weights (torch.Tensor, optional): Weights for different regions
            
        Returns:
            torch.Tensor: ALR loss
        """
        if adaptive_weights is None:
            # Use uniform weights if not provided
            loss = F.mse_loss(gen_ssm, real_ssm)
        else:
            # Weight loss by adaptive weights (focusing on hard regions)
            hard_regions = 1.0 - adaptive_weights
            loss = F.mse_loss(gen_ssm * hard_regions, real_ssm * hard_regions)
        
        return loss

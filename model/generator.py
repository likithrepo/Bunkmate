import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual Block for the generator"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class ConditionalBatchNorm(nn.Module):
    """Conditional Batch Normalization for text conditioning"""
    def __init__(self, in_channels, text_dim):
        super(ConditionalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.gamma_beta = nn.Linear(text_dim, in_channels * 2)
        self.in_channels = in_channels
        
    def forward(self, x, text_embedding):
        out = self.bn(x)
        gamma_beta = self.gamma_beta(text_embedding)
        gamma = gamma_beta[:, :self.in_channels].unsqueeze(2).unsqueeze(3)
        beta = gamma_beta[:, self.in_channels:].unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out

class UpsampleBlock(nn.Module):
    """Upsampling block with text conditioning"""
    def __init__(self, in_channels, out_channels, text_dim):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.cbn = ConditionalBatchNorm(out_channels, text_dim)
        
    def forward(self, x, text_embedding):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.cbn(x, text_embedding)
        x = F.relu(x)
        return x

class Generator(nn.Module):
    """
    Multi-stage generator for ALR-GAN with progressive resolution
    Generates images at 64x64, 128x128, and 256x256 resolutions
    """
    def __init__(self, text_dim=768, z_dim=100):
        super(Generator, self).__init__()
        
        # Initial projection for noise vector z
        self.fc = nn.Linear(z_dim, 4 * 4 * 512)
        
        # Stage 1: Generate 64x64 image
        # Initial projection for text embedding
        self.text_proj = nn.Linear(text_dim, text_dim)
        
        # Residual blocks with upsampling to 64x64
        self.stage1 = nn.ModuleList([
            UpsampleBlock(512, 512, text_dim),  # 8x8
            UpsampleBlock(512, 256, text_dim),  # 16x16
            UpsampleBlock(256, 128, text_dim),  # 32x32
            UpsampleBlock(128, 64, text_dim)    # 64x64
        ])
        
        # Output conv for stage 1
        self.to_rgb_64 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        # Stage 2: Refine to 128x128
        self.stage2_input = nn.Conv2d(64, 128, 3, 1, 1)
        self.stage2 = nn.ModuleList([
            ResidualBlock(128, 128),
            UpsampleBlock(128, 64, text_dim)  # 128x128
        ])
        
        # Output conv for stage 2
        self.to_rgb_128 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        # Stage 3: Refine to 256x256
        self.stage3_input = nn.Conv2d(64, 128, 3, 1, 1)
        self.stage3 = nn.ModuleList([
            ResidualBlock(128, 128),
            UpsampleBlock(128, 64, text_dim)  # 256x256
        ])
        
        # Output conv for stage 3
        self.to_rgb_256 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def generate_64(self, z, text_embedding):
        """
        Generate 64x64 image
        
        Args:
            z (torch.Tensor): Random noise vector (batch_size, z_dim)
            text_embedding (dict): Text embeddings with sentence and word features
            
        Returns:
            torch.Tensor: Generated 64x64 image
            torch.Tensor: Features for ALR module
        """
        # Extract sentence embedding
        sentence_embedding = text_embedding['sentence_embedding']
        
        # Initial projection
        x = self.fc(z).view(-1, 512, 4, 4)
        
        # Project text embedding
        text_proj = self.text_proj(sentence_embedding)
        
        # Apply stage 1 upsampling blocks
        for block in self.stage1:
            x = block(x, text_proj)
        
        # Features for ALR module
        features = x.clone()
        
        # Generate RGB image
        img = self.to_rgb_64(x)
        
        return img, features
    
    def generate_128(self, img_64, features, text_embedding):
        """
        Generate 128x128 image by refining 64x64 image
        
        Args:
            img_64 (torch.Tensor): 64x64 image from stage 1
            features (torch.Tensor): Features from previous stage or ALR module
            text_embedding (dict): Text embeddings
            
        Returns:
            torch.Tensor: Generated 128x128 image
            torch.Tensor: Features for ALR module
        """
        sentence_embedding = text_embedding['sentence_embedding']
        
        # Process input features
        x = self.stage2_input(features)
        
        # Apply stage 2 blocks
        x = self.stage2[0](x)  # Residual block
        x = self.stage2[1](x, sentence_embedding)  # Upsampling
        
        # Features for ALR module
        features = x.clone()
        
        # Generate RGB image
        img = self.to_rgb_128(x)
        
        return img, features
    
    def generate_256(self, img_128, features, text_embedding):
        """
        Generate 256x256 image by refining 128x128 image
        
        Args:
            img_128 (torch.Tensor): 128x128 image from stage 2
            features (torch.Tensor): Features from previous stage or ALR module
            text_embedding (dict): Text embeddings
            
        Returns:
            torch.Tensor: Generated 256x256 image
            torch.Tensor: Features for next stage (if needed)
        """
        sentence_embedding = text_embedding['sentence_embedding']
        
        # Process input features
        x = self.stage3_input(features)
        
        # Apply stage 3 blocks
        x = self.stage3[0](x)  # Residual block
        x = self.stage3[1](x, sentence_embedding)  # Upsampling
        
        # Generate RGB image
        img = self.to_rgb_256(x)
        
        return img, x
    
    def forward(self, z, text_embedding):
        """
        Forward pass through all stages to generate 256x256 image
        
        Args:
            z (torch.Tensor): Random noise vector
            text_embedding (dict): Text embeddings
            
        Returns:
            tuple: Generated images at different resolutions
        """
        # Generate 64x64 image
        img_64, features_64 = self.generate_64(z, text_embedding)
        
        # Generate 128x128 image
        img_128, features_128 = self.generate_128(img_64, features_64, text_embedding)
        
        # Generate 256x256 image
        img_256, _ = self.generate_256(img_128, features_128, text_embedding)
        
        return img_64, img_128, img_256

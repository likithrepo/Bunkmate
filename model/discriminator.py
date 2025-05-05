import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm:
    """Spectral normalization for GANs stability"""
    def __init__(self, name):
        self.name = name
        self.power_iterations = 1
        
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.view(size[0], -1)
        
        with torch.no_grad():
            for _ in range(self.power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0)
                
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        
        return weight, u
    
    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)
        
        weight = getattr(module, name)
        height = weight.size(0)
        
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0)
        
        delattr(module, name)
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_buffer(name + "_u", u)
        
        module.register_forward_pre_hook(fn)
        
        return fn
    
    def __call__(self, module, inputs):
        weight, u = self.compute_weight(module)
        setattr(module, self.name, weight)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    """Apply spectral normalization to a module"""
    SpectralNorm.apply(module, name)
    return module

class DiscrBlock(nn.Module):
    """Discriminator block with optional downsampling"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscrBlock, self).__init__()
        
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.downsample = downsample
        
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        
    def forward(self, x):
        residual = x
        
        out = self.conv(x)
        out += self.skip(residual)
        
        if self.downsample:
            out = F.avg_pool2d(out, 2)
            
        return out

class Discriminator(nn.Module):
    """
    Multi-scale discriminator with text conditioning for ALR-GAN
    """
    def __init__(self, text_dim=768):
        super(Discriminator, self).__init__()
        
        # Stage 1: 64x64 discriminator
        self.stage1 = nn.ModuleList([
            DiscrBlock(3, 64, downsample=True),            # 32x32
            DiscrBlock(64, 128, downsample=True),          # 16x16
            DiscrBlock(128, 256, downsample=True),         # 8x8
            DiscrBlock(256, 512, downsample=True)          # 4x4
        ])
        
        # Stage 2: 128x128 discriminator
        self.stage2 = nn.ModuleList([
            DiscrBlock(3, 32, downsample=True),            # 64x64
            DiscrBlock(32, 64, downsample=True),           # 32x32
            DiscrBlock(64, 128, downsample=True),          # 16x16
            DiscrBlock(128, 256, downsample=True),         # 8x8
            DiscrBlock(256, 512, downsample=True)          # 4x4
        ])
        
        # Stage 3: 256x256 discriminator
        self.stage3 = nn.ModuleList([
            DiscrBlock(3, 16, downsample=True),            # 128x128
            DiscrBlock(16, 32, downsample=True),           # 64x64
            DiscrBlock(32, 64, downsample=True),           # 32x32
            DiscrBlock(64, 128, downsample=True),          # 16x16
            DiscrBlock(128, 256, downsample=True),         # 8x8
            DiscrBlock(256, 512, downsample=True)          # 4x4
        ])
        
        # Text projection for conditioning
        self.text_proj = spectral_norm(nn.Linear(text_dim, 512))
        
        # Final classification layers
        self.stage1_output = spectral_norm(nn.Linear(512, 1))
        self.stage2_output = spectral_norm(nn.Linear(512, 1))
        self.stage3_output = spectral_norm(nn.Linear(512, 1))
        
    def forward_stage(self, x, blocks, output_layer, text_embedding):
        """Forward pass for a specific stage"""
        # Apply convolution blocks
        for block in blocks:
            x = block(x)
        
        # Global pooling and flatten
        x = torch.mean(x, dim=(2, 3))
        
        # Compute text-image similarity
        text_proj = self.text_proj(text_embedding)
        text_proj = F.normalize(text_proj, dim=1)
        x = F.normalize(x, dim=1)
        
        # Dot product similarity
        similarity = torch.sum(x * text_proj, dim=1, keepdim=True)
        
        # Final output - combine unconditional + text conditioning
        output = output_layer(x) + similarity
        
        return output, x
    
    def forward(self, images, text_embedding):
        """
        Forward pass through the multi-scale discriminator
        
        Args:
            images (tuple): Tuple of (img_64, img_128, img_256)
            text_embedding (dict): Text embeddings
            
        Returns:
            tuple: Classification outputs for each stage
        """
        img_64, img_128, img_256 = images
        sentence_embedding = text_embedding['sentence_embedding']
        
        # Process each stage
        stage1_output, stage1_features = self.forward_stage(
            img_64, self.stage1, self.stage1_output, sentence_embedding
        )
        
        stage2_output, stage2_features = self.forward_stage(
            img_128, self.stage2, self.stage2_output, sentence_embedding
        )
        
        stage3_output, stage3_features = self.forward_stage(
            img_256, self.stage3, self.stage3_output, sentence_embedding
        )
        
        return stage1_output, stage2_output, stage3_output, {
            'stage1': stage1_features,
            'stage2': stage2_features,
            'stage3': stage3_features
        }

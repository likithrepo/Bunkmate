import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import inception_score
import matplotlib.pyplot as plt

# Define image transformation for preprocessing
def get_transforms(image_size=256):
    """
    Get image transformations for preprocessing
    
    Args:
        image_size (int): Target image size
        
    Returns:
        transforms.Compose: Transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def tensor_to_image(tensor):
    """
    Convert a tensor to PIL Image
    
    Args:
        tensor (torch.Tensor): Image tensor in range [-1, 1]
        
    Returns:
        PIL.Image: Converted image
    """
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy and transpose
    if tensor.dim() == 4:  # batch of images
        tensor = tensor[0]  # take first image
    
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    img = img.astype(np.uint8)
    
    return Image.fromarray(img)

def save_image(tensor, path):
    """
    Save a tensor as an image
    
    Args:
        tensor (torch.Tensor): Image tensor
        path (str): Output path
    """
    img = tensor_to_image(tensor)
    img.save(path)

def visualize_attention(img, attention_map, save_path=None):
    """
    Visualize attention map overlaid on image
    
    Args:
        img (torch.Tensor): Image tensor
        attention_map (torch.Tensor): Attention map
        save_path (str, optional): Path to save visualization
        
    Returns:
        plt.Figure: Matplotlib figure with visualization
    """
    # Convert image to numpy
    img_np = tensor_to_image(img)
    
    # Convert attention map to numpy and resize to match image
    if attention_map.dim() == 4:
        attention_map = attention_map[0]  # take first item from batch
    
    if attention_map.dim() == 3:
        attention_map = attention_map.mean(0)  # average across channels
        
    attn_np = attention_map.detach().cpu().numpy()
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax[0].imshow(img_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Plot attention map
    ax[1].imshow(attn_np, cmap="jet")
    ax[1].set_title("Attention Map")
    ax[1].axis("off")
    
    # Plot overlay
    ax[2].imshow(img_np)
    ax[2].imshow(attn_np, cmap="jet", alpha=0.5)
    ax[2].set_title("Overlay")
    ax[2].axis("off")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig

def calculate_inception_score(images, n_splits=10):
    """
    Calculate Inception Score for a batch of images
    
    Args:
        images (torch.Tensor): Batch of images
        n_splits (int): Number of splits for IS calculation
        
    Returns:
        tuple: Mean and standard deviation of IS
    """
    # This is a placeholder - actual implementation would use a pretrained Inception model
    # For proper implementation, we'd need to load pretrained Inception model
    
    # Convert images to numpy
    imgs = []
    for img in images:
        img_np = tensor_to_image(img)
        imgs.append(np.array(img_np))
    
    imgs = np.stack(imgs)
    
    # Placeholder for actual IS calculation
    # In real implementation, we'd use:
    # mean, std = inception_score(imgs, n_splits)
    
    # Return placeholder values
    mean, std = 0.0, 0.0
    
    return mean, std

def calculate_fid(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance between real and generated images
    
    Args:
        real_images (torch.Tensor): Batch of real images
        generated_images (torch.Tensor): Batch of generated images
        
    Returns:
        float: FID score
    """
    # This is a placeholder - actual implementation would use a pretrained Inception model
    # For proper implementation, we'd need to calculate mean and covariance of features
    
    # Placeholder for actual FID calculation
    fid = 0.0
    
    return fid

def calculate_precision_recall(real_features, gen_features, k=5):
    """
    Calculate precision and recall metrics
    
    Args:
        real_features (torch.Tensor): Features from real images
        gen_features (torch.Tensor): Features from generated images
        k (int): Number of nearest neighbors
        
    Returns:
        tuple: Precision and recall values
    """
    # Placeholder for precision-recall calculation
    precision = 0.0
    recall = 0.0
    
    return precision, recall

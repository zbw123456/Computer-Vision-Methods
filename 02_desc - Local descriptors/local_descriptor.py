import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import * 
from local_detector import *


def affine_from_location(b_ch_d_y_x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    B = b_ch_d_y_x.size(0)
    A = torch.zeros((B, 3, 3), device=b_ch_d_y_x.device)
    d = b_ch_d_y_x[:, 2]
    y = b_ch_d_y_x[:, 3]
    x = b_ch_d_y_x[:, 4]
    
    A[:, 0, 0] = d
    A[:, 1, 1] = d
    A[:, 0, 2] = x
    A[:, 1, 2] = y
    A[:, 2, 2] = 1.0
    
    img_idxs = b_ch_d_y_x[:, 0].long().view(-1, 1)
    return A, img_idxs


def affine_from_location_and_orientation(b_ch_d_y_x: torch.Tensor,
                                         ori: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
  B = b_ch_d_y_x.size(0)
    A = torch.zeros((B, 3, 3), device=b_ch_d_y_x.device)
    d = b_ch_d_y_x[:, 2]
    y = b_ch_d_y_x[:, 3]
    x = b_ch_d_y_x[:, 4]
    cos_ori = torch.cos(ori.squeeze())
    sin_ori = torch.sin(ori.squeeze())
    
    A[:, 0, 0] = d * cos_ori
    A[:, 0, 1] = -d * sin_ori
    A[:, 0, 2] = x
    A[:, 1, 0] = d * sin_ori
    A[:, 1, 1] = d * cos_ori
    A[:, 1, 2] = y
    A[:, 2, 2] = 1.0
    
    img_idxs = b_ch_d_y_x[:, 0].long().view(-1, 1)
    return A, img_idxs


def affine_from_location_and_orientation_and_affshape(b_ch_d_y_x: torch.Tensor,
                                                      ori: torch.Tensor,
                                                      aff_shape: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = b_ch_d_y_x.size(0)
    device = b_ch_d_y_x.device
    
    # Extract components
    d = b_ch_d_y_x[:, 2]
    y = b_ch_d_y_x[:, 3]
    x = b_ch_d_y_x[:, 4]
    cos_ori = torch.cos(ori.squeeze())
    sin_ori = torch.sin(ori.squeeze())
    a, b_val, c = aff_shape[:, 0], aff_shape[:, 1], aff_shape[:, 2]
    
    # Compute eigenvalues and eigenvectors for affine shape
    trace = a + c
    det = a * c - b_val**2
    sqrt_disc = torch.sqrt((a - c)**2 + 4 * b_val**2 + 1e-6)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    
    # Avoid division by zero
    safe_lambda1 = torch.sqrt(lambda1 + 1e-6)
    safe_lambda2 = torch.sqrt(lambda2 + 1e-6)
    inv_sqrt_lambda1 = 1.0 / safe_lambda1
    inv_sqrt_lambda2 = 1.0 / safe_lambda2
    
    # Construct inverse square root matrix
    M_inv_sqrt = torch.zeros(B, 2, 2, device=device)
    for i in range(B):
        V = torch.tensor([[b_val[i], b_val[i]],
                          [lambda1[i] - a[i], lambda2[i] - a[i]]], device=device)
        V_norm = V / torch.norm(V, dim=0)
        D = torch.diag(torch.stack([inv_sqrt_lambda1[i], inv_sqrt_lambda2[i]]))
        M_inv_sqrt[i] = V_norm @ D @ V_norm.T
    
    # Build affine matrix
    A = torch.zeros(B, 3, 3, device=device)
    A[:, :2, :2] = M_inv_sqrt
    A[:, 0, 2] = x
    A[:, 1, 2] = y
    A[:, 2, 2] = 1.0
    
    # Combine with orientation and scale
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = cos_ori
    R[:, 0, 1] = -sin_ori
    R[:, 1, 0] = sin_ori
    R[:, 1, 1] = cos_ori
    R[:, 2, 2] = 1.0
    
    S = torch.zeros(B, 3, 3, device=device)
    S[:, 0, 0] = d
    S[:, 1, 1] = d
    S[:, 2, 2] = 1.0
    
    A = A @ R @ S
    img_idxs = b_ch_d_y_x[:, 0].long().view(-1, 1)
    return A, img_idxs


def estimate_patch_dominant_orientation(x: torch.Tensor, num_angular_bins: int = 36):
    B, C, H, W = x.shape
    angles = torch.zeros(B, 1, device=x.device)
    bin_width = 2 * np.pi / num_angular_bins
    
    for i in range(B):
        patch = x[i, 0]
        dx = F.conv2d(patch.unsqueeze(0).unsqueeze(0), 
                      torch.tensor([[[[-1, 0, 1]]]], device=x.device), padding=1)
        dy = F.conv2d(patch.unsqueeze(0).unsqueeze(0), 
                      torch.tensor([[[[-1], [0], [1]]]], device=x.device), padding=1)
        mag = torch.sqrt(dx**2 + dy**2)
        ori = (torch.atan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
        
        hist = torch.histc(ori, bins=num_angular

def create_gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
    """Create 2D Gaussian window for spatial weighting"""
    coords = torch.arange(size, dtype=torch.float32) - size//2
    x = coords.view(1, -1)
    y = coords.view(-1, 1)
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def estimate_patch_affine_shape(x: torch.Tensor) -> torch.Tensor:
    """Estimates affine shape parameters using second moment matrix
    
    Args:
        x: Input patches (B, 1, PS, PS)
        
    Returns:
        Tensor: Affine shape parameters [a, b, c] for each patch (B, 3)
    """
    B, C, PS, _ = x.shape
    device = x.device
    
    # Create Gaussian spatial weighting window
    gauss = create_gaussian_window(PS).to(device).view(1, 1, PS, PS)
    
    # Compute image gradients
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                          dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                          dtype=torch.float32, device=device)
    
    Ix = F.conv2d(x, sobel_x, padding=1)
    Iy = F.conv2d(x, sobel_y, padding=1)
    
    # Compute second moment matrix components
    Ix2 = Ix.pow(2) * gauss
    Iy2 = Iy.pow(2) * gauss
    Ixy = (Ix * Iy) * gauss
    
    # Sum over spatial dimensions
    a = Ix2.sum(dim=(2,3))  # (B, 1)
    b = Ixy.sum(dim=(2,3))  # (B, 1)
    c = Iy2.sum(dim=(2,3))  # (B, 1)
    
    # Combine parameters and handle numerical stability
    eps = 1e-6
    return torch.cat([a, b, c], dim=1) + eps

def calc_sift_descriptor(input: torch.Tensor,
                        num_ang_bins: int = 8,
                        num_spatial_bins: int = 4,
                        clipval: float = 0.2) -> torch.Tensor:
    """Implementation of SIFT descriptor calculation"""
    B, C, H, W = input.shape
    device = input.device
    bin_width = 2 * np.pi / num_ang_bins
    
    # 1. Photometric normalization
    patches = photonorm(input)
    
    # 2. Compute gradients
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                          dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                          dtype=torch.float32, device=device)
    
    dx = F.conv2d(patches, sobel_x, padding=1)
    dy = F.conv2d(patches, sobel_y, padding=1)
    
    # 3. Compute magnitude and orientation
    mag = torch.sqrt(dx.pow(2) + dy.pow(2))
    ori = (torch.atan2(dy, dx) + 2*np.pi) % (2*np.pi)  # [0, 2pi)
    
    # 4. Create Gaussian window
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                                   torch.linspace(-1, 1, W, device=device))
    gaussian = torch.exp(-(grid_x**2 + grid_y**2)/0.5)
    mag *= gaussian[None, None, ...]  # Apply spatial weighting

    # 5. Initialize descriptor tensor
    descriptors = torch.zeros(B, num_spatial_bins**2 * num_ang_bins, device=device)
    spatial_bin_size = H / num_spatial_bins
    
    for b in range(B):
        hist = torch.zeros(num_spatial_bins, num_spatial_bins, num_ang_bins, device=device)
        
        for i in range(H):
            for j in range(W):
                # Spatial bin calculation with soft assignment
                y_pos = (i + 0.5) / spatial_bin_size - 0.5
                x_pos = (j + 0.5) / spatial_bin_size - 0.5
                
                # Spatial bin indices
                y_bin = torch.floor(y_pos).clamp(0, num_spatial_bins-1)
                x_bin = torch.floor(x_pos).clamp(0, num_spatial_bins-1)
                
                # Orientation bin calculation
                angle = ori[b, 0, i, j]
                o_bin = torch.floor(angle / bin_width).long() % num_ang_bins
                o_weight = 1.0  # Could implement soft binning here
                
                # Add weighted magnitude to histogram
                hist[y_bin.long(), x_bin.long(), o_bin] += mag[b, 0, i, j] * o_weight

        # Normalize and clip descriptor
        hist_flat = hist.flatten()
        norm = torch.norm(hist_flat) + 1e-7
        hist_flat = hist_flat / norm
        hist_flat = torch.clamp(hist_flat, 0, clipval)
        hist_flat = hist_flat / (torch.norm(hist_flat) + 1e-7)
        
        descriptors[b] = hist_flat

    return descriptors


def photonorm(x: torch.Tensor) -> torch.Tensor:
    """Normalize patches to zero mean and unit variance with value clipping"""
    # Calculate mean and std over spatial dimensions
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + 1e-6  # Prevent division by zero
    
    # Normalize and clip values
    x_norm = (x - mean) / std
    return torch.clamp(x_norm, -3.0, 3.0)




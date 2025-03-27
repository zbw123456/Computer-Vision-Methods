import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import *

def harris_response(x: torch.Tensor,
                    sigma_d: float,
                    sigma_i: float,
                    alpha: float = 0.04) -> torch.Tensor:
    r"""Computes the Harris cornerness function."""
    # Compute Gaussian gradients
    gradients = spatial_gradient(x, sigma=sigma_d)
    Ix = gradients[:, :, 0]
    Iy = gradients[:, :, 1]
    
    # Compute products of gradients
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy
    
    # Gaussian blur with sigma_i
    Ix2_blur = gaussian_blur2d(Ix2, sigma_i)
    Iy2_blur = gaussian_blur2d(Iy2, sigma_i)
    IxIy_blur = gaussian_blur2d(IxIy, sigma_i)
    
    # Compute determinant and trace
    det = Ix2_blur * Iy2_blur - IxIy_blur ** 2
    trace = Ix2_blur + Iy2_blur
    
    # Harris response
    response = det - alpha * trace ** 2
    return response

def nms2d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression in 3x3 neighborhood."""
    max_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    mask = (x == max_pool) & (x > th)
    return x * mask

def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    r"""Returns coordinates of Harris function maxima."""
    response = harris_response(x, sigma_d, sigma_i)
    nms_response = nms2d(response, th)
    return torch.nonzero(nms_response)

def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    r"""Creates Gaussian scale pyramid."""
    b, ch, h, w = x.size()
    pyramid = torch.zeros(b, ch, n_levels, h, w, device=x.device, dtype=x.dtype)
    sigmas = []
    for i in range(n_levels):
        sigma = sigma_step * math.sqrt(i)
        if sigma == 0:
            pyramid[:, :, i] = x
        else:
            blurred = gaussian_blur2d(x, sigma=sigma)
            pyramid[:, :, i] = blurred
        sigmas.append(sigma)
    return pyramid, sigmas

def nms3d(x: torch.Tensor, th: float = 0):
    r"""Applies 3D non maxima suppression in 3x3x3 neighborhood."""
    max_pool = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    mask = (x == max_pool) & (x > th)
    return x * mask

def scalespace_harris_response(x: torch.Tensor,
                               n_levels: int = 40,
                               sigma_step: float = 1.1):
    r"""Computes Harris response across scale space."""
    pyramid, sigmas = create_scalespace(x, n_levels, sigma_step)
    b, ch, n_levels, h, w = pyramid.shape
    responses = torch.zeros_like(pyramid)
    for i in range(n_levels):
        level_img = pyramid[:, :, i]
        response = harris_response(level_img, sigma_d=sigmas[i], sigma_i=sigmas[i])
        responses[:, :, i] = response
    return responses, sigmas

def scalespace_harris(x: torch.Tensor,
                      th: float = 0,
                      n_levels: int = 40,
                      sigma_step: float = 1.1):
    r"""Returns scale-space Harris corners with scale adaptation."""
    responses, sigmas = scalespace_harris_response(x, n_levels, sigma_step)
    nms_responses = nms3d(responses, th)
    coords = torch.nonzero(nms_responses)
    
    if coords.size(0) == 0:
        return torch.zeros(0, 5, device=x.device)
    
    # Convert scale index to sigma value
    d_indices = coords[:, 2]
    sigmas_tensor = torch.tensor(sigmas, device=x.device, dtype=coords.dtype)
    sigma_values = sigmas_tensor[d_indices.long()]
    
    # Build output tensor [b, c, sigma, h, w]
    return torch.stack([
        coords[:, 0], coords[:, 1], sigma_values,
        coords[:, 3], coords[:, 4]
    ], dim=1)

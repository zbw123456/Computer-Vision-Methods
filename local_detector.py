import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import gaussian_blur, spatial_gradient

def harris_response(x: torch.Tensor,
                    sigma_d: float,
                    sigma_i: float,
                    alpha: float = 0.04) -> torch.Tensor:
    # Compute image gradients
    gradients = spatial_gradient(x, sigma_d)
    Ix = gradients[:, :, 0]
    Iy = gradients[:, :, 1]

    # Compute elements of the structure tensor
    Ix2 = gaussian_blur(Ix.pow(2), sigma_i)
    Iy2 = gaussian_blur(Iy.pow(2), sigma_i)
    Ixy = gaussian_blur(Ix * Iy, sigma_i)

    # Compute determinant and trace
    det = Ix2 * Iy2 - Ixy.pow(2)
    trace = Ix2 + Iy2
    response = det - alpha * trace.pow(2)

    return response

def nms2d(x: torch.Tensor, th: float = 0):
    # Max pooling with 3x3 kernel
    pooled = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    # Keep values equal to max and above threshold
    mask = (x == pooled) & (x > th)
    return x * mask.float()

def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    # Compute Harris response
    response = harris_response(x, sigma_d, sigma_i)
    # Apply NMS
    nms_response = nms2d(response, th)
    # Get coordinates of maxima
    coords = torch.nonzero(nms_response)
    return coords

def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    b, ch, h, w = x.size()
    pyramid = torch.zeros(b, ch, n_levels, h, w, device=x.device)
    sigmas = [sigma_step**i for i in range(n_levels)]
    
    for level in range(n_levels):
        sigma = sigmas[level]
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        pyramid[:, :, level] = gaussian_blur(x, sigma, kernel_size)
        
    return pyramid, sigmas

def nms3d(x: torch.Tensor, th: float = 0):
    # 3D max pooling
    pooled = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    # Keep values equal to max and above threshold
    mask = (x == pooled) & (x > th)
    return x * mask.float()

def scalespace_harris_response(x: torch.Tensor,
                               n_levels: int = 40,
                               sigma_step: float = 1.1):
    # Create scale space
    pyramid, sigmas = create_scalespace(x, n_levels, sigma_step)
    b, ch, _, h, w = pyramid.size()
    responses = torch.zeros(b, ch, n_levels, h, w, device=x.device)
    
    # Compute Harris response at each level
    for level in range(n_levels):
        responses[:, :, level] = harris_response(pyramid[:, :, level], 
                                                sigma_d=sigmas[level], 
                                                sigma_i=sigmas[level])
    return responses, sigmas

def scalespace_harris(x: torch.Tensor,
                      th: float = 0,
                      n_levels: int = 40,
                      sigma_step: float = 1.1):
    # Compute scale-space responses
    responses, sigmas = scalespace_harris_response(x, n_levels, sigma_step)
    # Apply 3D NMS
    nms_responses = nms3d(responses, th)
    # Get coordinates with scale information
    coords = torch.nonzero(nms_responses)
    # Convert scale indices to sigma values
    scale_coords = []
    for coord in coords:
        b, c, d, h, w = coord.tolist()
        scale = sigmas[d]
        scale_coords.append([b, c, scale, h, w])
    
    return torch.tensor(scale_coords)

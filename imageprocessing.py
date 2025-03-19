import torch
import math

def gaussian_kernel(sigma: float, size: int = None) -> torch.Tensor:
    if not size:
        size = 2 * math.ceil(3 * sigma) + 1
    x = torch.arange(size).float() - size//2
    kernel = torch.exp(-x.pow(2)/(2*sigma**2))
    return kernel / kernel.sum()

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma)[None, None, :]
    return F.conv2d(img, kernel, padding=kernel.size(-1)//2)

def spatial_gradient(img: torch.Tensor, sigma: float) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = sobel_x.t()
    return torch.stack([
        F.conv2d(img, sobel_x[None, None], padding=1),
        F.conv2d(img, sobel_y[None, None], padding=1)
    ], dim=2)

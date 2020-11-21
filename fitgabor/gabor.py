import numpy as np
import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

class GaborGenerator(nn.Module):
    def __init__(self, image_size, target_std=1.):
        super().__init__()
        self.theta = nn.Parameter(torch.rand(1))
        self.sigma = nn.Parameter(torch.rand(1)+3., requires_grad=False)
        self.Lambda = nn.Parameter(torch.rand(1)+9.5)
        self.psi = nn.Parameter(torch.rand(1)*torch.pi/2)
        self.gamma = nn.Parameter(torch.zeros(1)+1., requires_grad=False)
        self.center = nn.Parameter(torch.tensor([0., 0.]))
        self.image_size = image_size
        self.target_std = target_std
    
    """
    Returns a gabor filter.
    Args:
        theta (float): Orientation of the sinusoid (in ratian).
        sigma (float): std deviation of the Gaussian.
        Lambda (float): Sinusoid wavelengh (1/frequency).
        psi (float): Phase of the sinusoid.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        image_size (tuple of integers): Image height and width.
        target_std:
    Returns:
        2D Numpy array: A gabor filter.
    """
    
    def forward(self):
        return self.gen_gabor()
    
    def gen_gabor(self):
        
        # clip values in reasonable range
        self.theta.data.clamp_(-torch.pi, torch.pi)
        self.sigma.data.clamp_(3., min(self.image_size)/2) #min(self.image_size)/7, min(self.image_size)/5) #2)
        
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        ymax, xmax = self.image_size
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = torch.meshgrid(torch.arange(ymin, ymax+1), torch.arange(xmin, xmax+1))

        # Rotation
        x_theta = (x - self.center[0]) * torch.cos(self.theta) + (y - self.center[1]) * torch.sin(self.theta)
        y_theta = -(x - self.center[0]) * torch.sin(self.theta) + (y - self.center[1]) * torch.cos(self.theta)

        gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * torch.pi / self.Lambda * x_theta + self.psi)

        return gb
    
    def apply_changes(self):
        self.sigma.requires_grad_(True)
        
    
    
    
def gabor_fn(theta, sigma=2, Lambda=10, psi=np.pi/2, gamma=.8, center=(0, 0), size=(28, 28), normalize=True):
    """Returns a gabor filter.
    Args:
        theta (float): Orientation of the sinusoid (in radian).
        sigma (float): std deviation of the Gaussian.
        Lambda (float): Sinusoid wavelengh (1/frequency).
        psi (float): Phase of the sinusoid.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        image_size (tuple of integers): Image height and width.
        normalize (bool): Whether to normalize the entries. This is computed by
            dividing each entry by the root sum squared of the whole image.
    Returns:
        2D Numpy array: A gabor filter.
    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = size
    xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    # shift the positon
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)

    if normalize:
        # root sum squared
        gb /= np.sqrt(np.sum(gb ** 2))
        # make sure the sum is equal to zero
        # gb[gb > 0] = gb[gb > 0] * (np.abs(gb[gb < 0].sum()) / gb[gb > 0].sum())
        gb -= gb.mean()

    return gb
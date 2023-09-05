import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from keras.datasets.mnist import load_data

from unet import UNet
import matplotlib.pyplot as plt

class DiffusionModel():
    
    def __init__(self, T : int, model : nn.Module, device : str):
        
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device
        
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def training(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """
        
        x0 = sample_batch(batch_size, self.device)
        
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long)
        
        eps = torch.randn_like(x0)
        
        # Take one gradient descent step
        alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_predicted = self.function_approximator(torch.sqrt(
            alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t-1)
        loss = nn.functional.mse_loss(eps, eps_predicted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), use_tqdm=True):
        
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]), 
                         device=self.device)
        
        progress_bar = tqdm if use_tqdm else lambda x : x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t 
            
            beta_t = self.beta[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.function_approximator(x, t-1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
    
        return x
            
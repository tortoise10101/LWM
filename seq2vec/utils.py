import torch
import numpy as np


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)


def loss_kl(mu, logvar):
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)
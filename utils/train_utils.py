import os
import glob
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image


def init(seed):
    """Initialize random seeds and configure the computing device.

    This function sets the random seed for both PyTorch and NumPy to ensure
    reproducibility. It also configures PyTorch's cuDNN backend to operate
    deterministically by disabling benchmark mode, which can introduce
    variability in the results. Finally, it returns the appropriate computing
    device, favoring CUDA if available.

    Args:
        seed (int): The random seed value for initialization.

    Returns:
        torch.device: The device to be used for computation (CUDA if available,
            otherwise CPU).
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def kendall_loss_function(gts, means, logvars):
    """Compute the loss function presented for regression problems in the paper
    titled What Uncertainties Do We Need in Bayesian Deep Learning for Computer
    Vision? by Alex Kendall and Yarin Gal.

    Args:
        gts (torch.Tensor): Ground truth values.
        means (torch.Tensor): Predicted mean values.
        logvars (torch.Tensor): Predicted log variances representing uncertainty.

    Returns:
        torch.Tensor: The computed Kendall loss.
    """
    return torch.mean(((gts - means) ** 2) * torch.exp(-logvars) + logvars)

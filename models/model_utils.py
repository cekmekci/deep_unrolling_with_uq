import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Monte Carlo Dropout module.

    This module applies dropout to the input tensor with a fixed dropout probability `p`.
    Unlike typical dropout layers that only apply dropout during training, this module always
    applies dropout (by setting `training=True`), enabling Monte Carlo sampling during both
    training and inference. This is particularly useful for estimating uncertainty in neural
    network predictions.

    Args:
        p (float, optional): The probability of an element to be zeroed out. Defaults to 0.1.

    Returns:
        torch.Tensor: The output tensor after dropout is applied.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        return nn.functional.dropout(inputs, p=self.p, training=True)

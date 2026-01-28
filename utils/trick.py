import torch
import torch.nn.functional as F


def gumbel_softmax(h, tau=0.1, device="cuda"):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    # p = F.softmax(h, dim=1)
    y = torch.rand(shape_h, device=device) + 1e-25  # ensure all y is positive.
    x = torch.log(y / (1 - y)) + h  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    x = x/tau
    x = F.sigmoid(x)  # now, the x approximates a one_hot vector.
    return x

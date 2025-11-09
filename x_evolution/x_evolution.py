import torch
from torch.nn import Module
from torch.func import functional_call, vmap

from x_mlps_pytorch.noisable import Noisable

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# class

class Population(Module):
    def __init__(self):
        super().__init__()

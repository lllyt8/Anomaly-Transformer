import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        

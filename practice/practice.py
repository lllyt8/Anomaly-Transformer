import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
        
        @property
        def mask(self):
            return self._mask
        

class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        # Calculate distances matrix. Use GPU if available
        if torch.cuda.is_available():
            # zeros is a function that creates a tensor filled with zeros.
            self.distances = torch.zeros((window_size, window_size)).cuda()
        else:
            self.distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)
    # forward is a core function in NNs, we use it to define data input and output.
    def forward(self, queries, keys, values, sigma, attn_mask):
        """
        B: batch size
        L: length of query
        H: number of heads
        E: embedding dimension
        S: length of key
        D: dimension of value
        queries: [B, L, H, E]
        keys: [B, S, H, E]
        values: [B, S, H, D]
        sigma: [B, L, H]
        attn_mask: [B, L, S]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # Calculate attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask

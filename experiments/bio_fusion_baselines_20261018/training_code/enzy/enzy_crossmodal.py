"""Verbatim author definitions; see SOURCES.json for provenance."""
import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

class CrossModalAdapter(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, output_dim):
        super(CrossModalAdapter, self).__init__()
        '''
         query: protein
         key: substrate
        '''
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = self.out_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
        self.bb_node_update = nn.Linear(query_input_dim + output_dim, output_dim)
    
    def forward(self, query_input, key_input, value_input, query_input_mask=None, key_input_mask=None):
        query = self.W_Q(query_input)
        key = self.W_K(key_input)
        value = self.W_V(value_input)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.scale_val
        attn_mask = query_input_mask.unsqueeze(-1) * key_input_mask.unsqueeze(1)
        attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, value)
        
        output = torch.cat([query_input, output], dim=-1)
        output = self.bb_node_update(output)
        
        return output, attn_weights

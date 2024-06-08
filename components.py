import torch 
import torch.nn as nn
import torch.nn.functional as F

def scaled_dpa(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2))

    

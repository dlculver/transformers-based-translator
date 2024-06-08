import torch 
import torch.nn as nn
import torch.nn.functional as F
def scaled_dpa(query, key, value, mask=None):
    """
    Implementation of Scaled Dot-Product Attention from `Attention is All You Need`.
    Args:
        query: (batch_size, num_heads, seq_length, dim_k)
        key: (batch_size, num_heads, seq_length, dim_k)
        value: (batch_size, num_heads, seq_length, dim_v)
        mask:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2)) # Dimension (bs, nh, sl, sl)

    # normalize the scores
    scores = scores/torch.sqrt(torch.tensor(d_k, dtype=torch.float))

    if mask is not None:
        scores = scores.masked_fill(mask==0, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1) # softmax along rows, each row is a probability vector
    output = torch.matmul(attention_weights, value)
    return output, attention_weights



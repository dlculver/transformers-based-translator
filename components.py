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


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model

        # The paper assumes d_k=d_v=d_model/num_heads throughout. They take it to be 64
        self.d_k = d_model // num_heads
        self.query_linear = torch.nn.Linear(d_model, d_model)
        self.key_linaer = torch.nn.Linear(d_model, d_model)
        self.value_linear = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        pass


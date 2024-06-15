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
    
    # Where the mask is 0, we replace the scores entry with -\infty
    if mask is not None:
        scores = scores.masked_fill(mask==0, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1) # softmax along rows, each row is a probability vector
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_model = d_model

        # The paper assumes d_k=d_v=d_model/num_heads throughout. They take it to be 64
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linaer = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query shape is bs, t, d_model
        # key shape is bs, t, d_model
        # value shape is bs, d_model, d_model
        batch_size = query.size(0)
        
        # Apply the linear layers
        query = self.query_linear(query) # shape bs, t, d_model
        key = self.key_linear(key) # shape bs, t, d_model
        value = self.value_linear(value) # shape bs, t, d_model

        # Reshape and split into the number of heads, resulting dimensions bs, num_heads, t, d_k
        # TODO(dominic): There is some interesting technical reasons why you have to reshape this way and then transpose. Do some examples in a notebook to see why. 
        query = query.view(batch_size, -1, self.num_heads, 



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """
    def __init__(self, encoder, decoder, src_embedder, tgt_embedder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.generator = generator # what is this doing??

    def forward(self, src, tgt, src_mask, tgt_mask):
        # explain this one in your own words....
        return self.decod(self.encode(src, src_mask), src_mask, tgt, tgt_mask) # why this signature??

    def encode(self, src, src_mask):
        return self.encoder(self.src_embedder(src), src_mask)

    def decode(self, hstates, src_mask, tgt, tgt_mask): # why does the decoder need the source mask??
        return self.decoder(self.tgt_embedder(tgt), hstates, src_mask, tgt_mask)

 

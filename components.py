import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dpa(query, key, value, mask=None):
    """
    Implementation of Scaled Dot-Product Attention from `Attention is All You Need`.
    Args:
        query: (batch_size, num_heads, seq_length, dim_k)
        key: (batch_size, num_heads, seq_length, dim_k)
        value: (batch_size, num_heads, seq_length, dim_v)
        mask: (batch_size, num_heads, seq_length, seq_length) or None
    Returns:
        output: (batch_size, num_heads, seq_length, dim_v)
        attention_weights: (batch_size, num_heads, seq_length, seq_length)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2))  # Dimension (bs, nh, t, d_k)

    # normalize the scores by the root of the dimension
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float))  # bs, nh, t, t

    # Where the mask is 0, we replace the scores entry with -\infty
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # bs, nh, t, t

    attention_weights = F.softmax(
        scores, dim=-1
    )  # softmax along rows, each row is a probability vector
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class PositionalEncoding(nn.Module):
    """Implementation of the trigonometric positional embeddings from `Attention is all you need`"""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space?? Why?
        pe = torch.zeros(max_len, d_model)  # shape: max_len, d_model
        position = torch.arange(0, max_len).unsqueeze(1)  # shape: max_len, 1
        denominator = torch.exp(
            torch.arange(0, d_model, 2)
            * -(math.log(10**4) / d_model)  # shape: d_model/2
        )
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(
            0
        )  # shape: 1, max_len, d_model. This is necessarily later for broadcasting along the batch dimension.
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(
            False
        )  # shape: (bs, t, d_model). In this case the sequence length is determined by x.size(1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # The paper assumes d_k=d_v=d_model/num_heads throughout. They take it to be 64
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query shape is bs, t, d_model
        # key shape is bs, t, d_model
        # value shape is bs, d_model, d_model
        batch_size = query.size(0)

        # Apply the linear layers
        query = self.query_linear(query)  # shape bs, t, d_model
        key = self.key_linear(key)  # shape bs, t, d_model
        value = self.value_linear(value)  # shape bs, t, d_model

        # Reshape and split into the number of heads, resulting dimensions bs, num_heads, t, d_k
        # TODO(dominic): There is some interesting technical reasons why you have to reshape this way and then transpose. Do some examples in a notebook to see why.
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # bs, num_heads, t, d_k
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # bs, num_heads, t, d_k
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # bs, num_heads, t, d_k

        # Apply scaled dot product attention to each head separately
        attention_output, attention_weight = scaled_dpa(query, key, value, mask)

        # Concatenate the heads
        # attention_output shape: (bs, nh, t, d_k)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.output_linear(attention_output)  # bs, t, d_model


class PositionwiseFFN(nn.Module):
    def __init__(self, d_ff: int, d_model: int, dropout: float = 0.1):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x)))


class EncoderLayer(nn.Module):
    """Encoder layer as in `Attention is All You Need`. We use postlayer normalization."""

    def __init__(self, num_heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, dropout=dropout
        )
        self.ffn = PositionwiseFFN(d_ff=d_ff, d_model=d_model, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Self-attention
        attn_output = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout(attn_output))

        # Feedforward
        ff_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout(ff_output))

        return out2


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for block in self.encoder_blocks:
            x = block(x, mask)

        return self.layernorm(x)


class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, dropout=dropout
        )
        self.src_attn = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, dropout=dropout
        )
        self.ffn = PositionwiseFFN(d_ff=d_ff, d_model=d_model, dropout=dropout)
        self.layernorms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # masked self_attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.layernorms[0](x + self.dropout(self_attn_output))

        # encoder decoder attn
        enc_dec_attn_output = self.src_attn(x, enc_output, enc_output, src_mask)
        x = self.layernorms[1](x + self.dropout(enc_dec_attn_output))

        # feed forward network
        ffn_output = self.ffn(x)
        x = self.layernorms[2](x + self.dropout(ffn_output))

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for block in self.decoder_blocks:
            x = block(x, enc_output, src_mask, tgt_mask)
        return self.layernorm(x)
    
class Generator(nn.Module):
    """Define the standard linear layer followed by softmax generation step. From the havard annotated blog."""
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask=src_mask)
        dec_output = self.decode(tgt, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.generator(dec_output)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), enc_output, src_mask, tgt_mask)


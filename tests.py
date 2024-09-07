import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import itertools
from functools import wraps
import math
import numpy as np


from components import (
    scaled_dpa,
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFFN,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    Generator,
    EncoderDecoder
)

# set random seed
torch.manual_seed(42)
np.random.seed(42)


BATCH_SIZES = [1, 8, 16, 32]
NUM_HEADS = [1, 2, 4, 8]
SEQ_LENGTHS = [10, 50, 100]
DIM_KS = [8, 16, 32, 64]
EMB_SIZES = [64, 128, 256, 512]
DROPOUTS = [0] # can't use other values here since the drop outs occur randomly...
MAXLEN = [500, 1000, 5000]
D_FFS = [10, 50, 100, 500]
NUM_BLOCKS = [1, 2, 4, 8]
VOCAB_SIZES = [100, 1000, 10000]

PARAMS = {
    "batch_size": BATCH_SIZES,
    "num_heads": NUM_HEADS,
    'seq_length': SEQ_LENGTHS,
    "dim_k": DIM_KS,
    "d_model": EMB_SIZES,
    "dropout": DROPOUTS,
    "maxlen": MAXLEN,
    "d_ff": D_FFS,
    "num_blocks": NUM_BLOCKS,
    "vocab_size": VOCAB_SIZES
}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

def generate_params(*param_names):
    def decorator(test_func):
        param_values = [PARAMS[param] for param in param_names]

        combinations = list(itertools.product(*param_values))

        @pytest.mark.parametrize(", ".join(param_names), combinations)
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            return test_func(*args, **kwargs)
        
        return wrapper
    return decorator

def count_parameters(module: torch.nn.Module):
    """Returns the total number of parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

@generate_params("batch_size", "num_heads", "seq_length", "dim_k")
def test_scaled_dpa(batch_size, num_heads, seq_length, dim_k):

    query = torch.rand(batch_size, num_heads, seq_length, dim_k)
    key = torch.rand(batch_size, num_heads, seq_length, dim_k)
    value = torch.rand(batch_size, num_heads, seq_length, dim_k)
    mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    # my implementation
    output, attention_weights = scaled_dpa(query, key, value, mask)

    # builtin pytorch implementation
    torch_output = F.scaled_dot_product_attention(query, key, value, mask)

    assert torch.allclose(output, torch_output, atol=1e-6), "Output does not match PyTorch's implementation"
    assert output.dtype == torch_output.dtype, "Dtypes do not match!"


# the following implementation of positional encodings was taken from https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
class TorchPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(TorchPositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# @generate_params("batch_size", "seq_length", "d_model", "dropout", "maxlen")
# def test_positional_encodings(batch_size, seq_length, d_model, dropout, maxlen):

#     token_emb = torch.rand(batch_size, seq_length, d_model)

#     my_pe = PositionalEncoding(d_model, dropout, maxlen)

#     my_output = my_pe(token_emb)

#     torch_pe = TorchPositionalEncoding(d_model, dropout, maxlen)
#     torch_output = torch_pe(token_emb)

#     assert my_output.shape == torch_output.shape, "Outputs don't have matching shape. "
#     assert my_output.dtype == torch_output.dtype, "Dtypes do not match!"
#     assert torch.allclose(my_output, torch_output, atol=1e-6), "Output does not match PyTorch's implementation."


    
@generate_params("batch_size", "num_heads", "seq_length", "d_model")
def test_multihead_attention(batch_size, num_heads, seq_length, d_model):

    query = torch.rand(batch_size, seq_length, d_model).to(DEVICE)
    key = torch.rand(batch_size, seq_length, d_model).to(DEVICE)
    value = torch.rand(batch_size, seq_length, d_model).to(DEVICE)

    # Instantiate your custom MultiHeadAttention
    my_mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model).to(DEVICE)
    
    # Instantiate PyTorch's MultiheadAttention
    torch_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True).to(DEVICE)

    # Manually copy weights from torch_mha to your my_mha
    my_mha.query_linear.weight.data = torch_mha.in_proj_weight[:d_model, :].detach().clone().to(DEVICE)
    my_mha.key_linear.weight.data = torch_mha.in_proj_weight[d_model:2*d_model, :].detach().clone().to(DEVICE)
    my_mha.value_linear.weight.data = torch_mha.in_proj_weight[2*d_model:, :].detach().clone().to(DEVICE)

    my_mha.query_linear.bias.data = torch_mha.in_proj_bias[:d_model].detach().clone().to(DEVICE)
    my_mha.key_linear.bias.data = torch_mha.in_proj_bias[d_model:2*d_model].detach().clone().to(DEVICE)
    my_mha.value_linear.bias.data = torch_mha.in_proj_bias[2*d_model:].detach().clone().to(DEVICE)

    # Output linear layer
    my_mha.output_linear.weight.data = torch_mha.out_proj.weight.detach().clone().to(DEVICE)
    my_mha.output_linear.bias.data = torch_mha.out_proj.bias.detach().clone().to(DEVICE)

    # Forward pass through both models
    my_output = my_mha(query, key, value)
    torch_output, _ = torch_mha(query, key, value)

    # Verify shape, dtype, and closeness of outputs
    assert my_output.shape == torch_output.shape, "MHA output doesn't have the same shape as PyTorch's implementation."
    assert my_output.dtype == torch_output.dtype, "MHA outputs don't have matching dtypes!"
    assert torch.allclose(my_output, torch_output, atol=1e-6), "Outputs do not match PyTorch's implementation."


@generate_params("batch_size", "seq_length", "num_heads", "d_ff", "d_model", "dropout")
def test_encoder_layer(batch_size, seq_length, num_heads, d_ff, d_model, dropout):

    src = torch.rand(batch_size, seq_length, d_model).to(DEVICE)

    my_encoder_layer = EncoderLayer(num_heads=num_heads, d_ff=d_ff, d_model=d_model, dropout=dropout, testing_mode=True).to(DEVICE)
    torch.manual_seed(42)
    np.random.seed(42)
    torch_encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True).to(DEVICE)

    my_output = my_encoder_layer(src)
    torch_output = torch_encoder_layer(src)

    assert my_output.shape == torch_output.shape, "Encoder Layer shape doesn't match PyTorch's implementation."
    assert my_output.dtype == torch_output.dtype, "Encoder Layer's output dtype doesn't match PyTorch's implementation."
    assert torch.allclose(my_output, torch_output, atol=1e-6), "Outputs do not match"

# @pytest.mark.parametrize("batch_size, seq_length, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
# def test_decoder_layer(batch_size, seq_length, num_heads, d_model, d_ff, dropout):

#     tgt = torch.rand(batch_size, seq_length, d_model)
#     enc_output = torch.rand(batch_size, seq_length, d_model)

#     my_decoder_layer = DecoderLayer(num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
#     torch_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)

#     my_output = my_decoder_layer(tgt, enc_output)
#     torch_output = torch_decoder_layer(tgt, enc_output)

#     assert my_output.shape == torch_output.shape, "Decoder Layer shape doesn't match Pytorch's implementation."
#     assert my_output.dtype == torch_output.dtype, "Decoder Layer's output's dtype doesn't match PyTorch's."

# @pytest.mark.parametrize("batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_BLOCKS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
# def test_encoder(batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout):

#     src = torch.rand(batch_size, seq_length, d_model)

#     my_encoder = Encoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
#     torch_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout), num_blocks)

#     my_output = my_encoder(src)
#     torch_output = torch_encoder(src)

#     assert my_output.shape == torch_output.shape, "Encoder output doesn't have the same shape as PyTorch's implementation."
#     assert my_output.dtype == torch_output.dtype, "Encoder output's dtype doesn't match PyTorch's."

# @pytest.mark.parametrize("batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_BLOCKS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
# def test_decoder(batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout):

#     tgt = torch.rand(batch_size, seq_length, d_model)
#     enc_output = torch.rand(batch_size, seq_length, d_model)

#     my_decoder = Decoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
#     torch_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout), num_blocks)

#     my_output = my_decoder(tgt, enc_output)
#     torch_output = torch_decoder(tgt, enc_output)

#     assert my_output.shape == torch_output.shape, "Decoder output doesn't have the same shape as PyTorch's implementation."
#     assert my_output.dtype == torch_output.dtype, "Decoder's output's dtype doesn't match PyTorch's."

# @pytest.mark.parametrize("batch_size, seq_length, vocab_size, num_blocks, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, VOCAB_SIZES, NUM_BLOCKS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
# def test_transformer(batch_size, seq_length, vocab_size, num_blocks, num_heads, d_model, d_ff, dropout):
#     pass 
    
#     src_embed = nn.Sequential(nn.Embedding(vocab_size, d_model), PositionalEncoding(d_model, dropout))
#     tgt_embed = nn.Sequential(nn.Embedding(vocab_size, d_model), PositionalEncoding(d_model, dropout))
    
#     my_transformer = EncoderDecoder(
#         Encoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff = d_ff, dropout=dropout),
#         Decoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout),
#         src_embed=src_embed, 
#         tgt_embed=tgt_embed, 
#         generator=Generator(d_model, vocab_size)
#         )
    
#     torch_transformer = nn.Transformer(
#         d_model=d_model, 
#         nhead=num_heads, 
#         num_encoder_layers=num_blocks, 
#         num_decoder_layers=num_blocks,
#         dim_feedforward=d_ff, 
#         dropout=dropout, 
#         batch_first=True
#     )

#     # Create random token tensors for `src` and `tgt` for your custom transformer model
#     src_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
#     tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
#     # Create random embedded tensors for `src` and `tgt` for the PyTorch transformer model
#     src_embeddings = torch.rand(batch_size, seq_length, d_model)
#     tgt_embeddings = torch.rand(batch_size, seq_length, d_model)
    
#     # Forward pass through your custom transformer
#     my_output = my_transformer(src_tokens, tgt_tokens, src_mask=None, tgt_mask=None)
    
#     # Forward pass through the PyTorch transformer
#     torch_output = torch_transformer(src_embeddings, tgt_embeddings)
    
#     # Check if the output shapes match
#     assert my_output.shape == torch_output.shape, f"Output shapes do not match: {my_output.shape} vs {torch_output.shape}"
#     assert my_output.dtype == torch_output.dtype, f"Transformer output dtypes do not match: {my_output.dtype} vs {torch_output.dtype}"

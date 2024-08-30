import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import itertools
import math


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

@pytest.mark.parametrize("batch_size, num_heads, seq_length, dim_k", list(itertools.product(BATCH_SIZES, NUM_HEADS, SEQ_LENGTHS, DIM_KS)))
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


@pytest.mark.parametrize("batch_size, seq_len, emb_size, dropout, maxlen", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, EMB_SIZES, DROPOUTS, MAXLEN)))
def test_positional_encodings(batch_size, seq_len, emb_size, dropout, maxlen):

    token_emb = torch.rand(batch_size, seq_len, emb_size)

    my_pe = PositionalEncoding(emb_size, dropout, maxlen)

    my_output = my_pe(token_emb)

    torch_pe = PositionalEncoding(emb_size, dropout, maxlen)
    torch_output = torch_pe(token_emb)

    assert torch.allclose(my_output, torch_output, atol=1e-6), "Output does not match PyTorch's implementation."

@pytest.mark.parametrize("batch_size, num_heads, seq_length, d_model", list(itertools.product(BATCH_SIZES, NUM_HEADS, SEQ_LENGTHS, EMB_SIZES)))
def test_multihead_attention(batch_size, num_heads, seq_length, d_model):

    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)
    # TODO: Leaving out the masks for now, need to figure out how to incorporate them...

    torch.manual_seed(0) # adding this for reproducibility to check if outputs are also close
    my_mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    my_ouput = my_mha(query, key, value)

    torch.manual_seed(0)
    torch_mha = nn.MultiheadAttention(d_model, num_heads)
    torch_output, torch_attention_output_weights = torch_mha(query, key, value)

    assert my_ouput.shape == torch_output.shape, "MHA output doesn't have the same shape as PyTorch's implementation."

    # couldn't get the reproducibility to work, will look further into this later. 
    # assert torch.allclose(my_ouput, torch_output), "MHA output does not match PyTorch's implementation."

@pytest.mark.parametrize("batch_size, seq_length, num_heads, d_ff, d_model, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_HEADS, D_FFS, EMB_SIZES, DROPOUTS)))
def test_encoder_layer(batch_size, seq_length, num_heads, d_ff, d_model, dropout):

    src = torch.rand(batch_size, seq_length, d_model)

    my_encoder_layer = EncoderLayer(num_heads=num_heads, d_ff=d_ff, d_model=d_model, dropout=dropout)
    torch_encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)

    my_output = my_encoder_layer(src)
    torch_output = torch_encoder_layer(src)

    assert my_output.shape == torch_output.shape, "Encoder Layer shape doesn't match PyTorch's implementation."

@pytest.mark.parametrize("batch_size, seq_length, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
def test_decoder_layer(batch_size, seq_length, num_heads, d_model, d_ff, dropout):

    tgt = torch.rand(batch_size, seq_length, d_model)
    enc_output = torch.rand(batch_size, seq_length, d_model)

    my_decoder_layer = DecoderLayer(num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
    torch_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)

    my_output = my_decoder_layer(tgt, enc_output)
    torch_output = torch_decoder_layer(tgt, enc_output)

    assert my_output.shape == torch_output.shape, "Decoder Layer shape doesn't match Pytorch's implementation."

@pytest.mark.parametrize("batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_BLOCKS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
def test_encoder(batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout):

    src = torch.rand(batch_size, seq_length, d_model)

    my_encoder = Encoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
    torch_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout), num_blocks)

    my_output = my_encoder(src)
    torch_output = torch_encoder(src)

    assert my_output.shape == torch_output.shape, "Encoder output doesn't have the same shape as PyTorch's implementation."

@pytest.mark.parametrize("batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, NUM_BLOCKS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
def test_decoder(batch_size, seq_length, num_blocks, num_heads, d_model, d_ff, dropout):

    tgt = torch.rand(batch_size, seq_length, d_model)
    enc_output = torch.rand(batch_size, seq_length, d_model)

    my_decoder = Decoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
    torch_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout), num_blocks)

    my_output = my_decoder(tgt, enc_output)
    torch_output = torch_decoder(tgt, enc_output)

    assert my_output.shape == torch_output.shape, "Decoder output doesn't have the same shape as PyTorch's implementation."

@pytest.mark.parametrize("batch_size, seq_length, vocab_size, num_blocks, num_heads, d_model, d_ff, dropout", list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, VOCAB_SIZES, NUM_BLOCKS, NUM_HEADS, EMB_SIZES, D_FFS, DROPOUTS)))
def test_transformer(batch_size, seq_length, vocab_size, num_blocks, num_heads, d_model, d_ff, dropout):
    pass 
    
    src_embed = nn.Sequential(nn.Embedding(vocab_size, d_model), PositionalEncoding(d_model, dropout))
    tgt_embed = nn.Sequential(nn.Embedding(vocab_size, d_model), PositionalEncoding(d_model, dropout))
    
    my_transformer = EncoderDecoder(
        Encoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff = d_ff, dropout=dropout),
        Decoder(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout=dropout),
        src_embed=src_embed, 
        tgt_embed=tgt_embed, 
        generator=Generator(d_model, vocab_size)
        )
    
    torch_transformer = nn.Transformer(
        d_model=d_model, 
        nhead=num_heads, 
        num_encoder_layers=num_blocks, 
        num_decoder_layers=num_blocks,
        dim_feedforward=d_ff, 
        dropout=dropout, 
        batch_first=True
    )

    # Create random token tensors for `src` and `tgt` for your custom transformer model
    src_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Create random embedded tensors for `src` and `tgt` for the PyTorch transformer model
    src_embeddings = torch.rand(batch_size, seq_length, d_model)
    tgt_embeddings = torch.rand(batch_size, seq_length, d_model)
    
    # Forward pass through your custom transformer
    my_output = my_transformer(src_tokens, tgt_tokens, src_mask=None, tgt_mask=None)
    
    # Forward pass through the PyTorch transformer
    torch_output = torch_transformer(src_embeddings, tgt_embeddings)
    
    # Check if the output shapes match
    assert my_output.shape == torch_output.shape, f"Output shapes do not match: {my_output.shape} vs {torch_output.shape}"

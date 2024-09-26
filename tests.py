import torch
import torch.nn as nn
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
    EncoderDecoder,
)

# set random seed
torch.manual_seed(42)


def test_scaled_dpa():
    batch_size = 2
    num_heads = 4
    seq_length = 5
    dim_k = 8
    query = torch.rand(batch_size, num_heads, seq_length, dim_k)
    key = torch.rand(batch_size, num_heads, seq_length, dim_k)
    value = torch.rand(batch_size, num_heads, seq_length, dim_k)
    mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    output, attention_weights = scaled_dpa(query, key, value, mask)

    assert output.shape == (batch_size, num_heads, seq_length, dim_k)
    assert attention_weights.shape == (batch_size, num_heads, seq_length, seq_length)


def test_positional_encoding():
    d_model = 16
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    pe = PositionalEncoding(d_model, dropout)
    x = torch.rand(batch_size, seq_length, d_model)

    out = pe(x)
    assert out.shape == (batch_size, seq_length, d_model)


def test_multi_head_attention():
    batch_size = 2
    num_heads = 4
    seq_length = 10
    d_model = 16
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)
    mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    mha = MultiHeadAttention(num_heads, d_model)
    output = mha(query, key, value, mask)

    assert output.shape == (batch_size, seq_length, d_model)


def test_positionwise_ffn():
    d_ff = 32
    d_model = 16
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    ffn = PositionwiseFFN(d_ff, d_model, dropout)
    x = torch.rand(batch_size, seq_length, d_model)

    out = ffn(x)
    assert out.shape == (batch_size, seq_length, d_model)


def test_encoder_layer():
    num_heads = 4
    d_model = 16
    d_ff = 32
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    x = torch.rand(batch_size, seq_length, d_model)
    mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    encoder_layer = EncoderLayer(num_heads, d_model, d_ff, dropout)
    out = encoder_layer(x, mask)

    assert out.shape == (batch_size, seq_length, d_model)


def test_encoder():
    num_blocks = 2
    num_heads = 4
    d_model = 16
    d_ff = 32
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    x = torch.rand(batch_size, seq_length, d_model)
    mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    encoder = Encoder(num_blocks, num_heads, d_model, d_ff, dropout)
    out = encoder(x, mask)

    assert out.shape == (batch_size, seq_length, d_model)


def test_decoder_layer():
    num_heads = 4
    d_model = 16
    d_ff = 32
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    x = torch.rand(batch_size, seq_length, d_model)
    enc_output = torch.rand(batch_size, seq_length, d_model)
    src_mask = torch.ones(batch_size, num_heads, seq_length, seq_length)
    tgt_mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    decoder_layer = DecoderLayer(num_heads, d_model, d_ff, dropout)
    out = decoder_layer(x, enc_output, src_mask, tgt_mask)

    assert out.shape == (batch_size, seq_length, d_model)


def test_decoder():
    num_blocks = 2
    num_heads = 4
    d_model = 16
    d_ff = 32
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    x = torch.rand(batch_size, seq_length, d_model)
    enc_output = torch.rand(batch_size, seq_length, d_model)
    src_mask = torch.ones(batch_size, num_heads, seq_length, seq_length)
    tgt_mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    decoder = Decoder(num_blocks, num_heads, d_model, d_ff, dropout)
    out = decoder(x, enc_output, src_mask, tgt_mask)

    assert out.shape == (batch_size, seq_length, d_model)


def test_generator():
    d_model = 16
    vocab_size = 50
    seq_length = 10
    batch_size = 2
    x = torch.rand(batch_size, seq_length, d_model)

    generator = Generator(d_model, vocab_size)
    out = generator(x)

    assert out.shape == (batch_size, seq_length, vocab_size)


def test_encoder_decoder():
    num_blocks = 2
    num_heads = 4
    d_model = 16
    d_ff = 32
    vocab_size = 50
    seq_length = 10
    batch_size = 2
    dropout = 0.1
    src = torch.randint(0, vocab_size, (batch_size, seq_length))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_length))
    src_mask = torch.ones(batch_size, num_heads, seq_length, seq_length)
    tgt_mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

    encoder = Encoder(num_blocks, num_heads, d_model, d_ff, dropout)
    decoder = Decoder(num_blocks, num_heads, d_model, d_ff, dropout)
    src_embed = nn.Embedding(vocab_size, d_model)
    tgt_embed = nn.Embedding(vocab_size, d_model)
    generator = Generator(d_model, vocab_size)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
    out = model(src, tgt, src_mask, tgt_mask)

    assert out.shape == (batch_size, seq_length, vocab_size)


if __name__ == "__main__":
    test_scaled_dpa()
    test_positional_encoding()
    test_multi_head_attention()
    test_positionwise_ffn()
    test_encoder_layer()
    test_encoder()
    test_decoder_layer()
    test_decoder()
    test_generator()
    test_encoder_decoder()

    print("All tests passed!")

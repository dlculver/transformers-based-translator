import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


def scaled_dpa(query, key, value, mask=None, verbose=False):
    """
    Implements scaled dot product attention.
    Args:
        query: (batch_size, seq_length, dim_k)
        key: (batch_size, seq_length, dim_k)
        value: (batch_size, seq_length, dim_v)
        mask: (batch_size, seq_length) or None
        verbose: Boolean default False
    Returns:
        attention_output: (batch_size, seq_length, dim_v)
        attention_weights: (batch_size, seq_length, seq_length)
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # (bs, seq_length, seq_length)

    if verbose:
        print(f"Scores shape: {scores.shape}")

    # apply the mask if necessary
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # apply softmax to get attention_weights
    attention_weights = F.softmax(scores, dim=-1)  # (bs, seq_length, seq_length)

    if verbose:
        print(f"Attention weights shape: {attention_weights.shape}")

    output = torch.matmul(attention_weights, value)

    if verbose:
        print(f"Attention output shape: {output.shape}")

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
            * -(math.log(10000.0) / d_model)  # shape: d_model/2
        )
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(
            0
        )  # shape: 1, max_len, d_model. This is necessarily later for broadcasting along the batch dimension.
        self.register_buffer("pe", pe)  # what is this for?

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].requires_grad_(
            False
        )  # shape: (bs, t, d_model). In this case the sequence length is determined by x.size(1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout=0.1, verbose=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.verbose = verbose

        if self.verbose:
            print(f"Num heads: {num_heads}")
            print(f"Embedding dimension: {d_model}")
            print(f"per head dimension: {self.d_k}")

        # linear layers to project the inputs to query, key, and value
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query shape is bs, seq_length, d_model
        # key shape is bs, seq_length, d_model
        # value shape is bs, d_model, d_model
        batch_size = query.size(0)
        seq_length = query.size(1)

        # need to unsqueeze at 1 so as to broadcast the same mask across all heads.
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask applied to all heads.

        if self.verbose and mask is not None:
            print(f"Mask shape (after unsqueezing at 1): {mask.shape}")

        # apply linear layers
        query = self.query_linear(query)  # shape: bs, seq_length, d_model
        key = self.key_linear(key)  # shape: bs, seq_length, d_model
        value = self.value_linear(value)  # shape: bs, d_model, d_model

        if self.verbose:
            print(f"Query shape: {query.shape}")
            print(f"Key shape: {key.shape}")
            print(f"Value shape: {value.shape}")

        # reshape and split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (bs, num_heads, seq_length, d_k)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (bs, num_heads, seq_length, d_k)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (bs, num_heads, seq_length, d_k)

        if self.verbose:
            print(f"Shapes after projections for query, key, value...")
            print(f"{query.shape}, {key.shape}, {value.shape}")

        attn_output, attn_weights = scaled_dpa(
            query, key, value, mask, verbose=self.verbose
        )

        # we've separated the query key and value into separate heads and then computed the scaled dot-product attention for each head.
        # Now we must put them back together.
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        if self.verbose:
            print(f"Attention output shape after concat: {attn_output.shape}")

        # apply the final linear layer transformation
        output = self.output_linear(attn_output)
        if self.verbose:
            print(f"Output shape: {output.shape}")

        return output, attn_weights


class PositionwiseFFN(nn.Module):
    def __init__(self, d_ff: int, d_model: int, dropout: float = 0.1):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        verbose: bool = False,
    ):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, dropout=dropout, verbose=verbose
        )
        self.ffn = PositionwiseFFN(d_ff=d_ff, d_model=d_model, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose

    def forward(self, x, mask=None):
        if self.verbose:
            print(f"Input to Encoder Layer: {x.shape}")

        # Multi-head attention with residual connection and layer normalization
        attn_output, _ = self.mha(x, x, x, mask)
        if self.verbose:
            print(f"attn_output shape: {attn_output.shape}")
        out1 = self.layernorm1(x + self.dropout(attn_output))

        # Feedforward with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(
            out1 + self.dropout(ffn_output)
        )  # Fixed: add out1, not x

        if self.verbose:
            print(f"Output from Encoder Layer: {out2.shape}")

        return out2


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        verbose: bool = False,
    ):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.verbose = verbose

        # encoder layers
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    verbose=verbose,
                )
                for _ in range(num_blocks)
            ]
        )

        # final layer normalization layer
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):

        if self.verbose:
            print(f"Input of shape: {x.shape}")

        for i, block in enumerate(self.encoder_blocks):
            if self.verbose:
                print(
                    f"\n------------ Passing Through Encoder block {i + 1} ----------------"
                )

            x = block(x, mask=src_mask)

        # apply final layer normalization
        x = self.layernorm(x)

        if self.verbose:
            print(f"\nFinal output shape is: {x.shape}")

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        verbose=False,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, dropout=dropout, verbose=verbose
        )
        self.src_attn = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, dropout=dropout, verbose=verbose
        )
        self.ffn = PositionwiseFFN(d_ff=d_ff, d_model=d_model, dropout=dropout)
        self.layernorms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):

        if self.verbose:
            print(f"Input shape x: {x.shape}")
            print(f"Encoder output shape: {enc_output.shape}\n")
        # masked self-attention over the target (with look-ahead mask)

        if self.verbose:
            print(f"Passing through self-attention")
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.layernorms[0](x + self.dropout(self_attn_output))

        if self.verbose:
            print(f"\nPassing Through encoder-decoder attention")
        # encoder-decoder attention over the encoder output (attend to source)
        enc_dec_attn_output, _ = self.src_attn(x, enc_output, enc_output, src_mask)
        x = self.layernorms[1](x + self.dropout(enc_dec_attn_output))

        if self.verbose:
            print(f"\nFinal feedforward of layer")
        # feedforward with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = self.layernorms[2](x + self.dropout(ffn_output))

        if self.verbose:
            print(f"\nOutput shape: {x.shape}")

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        verbose: bool = True,
    ):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    verbose=verbose,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layernorm = nn.LayerNorm(d_model)
        self.verbose = verbose

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        for i, block in enumerate(self.decoder_blocks):
            if self.verbose:
                print(
                    f"\n------------- Passing Through Decoder Block {i+1} ----------------"
                )
            tgt = block(tgt, enc_output, src_mask, tgt_mask)

        return self.layernorm(tgt)


class Generator(nn.Module):
    """Define the standard linear layer followed by softmax generation step. From the havard annotated blog."""

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        generator: Generator,
        embedding: nn.Embedding,
        pos_encoder: PositionalEncoding,
        verbose: bool = False,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.embedding = embedding
        self.pos_encoder = pos_encoder
        self.verbose = verbose

    def forward(self, src_tokens, tgt_input, src_mask, tgt_mask):
        # Encoder
        encoder_output = self.encode(src_tokens, src_mask)

        # Decoder
        tgt_embed = self.embedding(tgt_input)
        tgt_embed = self.pos_encoder(tgt_embed)
        dec_output = self.decoder(
            tgt=tgt_embed,
            enc_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        output_log_probs = self.generator(dec_output)

        return output_log_probs

    def encode(self, src_tokens, src_mask):
        """Pass source and source maskthrough the encoder
        Args:
            src_tokens -- torch.Tensor
            src_mask -- torch.Tensor
        Output:
            encoder_output -- torch.Tensor
        """
        src_embed = self.embedding(src_tokens)
        src_embed = self.pos_encoder(src_embed)
        return self.encoder(src_embed, src_mask)

    def decode(self, tgt_tokens, enc_output, src_mask, tgt_mask):
        """Pass tgt_tokens and encoder hidden state through the decoder
        Args:
            tgt_tokens: torch.Tensor
            enc_output: torch.Tensor
            src_mask: torch.Tensor
            tgt_mask: torch.Tensor
        Output:
            decoder_output: torch.Tensor"""
        tgt_embed = self.embedding(tgt_tokens)
        tgt_embed = self.pos_encoder(tgt_embed)
        return self.decoder(tgt_embed, enc_output, src_mask, tgt_mask)

    def embed_tokens(self, tokens: torch.Tensor):
        """embeds and adds positional encodings for tokens."""
        embed = self.embedding(tokens)
        embed = self.pos_encoder(embed)
        return embed

    @classmethod
    def from_hyperparameters(
        cls,
        num_blocks,
        num_heads,
        d_model,
        d_ff,
        vocab_size,
        max_len: int = 512,
        dropout: float = 0.1,
        verbose: bool = False,
    ):
        """Class method to initialize a new model."""
        # create encoder
        encoder = Encoder(num_blocks, num_heads, d_model, d_ff, dropout, verbose)

        # create decoder
        decoder = Decoder(num_blocks, num_heads, d_model, d_ff, dropout, verbose)

        # create embedding and positional encoder
        embedding = nn.Embedding(vocab_size, d_model)
        pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # create generator
        generator = Generator(d_model, vocab_size)

        return cls(
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            embedding=embedding,
            pos_encoder=pos_encoder,
            verbose=verbose,
        )


class Translator:
    def __init__(
        self,
        model: EncoderDecoder,
        tokenizer,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.bos = bos_token_id
        self.eos = eos_token_id
        self.pad = pad_token_id
        self.device = device

    def decode(self, src_sentence, max_len: int = 500, mode: str = "greedy"):
        if mode == "greedy":
            inferred_tokens = self._greedy_decode(src_sentence, max_len)
            return self._ids_to_text(inferred_tokens)
        else:
            raise NotImplementedError("Mode not yet implemented")

    def _tokenize_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text).ids
        tokens = [self.bos] + tokens + [self.eos]  # add bos and eos tokens
        tokens = torch.tensor(tokens)

        return tokens

    def _ids_to_text(self, tokens: torch.Tensor | List[int]) -> str:
        """Decode a torch tensor or list of integers into text"""
        # TODO(dominic): currently this only implements tokenization for a 1-dimensional tensor, should generalize to batches?
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        # remove special tokens
        tokens = [
            token for token in tokens if token not in [self.bos, self.eos, self.pad]
        ]
        return self.tokenizer.decode(tokens)

    def _greedy_decode(self, src_sentence, max_len: int = 500):
        self.model.eval()

        src_tokens = self._tokenize_text(src_sentence)
        src_mask = (src_tokens != self.pad).unsqueeze(0).to(self.device)

        # store encoder hidden states for the src_tokens
        encoder_output = self.model.encode(src_tokens=src_tokens, src_mask=src_mask)

        # initialize the tgt_tokens
        tgt_tokens = torch.tensor([self.bos], dtype=torch.long).to(self.device)

        for _ in range(max_len):

            # create target mask
            tgt_seq_len = tgt_tokens.size(0)
            tgt_mask = torch.tril(torch.ones(1, tgt_seq_len, tgt_seq_len)).to(
                self.device
            )

            tgt_embed = self.model.embed_tokens(tgt_tokens)

            output_logits = self.model.decoder(
                tgt_embed, encoder_output, src_mask, tgt_mask
            )
            output_log_probs = self.model.generator(output_logits)

            next_token = torch.argmax(output_log_probs[:, -1, :], dim=-1)
            tgt_tokens = torch.cat([tgt_tokens, next_token])

            if next_token.item() == self.eos or tgt_tokens.size(0) >= max_len:
                break

        return tgt_tokens

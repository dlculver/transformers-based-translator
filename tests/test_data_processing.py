import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import pytest
import random

import data_processing as dp

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

SEED = 42
MAX_LENGTH = 512

torch.manual_seed(seed=SEED)

@pytest.fixture(scope='module')
def setup_data():

    tokenizer = ByteLevelBPETokenizer(
            "bpe_tokenizer/vocab.json", 
            "bpe_tokenizer/merges.txt"
    )

    PAD_TOKEN_ID = tokenizer.token_to_id("<pad>")
    BOS_TOKEN_ID = tokenizer.token_to_id("<s>")
    EOS_TOKEN_ID = tokenizer.token_to_id("</s>")
    src_lang = "de"
    tgt_lang = "en"

    dataset = load_dataset("wmt14", "de-en")
    train_ds = dp.TranslationDataset(
        dataset["train"].shuffle().select(range(100)),
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        max_length=MAX_LENGTH
    )
    val_ds = dp.TranslationDataset(
        dataset["validation"].shuffle().select(range(10)),
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        max_length=MAX_LENGTH
    )

    return PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, train_ds, val_ds


def test_special_tokens(setup_data):
    pad, bos, eos, train_ds, val_ds = setup_data

    indices = random.sample(range(100), 10)

    for idx in indices:
        example = train_ds[idx]
        src_tokens = example['src_tokens']
        tgt_tokens = example['tgt_tokens']

        assert src_tokens[0] == bos, f'source tokens do not begin with bos token at {idx}: {src_tokens}'
        assert src_tokens[0] == bos, f'target tokens do not begin with bos token at {idx}: {tgt_tokens}'

        eos_token_src = (src_tokens == eos).nonzero(as_tuple=True)[0].item()
        assert all(token == pad for token in src_tokens[eos_token_src+1:]), f"One of the source tokens don't have only pads after the eos token."
        
        eos_token_tgt = (tgt_tokens == eos).nonzero(as_tuple=True)[0].item()
        assert all(token == pad for token in tgt_tokens[eos_token_tgt + 1:]), f"One of the target tokens don't have only pads after the eos token."

def test_pad_and_truncation(setup_data):
    pad, bos, eos, train_ds, val_ds = setup_data

    # create training dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size = 16,
        shuffle=True,
        collate_fn=lambda batch: dp.collate_fn(batch, pad_token_id=pad)
    )

    for batch in train_dl:
        src_tokens = batch['src_tokens']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']
        src_mask = batch['src_mask']
        tgt_mask = batch['tgt_mask']

        assert src_tokens.shape[1] == MAX_LENGTH
        assert tgt_input.shape[1] == MAX_LENGTH - 1



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset, DatasetDict

from tokenizers import ByteLevelBPETokenizer

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer: ByteLevelBPETokenizer, src_lang: str, tgt_lang: str, bos_token_id: int, eos_token_id: int, pad_token_id: int, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.bos = bos_token_id
        self.eos= eos_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_sentence = self.dataset[idx]['translation'][self.src_lang]
        tgt_sentence = self.dataset[idx]['translation'][self.tgt_lang]

        # tokenize the source and target
        src_tokens = self.tokenizer.encode(src_sentence).ids
        tgt_tokens = self.tokenizer.encode(tgt_sentence).ids

        # pad and truncate
        src_tokens = torch.tensor(self.pad_and_truncate(self.add_special_tokens(src_tokens)))
        tgt_tokens = torch.tensor(self.pad_and_truncate(self.add_special_tokens(tgt_tokens)))

        return {
            'src_sentence': src_sentence,
            'tgt_sentence': tgt_sentence,
            'src_tokens': src_tokens,
            'tgt_tokens': tgt_tokens,
        }

    def pad_and_truncate(self, tokens):
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return tokens

    def add_special_tokens(self, tokens):
        return [self.bos] + tokens + [self.eos]

# Functions for creating masks and collating into a dataloader

def create_causal_mask(size):
    """
    Creates a causal mask that prevents attending to future tokens. 
    Args: 
        size: length of the sequence
    Returns:
        torch.Tensor: causal mask of shape (1, size, size)
    """
    attn_shape = (1, size, size)
    return torch.tril(torch.ones(attn_shape)).type(torch.uint8)

def create_std_mask(tgt, pad_token_id: int):
    tgt_mask = (tgt != pad_token_id).unsqueeze(-2)
    tgt_mask = tgt_mask & create_causal_mask(tgt.size(-1))
    return tgt_mask

def collate_fn(batch, pad_token_id: int):
    src_batch = torch.stack([item['src_tokens'] for item in batch])
    tgt_batch = torch.stack([item['tgt_tokens'] for item in batch])

    # create masks
    src_mask = (src_batch != pad_token_id).unsqueeze(-2).int() # shape: (bs, 1, seq_length)
    tgt = tgt_batch[:, :-1]
    tgt_y = tgt_batch[:, 1:]
    tgt_mask = create_std_mask(tgt, pad_token_id=pad_token_id)

    return {
        'src_tokens': src_batch,
        'tgt_input': tgt,
        'tgt_output': tgt_output, 
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
    }


import torch
from torch.utils.data import Dataset

import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, pad_token_id:int, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_sentence = self.dataset[idx]['translation']['de']
        tgt_sentence = self.dataset[idx]['translation']['en']

        # tokenize the source and target
        src_tokens = self.tokenizer.encode(src_sentence).ids
        tgt_tokens = self.tokenizer.encode(tgt_sentence).ids

        # pad and truncate
        src_tokens = torch.tensor(self.pad_and_truncate(src_tokens))
        tgt_tokens = torch.tensor(self.pad_and_truncate(tgt_tokens))

        # create attention masks
        src_mask = (src_tokens != self.pad_token_id).int()
        tgt_mask = (src_tokens != self.pad_token_id).int()

        # create look ahead mask
        look_ahead_mask = self.create_causal_mask(len(tgt_tokens))


        return {
            'src_sentence': src_sentence, 
            'tgt_sentence': tgt_sentence, 
            'src_tokens': src_tokens,
            'tgt_tokens': tgt_tokens,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'look_ahead_mask': look_ahead_mask,
            'combined_mask': tgt_mask & look_ahead_mask
        }

    def pad_and_truncate(self, tokens):
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return tokens

    def create_causal_mask(self, size):
        # create an lower triangular matrix for the purposes of look ahead masking
        return torch.tril(torch.ones(size, size)).type(torch.uint8)
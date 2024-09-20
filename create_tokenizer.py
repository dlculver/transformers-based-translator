from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm 
import os

from typing import List

DEFAULT_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def create_corpus(dataset, corpus_path: str):
    """
    Creates a corpus from BOTH the source and target languages to create a common tokenizer.
    """

    with open(corpus_path, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset['train']):
            for lang, text in example['translation'].items():
                f.write(text + "\n") 


def train_tokenizer(corpus_path: str, save_path: str, vocab_size: int = 37000, min_freq: int = 2, special_tokens: List[str] = DEFAULT_SPECIAL_TOKENS):

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
            files=[corpus_path],
            vocab_size=vocab_size, 
            min_frequency=min_freq, 
            special_tokens=special_tokens,
    )

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_model(save_path)

def main():

    dataset = load_dataset("wmt14", "de-en")
    corpus_path = 'train_texts.txt'
    create_corpus(dataset, corpus_path)

    train_tokenizer(corpus_path, "new_tokenizer")

    print(f"Finished training BPE tokenizer!")



if __name__ == "__main__":
    main()

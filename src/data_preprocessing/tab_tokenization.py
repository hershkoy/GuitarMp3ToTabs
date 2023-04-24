import os
import numpy as np
from glob import glob
from collections import Counter
import guitarpro

class TabTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def fit(self, tabs):
        chars = set()
        for tab in tabs:
            chars.update(set(tab))
        chars = sorted(list(chars))

        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

    def tokenize(self, tab):
        return [self.char2idx[char] for char in tab]

    def detokenize(self, token_seq):
        return ''.join(self.idx2char[token] for token in token_seq)

    def save(self, file_path):
        np.savez(file_path, char2idx=self.char2idx, idx2char=self.idx2char)

    def load(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.char2idx = data['char2idx'].item()
        self.idx2char = data['idx2char'].item()




def tokenize_tabs(input_dir, output_dir, tokenizer_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tab_files = glob(os.path.join(input_dir, "*.txt"))
    tabs = []

    for tab_file in tab_files:
        with open(tab_file, "r", encoding="utf-8") as f:
            tab = f.read()
            tabs.append(tab)

    tokenizer = TabTokenizer()
    tokenizer.fit(tabs)
    tokenizer.save(tokenizer_path)

    for tab_file, tab in zip(tab_files, tabs):
        tokens = tokenizer.tokenize(tab)
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(tab_file))[0] + '.npy')
        np.save(output_file, tokens)
        print(f"Tokenized {tab_file} and saved to {output_file}")


if __name__ == "__main__":
    input_dir = "../data/tabs"
    output_dir = "../data/preprocessed/tokenized_tabs"
    tokenizer_path = "tab_tokenizer.npz"

    tokenize_tabs(input_dir, output_dir, tokenizer_path)

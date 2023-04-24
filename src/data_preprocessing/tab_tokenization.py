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

def gpx_to_tab_text(gpx_file):
    tab_text = ""
    gp = guitarpro.parse(gpx_file)

    for track in gp.tracks:
        for measure in track.measures:
            tab_text += f"Track {track.number}: {track.name}\n"
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        tab_text += f"{note.string}-{note.value} "
                    tab_text += "\n"
    return tab_text

def gpx_to_measure_tabs(gpx_file):
    gp = guitarpro.parse(gpx_file)
    measure_tabs = []

    for track in gp.tracks:
        for measure in track.measures:
            measure_tab = ""

            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        measure_tab += f"{note.string}-{note.value} "
                    measure_tab += "\n"

            measure_tabs.append(measure_tab)
    return measure_tabs


def tokenize_tabs(input_dir, output_dir, tokenizer_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tab_files = glob(os.path.join(input_dir, "*.gpx"))
    all_measure_tabs = []

    for tab_file in tab_files:
        measure_tabs = gpx_to_measure_tabs(tab_file)
        all_measure_tabs.extend(measure_tabs)

    tokenizer = TabTokenizer()
    tokenizer.fit(all_measure_tabs)
    tokenizer.save(tokenizer_path)

    for tab_file in tab_files:
        measure_tabs = gpx_to_measure_tabs(tab_file)
        tokens_list = [tokenizer.tokenize(measure_tab) for measure_tab in measure_tabs]
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(tab_file))[0] + '.npy')
        np.save(output_file, tokens_list)
        print(f"Tokenized {tab_file} and saved to {output_file}")

if __name__ == "__main__":
    input_dir = os.path.join('..', 'data', 'tabs')
    output_dir = os.path.join('..', 'data', 'preprocessed', 'tokenized_tabs') 
    tokenizer_path = "tab_tokenizer.npz"

    tokenize_tabs(input_dir, output_dir, tokenizer_path)

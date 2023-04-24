import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def synchronize_and_split_data(audio_input_dir, tab_input_dir, output_dir, test_size=0.1, random_state=42):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = sorted(glob(os.path.join(audio_input_dir, "*.npy")))
    tab_files = sorted(glob(os.path.join(tab_input_dir, "*.npy")))

    if len(audio_files) != len(tab_files):
        raise ValueError("The number of audio files and tab files must be equal.")

    X = []
    Y = []

    for audio_file, tab_file in zip(audio_files, tab_files):
        audio_features = np.load(audio_file, allow_pickle=True)
        tokenized_tabs = np.load(tab_file, allow_pickle=True)

        if len(audio_features) != len(tokenized_tabs):
            raise ValueError(f"File {audio_file} and {tab_file} must have the same number of measures.")

        for feature, tab in zip(audio_features, tokenized_tabs):
            X.append(feature)
            Y.append(tab)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(output_dir, "Y_test.npy"), Y_test)

if __name__ == "__main__":
    audio_input_dir = "../data/preprocessed/audio_features"
    tab_input_dir = "../data/preprocessed/tokenized_tabs"
    output_dir = "../data/preprocessed/synchronized"

    synchronize_and_split_data(audio_input_dir, tab_input_dir, output_dir)

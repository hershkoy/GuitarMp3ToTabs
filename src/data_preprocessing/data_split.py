import os
import random
import shutil
from glob import glob


def split_data(preprocessed_data_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    audio_features_path = os.path.join(preprocessed_data_path, "audio_features")
    tokenized_tabs_path = os.path.join(preprocessed_data_path, "tokenized_tabs")
    
    # Create output directories if they don't exist
    for subdir in ["train", "val", "test"]:
        os.makedirs(os.path.join(audio_features_path, subdir), exist_ok=True)
        os.makedirs(os.path.join(tokenized_tabs_path, subdir), exist_ok=True)

    # Get list of audio feature and tokenized tab files
    audio_feature_files = sorted(glob(os.path.join(audio_features_path, "*.npy")))
    tokenized_tab_files = sorted(glob(os.path.join(tokenized_tabs_path, "*.npy")))

    # Assert that audio feature files and tokenized tab files have the same length
    assert len(audio_feature_files) == len(tokenized_tab_files), "Mismatch between audio features and tokenized tabs"

    # Shuffle the list of files using a seed for reproducibility
    random.seed(42)
    combined = list(zip(audio_feature_files, tokenized_tab_files))
    random.shuffle(combined)
    audio_feature_files[:], tokenized_tab_files[:] = zip(*combined)

    # Calculate the number of files for each split
    total_files = len(audio_feature_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    # Function to move files to output directories
    def move_files(file_list, output_subdir):
        for file_path in file_list:
            shutil.move(file_path, os.path.join(os.path.dirname(file_path), output_subdir, os.path.basename(file_path)))

    # Move files to the respective output directories
    move_files(audio_feature_files[:train_count], "train")
    move_files(tokenized_tab_files[:train_count], "train")

    move_files(audio_feature_files[train_count:train_count + val_count], "val")
    move_files(tokenized_tab_files[train_count:train_count + val_count], "val")

    move_files(audio_feature_files[train_count + val_count:], "test")
    move_files(tokenized_tab_files[train_count + val_count:], "test")

    print("Data split complete.")


if __name__ == "__main__":
    preprocessed_data_path = "../data/preprocessed"
    split_data(preprocessed_data_path)

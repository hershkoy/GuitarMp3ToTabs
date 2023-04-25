import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from .seq2seq import create_seq2seq_model

# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def train_model(model):

    # Load and preprocess data
    def load_data(audio_features_dir, tokenized_tabs_dir):
        X = []
        y = []

        audio_feature_files = sorted(os.listdir(audio_features_dir))
        tokenized_tab_files = sorted(os.listdir(tokenized_tabs_dir))

        for feature_file, tab_file in zip(audio_feature_files, tokenized_tab_files):
            feature_path = os.path.join(audio_features_dir, feature_file)
            tab_path = os.path.join(tokenized_tabs_dir, tab_file)

            X.append(np.load(feature_path))
            y.append(np.load(tab_path))

        return X, y

    def preprocess_data(X, y, max_output_sequence_length):
        X = pad_sequences(X, padding='post', dtype='float32')
        y = pad_sequences(y, maxlen=max_output_sequence_length, padding='post', dtype='int32')

        X = np.array(X)
        y = np.array(y)
        
        y_input = y[:, :-1]
        y_output = y[:, 1:]

        # Convert y_output to one-hot encoding
        y_output_one_hot = to_categorical(y_output)

        return X, y_input, y_output_one_hot

    # Model training parameters
    batch_size = 64
    epochs = 100
    learning_rate = 0.001

    # Replace these paths with the paths to your preprocessed data
    train_audio_features_dir = os.path.join('..', 'data', 'preprocessed', 'audio_features', 'train') 
    train_tokenized_tabs_dir = os.path.join('..', 'data', 'preprocessed', 'tokenized_tabs', 'train') 

    # Load and preprocess the data
    X, y = load_data(train_audio_features_dir, train_tokenized_tabs_dir)
    X, y_input, y_output_one_hot = preprocess_data(X, y, max_output_sequence_length=100)

    optimizer = Adam(learning_rate=learning_rate)
    loss = CategoricalCrossentropy()
    accuracy = CategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    # Train the model
    history = model.fit([X, y_input], y_output_one_hot, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # Save the model
    model.save('trained_seq2seq_model.h5')


if __name__ == "__main__":
    train_model()
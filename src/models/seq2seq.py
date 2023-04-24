import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


def create_seq2seq_model(input_dim, output_dim, latent_dim=256):
    # Encoder
    encoder_inputs = Input(shape=(None, input_dim))
    encoder_lstm = LSTM(latent_dim, return_state=True)
    _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    encoder_states = [encoder_state_h, encoder_state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, output_dim))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the seq2seq model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


if __name__ == "__main__":
    input_dim = 13  # Replace with the dimension of your input features (e.g., MFCCs)
    output_dim = 100  # Replace with the size of your tokenized tab vocabulary
    latent_dim = 256

    seq2seq_model = create_seq2seq_model(input_dim, output_dim, latent_dim)
    seq2seq_model.summary()

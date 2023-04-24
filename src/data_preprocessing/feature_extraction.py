import os
import numpy as np
import pretty_midi
import librosa
from glob import glob


def extract_mfcc_from_midi(midi_path, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Synthesize the MIDI data into audio
    audio_data = midi_data.synthesize(fs=sr)
    
    # Extract MFCCs from the audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    
    return mfccs


def extract_audio_features(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.midi'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.npy')
                mfccs = extract_mfcc_from_midi(input_path)
                np.save(output_path, mfccs)
                print(f"Extracted features from {input_path} and saved to {output_path}")


if __name__ == "__main__":
    input_dir = "../data/midi"
    output_dir = "../data/preprocessed/audio_features"
    extract_audio_features(input_dir, output_dir)

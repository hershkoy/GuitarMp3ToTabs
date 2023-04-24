import os
import librosa
import pretty_midi
import numpy as np


def mp3_to_midi(input_path, output_path, sr=22050, hop_length=512, n_fft=2048):
    # Load the mp3 file
    audio_data, _ = librosa.load(input_path, sr=sr, mono=True)
    
    # Extract the chromagram
    chromagram = librosa.feature.chroma_stft(audio_data, sr=sr, hop_length=hop_length, n_fft=n_fft)
    
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Define a starting time
    start_time = 0.0
    
    # Iterate through the chromagram frames
    for frame in chromagram.T:
        # Get the pitch class with the highest energy
        pitch_class = np.argmax(frame)
        
        # Create a new note
        note = pretty_midi.Note(velocity=100, pitch=pitch_class + 60, start=start_time, end=start_time + 0.5)
        
        # Add the note to the MIDI file
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(note)
        midi.instruments.append(instrument)
        
        # Increment the start time
        start_time += 0.5
    
    # Write the MIDI file
    midi.write(output_path)


def convert_audio_to_midi(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.midi')
                mp3_to_midi(input_path, output_path)
                print(f"Converted {input_path} to {output_path}")


if __name__ == "__main__":
    input_dir = "../data/raw_audio/artist"  # Replace with the path to your .mp3 files
    output_dir = "../data/midi"
    convert_audio_to_midi(input_dir, output_dir)

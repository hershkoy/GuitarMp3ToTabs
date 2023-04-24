import os
import librosa
import numpy as np
from glob import glob

def process_audio(input_file, output_file, time_signature):
    y, sr = librosa.load(input_file, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y, sr)
    measures = group_beats_into_measures(beats, time_signature)
    
    measure_features = []
    for measure in measures:
        start, end = measure
        y_measure = y[start:end]
        mfcc = librosa.feature.mfcc(y_measure, sr, n_mfcc=13)
        measure_features.append(mfcc.T)
    
    np.save(output_file, measure_features)

def group_beats_into_measures(beats, time_signature):
    measure_beat_count = time_signature[0]
    measures = []
    for i in range(0, len(beats), measure_beat_count):
        if i + measure_beat_count >= len(beats):
            break
        start = beats[i]
        end = beats[i + measure_beat_count]
        measures.append((start, end))
    return measures

def convert_all_audio_files(input_dir, output_dir, time_signature=(4, 4)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = glob(os.path.join(input_dir, "*.mp3"))

    for audio_file in audio_files:
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0] + '.npy')
        process_audio(audio_file, output_file, time_signature)
        print(f"Processed {audio_file} and saved to {output_file}")

if __name__ == "__main__":
    input_dir = os.path.join('..', 'data', 'audio')  
    output_dir = os.path.join('..', 'data', 'preprocessed', 'audio_features')  
    
    
    # Adjust time_signature if needed
    time_signature = (6, 8)

    convert_all_audio_files(input_dir, output_dir, time_signature)


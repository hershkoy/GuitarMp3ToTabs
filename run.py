import librosa
import numpy as np
import aubio

# Load and preprocess the audio file
audio_file = 'song.mp3'
audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)

print("step1")

# Set parameters for pitch detection
window_size = 512
hop_size = 128

# Create the pitch detection object
pitch_detection = aubio.pitch("default", window_size, hop_size, sample_rate)
pitch_detection.set_unit("midi")

print("step2")

# Perform pitch detection
pitches = []
for i in range(0, len(audio_data), hop_size):
    window = audio_data[i:i + hop_size]  # Change this line to use hop_size instead of window_size
    
    # Pad the window if necessary
    if len(window) < hop_size:
        window = np.pad(window, (0, hop_size - len(window)))

    pitch = pitch_detection(window)[0]
    pitches.append(pitch)

print("step3")

# Convert pitches to notes
notes = [round(p) for p in pitches]
note_names = [librosa.midi_to_note(n) for n in notes]

# Initialize guitar tab strings
tab_lines = ['-' * len(notes) for _ in range(6)]

# Convert notes to guitar tab format
def note_to_guitar_tab(note, position):
    strings = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
    string_idx = 0
    min_diff = float('inf')
    
    for i, string_note in enumerate(strings):
        diff = librosa.note_to_midi(note) - librosa.note_to_midi(string_note)
        if 0 <= diff < min_diff:
            min_diff = diff
            string_idx = i
    
    fret = min_diff
    return string_idx, fret

print("step4")


# Update guitar tab strings with notes
for i, note in enumerate(note_names):
    string, fret = note_to_guitar_tab(note, i)
    tab_lines[string] = tab_lines[string][:i] + str(fret) + tab_lines[string][i+1:]

print("step5")

# Output the guitar tabs
with open('output_tabs_formatted.txt', 'w') as f:
    print("1")
    for line_idx, line in enumerate(reversed(tab_lines)):
        for i in range(0, len(line), 40):
            f.write(f'{line[i:i+40]}\n')
        if line_idx < len(tab_lines) - 1:
            f.write('\n')
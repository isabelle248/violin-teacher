# PART 2 - use pitch tracker to get frequencies from audio input

# Import libraries
import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter
import os
import logging
import sys
import google.generativeai as genai
from scipy.stats import mode
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=key)

# Suppress TensorFlow / CREPE / matplotlib warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TF debug logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

mxl_file = sys.argv[1]
audio_path = sys.argv[2]
tempo = int(sys.argv[3])

import matplotlib.pyplot as plt

# reads MusicXML file and turns it into Score object with notes, rests, measures
# score (object, hierarchical) contains Parts (instruments/voices)
# each Part contains Measures
# each Measure contains Notes or Rests
score = converter.parse(mxl_file)

# flatten hierarchical structure into a simple sequence (?)
notes_and_rests = score.flat.notesAndRests


# Load audio with librosa
# reads audio file from disk, resamples to 16,000 hz
# returns: y - actual audio waveform data (numpy array of floats between -1.0 and 1.0)
# sr - sampling rate (16,000 kHz - 16,000 samples per sec)
y, sr = librosa.load(audio_path, sr=16000)

# Run CREPE pitch tracking on audio waveform (y)
# inputs: y (waveform), sr, step_size=10 (analyze pitch every 10 ms), 
#          model_capacity='full' (highest accuracy CREPE model)
time, frequency, confidence, activation = crepe.predict(y, sr, step_size=7, model_capacity='full')

# TIME SHIFT

# filter out values that are less confident
threshold = 0.75
confident_idx = confidence > threshold
filtered_time = time[confident_idx]
filtered_freq = frequency[confident_idx]
filtered_conf = confidence[confident_idx]


# Visualize CREPE confidence vs. time
plt.figure(figsize=(12, 4))

plt.plot(time, confidence, label="CREPE Confidence", color="gray", alpha=0.6)
plt.scatter(filtered_time, filtered_conf, color="blue", s=10, label="Kept (confidence > threshold)")

plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")

plt.title("CREPE Confidence Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.legend()
plt.grid(True)
plt.show()

# # Find first confident prediction
# first_conf_idx = np.argmax(confidence > 0.8)
# first_conf_time = time[first_conf_idx]

#get first note
first_note = None
for n in notes_and_rests:
    if n.isRest or n.isChord:
        continue
    first_note = n
    break

first_note_freq = first_note.pitch.frequency
first_note_start = (first_note.offset / tempo) * 60

def cents_diff(f1, f2):
    return 1200 * np.log2(f1 / f2)

diff_cents = np.abs(cents_diff(filtered_freq, first_note_freq))

# Find first index where CREPE freq is within 100 cents of first note
match_idx = np.where(diff_cents <= 100)[0]

if len(match_idx) == 0:
    print("No prediction matched the first note.")
else:
    first_match_time = filtered_time[match_idx[0]]
    print(f"First CREPE match for {first_note.nameWithOctave} at {first_match_time:.3f} s")
    print(first_match_time)

print(filtered_time)
filtered_time = filtered_time - first_match_time
print(filtered_time)

print(f"Shifted CREPE times so first note starts at 0 (shift = {first_match_time:.3f} sec)")
print("match_idx: " + str(match_idx[0]))
print("diff_cents: " + str(diff_cents))
print("first_note_freq: " + str(first_note_freq))

# adjust so no zero values
valid_idx = filtered_time >= 0
adjusted_time = filtered_time[valid_idx]
adjusted_freq = filtered_freq[valid_idx]
adjusted_conf = filtered_conf[valid_idx]


# create empty list, each item is a tuple (offset, note_frequency)
note_data = []
all_differences = []
all_times = []

measure_numbers = []
note_names = []
cents_list = []
reasoning_lines = []
predicted_freqs = []
note_freqs = []

def process_note_frequencies(pred_freqs, target_freq):
    return np.median(pred_freqs)
    

# n is one note (or chord) at a time
for n in notes_and_rests:

    # store time for rests
    if n.isRest:
        all_times.append((n.offset / tempo) * 60)
        predicted_freqs.append(np.nan)
        note_freqs.append(np.nan)        # keep same length
        continue

    # Skip chords
    if n.isChord:
        continue

    note_start = (n.offset / tempo) * 60
    note_end = note_start + ((n.quarterLength / tempo) * 60)
    
    # get slice boundaries of CREPE frames that fall in note duration
    # finds first index where time >= note_start
    start_idx = np.searchsorted(adjusted_time, note_start, side='left')
    # finds index after last time that is >= note_end
    end_idx = np.searchsorted(adjusted_time, note_end, side='right')

    # slice frequency array to get all CREPE pitch values in note duration
    freq_slice = (adjusted_freq[start_idx:end_idx])

    # slice time
    time_slice = (adjusted_time[start_idx:end_idx])

    valid_idx = freq_slice > 0
    corrected_freq = np.nan  # default if no valid predictions


    # take median cents difference per note
    if np.any(valid_idx):
        corrected_freq = process_note_frequencies(freq_slice[valid_idx], n.pitch.frequency)
        
        note_cents = 1200 * np.log2(corrected_freq / n.pitch.frequency)
    else:
        # no valid predictions for this note
        note_cents = np.nan 

    # add this difference to all_differences array
    all_differences.append(note_cents)
    all_times.append(note_start)
    predicted_freqs.append(corrected_freq)

    # collect data for feedback output later
    measure_numbers.append(n.measureNumber)
    note_names.append(n.nameWithOctave)
    note_freqs.append(n.pitch.frequency)

    # classify pitch accuracy
    tolerance = 25
    if not np.isnan(note_cents):
        if abs(note_cents) <= tolerance:
            status = "In tune"
        elif note_cents > 0:
            status = "Too sharp"
        else:
            status = "Too flat"
        
        # Store data for reasoning
        reasoning_lines.append(
            f"Measure {n.measureNumber} | {n.nameWithOctave} at {note_start} : {status} ({note_cents:.2f} cents)"
        )

    print(f"Note: {n.nameWithOctave}, Target: {n.pitch.frequency:.2f} Hz, Median Detected: {corrected_freq:.2f} Hz, Cents: {note_cents:.2f}")

measures = ", ".join(str(n) for n in measure_numbers)
notes = ", ".join(str(n) for n in note_names)
final_differences = ", ".join(str(n) for n in all_differences)
final_times = ", ".join(str(n) for n in all_times)

print(measures)
print(notes)
print(final_differences)
print(final_times)

prompt = f"""
You are a violin teacher analyzing a student’s recording.

Here is the data:
- Measures: {measures}
- Notes: {notes}
- Pitch differences (in cents, positive = sharp, negative = flat): {final_differences}
- Timestamps (in seconds): {final_times}

Each index corresponds to the same note. For example:
Measure[i], Note[i], PitchDifference[i], Timestamp[i].

Please:
1. Identify notes that are significantly out of tune (greater than ±25 cents).
2. Generate specific feedback in plain English:
   - Note that it was supposed to be, timestamp, measure, sharp/flat
   - Suggest what to practice
3. End with an overall summary of their intonation.

If there are no notes out of tune, give a positive and supporting message like "Great job! All of your notes were in tune, keep up the good work!"
Don't make anything bolded.
"""

# Create the Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Generate content
response = model.generate_content(prompt)

print(response.text)

# print reasoning
print("\n")
print("Reasoning - Model output for each note:")
print("=" * 40)
print("\n".join(reasoning_lines))
print("=" * 40)

def plot_pitch_comparison(crepe_times, crepe_freqs, note_times, note_freqs, corrected_times, corrected_freqs):
    plt.figure(figsize=(14, 6))

    # Plot CREPE predictions
    plt.scatter(crepe_times, crepe_freqs, color="blue", s=8, alpha=0.6, label="CREPE Predicted Frequencies")

    # Plot target (sheet music) frequencies
    plt.scatter(note_times, note_freqs, color="red", s=50, marker="x", label="Target Note Frequencies")

    # Plot corrected crepe predictons
    plt.scatter(corrected_times, corrected_freqs, color="green", s=50, marker="x", label="Median of CREPE Predictions per note")


    # Make it pretty
    plt.title("Pitch Tracking: CREPE Predictions vs. Target Notes")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)

    plt.show()

plot_pitch_comparison(adjusted_time, adjusted_freq, all_times, note_freqs, all_times, predicted_freqs)
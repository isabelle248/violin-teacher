# PART 2 - use pitch tracker to get frequencies from audio input

# Import libraries
import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter
import random
import os
import logging
import json
import math
# converter is module in music21 that parses scores from files
import sys
from scipy.interpolate import interp1d
from scipy.signal import correlate
from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyBegRoTXaFwsjbEENHNFNqVpJ-uJmt-SHY",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# Suppress TensorFlow / CREPE / matplotlib warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TF debug logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

mxl_file = sys.argv[1]
audio_path = sys.argv[2]
tempo = int(sys.argv[3])

import matplotlib.pyplot as plt

# plot function
def plot_alignment(filtered_time_before, filtered_freq, filtered_time_after, gt_times, gt_freqs):
    plt.figure(figsize=(14, 6))

    # Plot ground truth frequencies as a step function
    plt.step(gt_times, gt_freqs, where='post', color="red", linewidth=2, label="Ground Truth (MusicXML)")

    # Plot CREPE predictions BEFORE shift
    plt.scatter(filtered_time_before, filtered_freq, color="blue", s=10, alpha=0.5, label="CREPE Before Alignment")

    # Plot CREPE predictions AFTER shift
    plt.scatter(filtered_time_after, filtered_freq, color="green", s=10, alpha=0.7, label="CREPE After Alignment")

    # Labels & Legend
    plt.title("Alignment of CREPE Predictions with Ground Truth")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
threshold = 0.9
confident_idx = confidence > threshold
filtered_time = time[confident_idx]
filtered_freq = frequency[confident_idx]
filtered_conf = confidence[confident_idx]

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


filtered_time = filtered_time - first_match_time

print(f"Shifted CREPE times so first note starts at 0 (shift = {first_match_time:.3f} sec)")

# adjust so no zero values
valid_idx = filtered_time >= 0
adjusted_time = filtered_time[valid_idx]
adjusted_freq = filtered_freq[valid_idx]
adjusted_conf = filtered_conf[valid_idx]




# Compare predicted frequencies to ground truth


# # Time alignment (shift CREPE predictions to best match ground truth)
# gt_times_no_rests = []
# gt_freqs_no_rests = []




# OLD TIME SHIFT CODE
# # store ground truth times and frequencies in arrays
# for n in notes_and_rests:
#     if n.isRest:
#         continue

#     if n.isChord:
#         continue

#     note_start = (n.offset / tempo) * 60
#     freq = float(n.pitch.frequency)

#     if not np.isnan(freq):
#         gt_times_no_rests.append(note_start)
#         gt_freqs_no_rests.append(freq)


# # Convert to NumPy arrays for consistency
# gt_times_no_rests = np.array(gt_times_no_rests, dtype=np.float64)
# gt_freqs_no_rests = np.array(gt_freqs_no_rests, dtype=np.float64)

# # Remove any NaNs, None, or invalid numbers
# valid_mask = np.isfinite(gt_times_no_rests) & np.isfinite(gt_freqs_no_rests)
# gt_times_no_rests = gt_times_no_rests[valid_mask]
# gt_freqs_no_rests = gt_freqs_no_rests[valid_mask]

# # Sort by time to ensure strictly increasing x-values for interp1d
# sort_idx = np.argsort(gt_times_no_rests)
# gt_times_no_rests = gt_times_no_rests[sort_idx]
# gt_freqs_no_rests = gt_freqs_no_rests[sort_idx]


# print("gt_times_no_rests dtype:", np.array(gt_times_no_rests).dtype)
# print("gt_freqs_no_rests dtype:", np.array(gt_freqs_no_rests).dtype)

# print("First few times:", gt_times_no_rests[:10])
# print("First few freqs:", gt_freqs_no_rests[:10])

# initial_latency = 0.5 
# filtered_time = filtered_time + initial_latency


# # create function to perform step interpolation
# gt_interp = interp1d(
#     gt_times_no_rests,
#     gt_freqs_no_rests,
#     kind="previous",
#     bounds_error=False,
#     fill_value="extrapolate"
# )



# # resample ground truth to match with CREPE time stamps
# gt_resampled = gt_interp(filtered_time)

# # use cross-correlation to find best time shift
# # remove DC offset (constant bias), subtract mean to "center" around zero
# crepe_centered = filtered_freq - np.mean(filtered_freq)
# gt_centered = gt_resampled - np.mean(gt_resampled)

# # compute cross correlation
# corr = np.correlate(crepe_centered, gt_centered, mode="full")

# # shift back to centered lag
# lags = np.arange(-len(gt_centered) + 1, len(crepe_centered))

# # convert lag to seconds
# time_step = filtered_time[1] - filtered_time[0]
# time_lags = lags * time_step

# # restrict search
# valid_idx = time_lags >= 0
# corr = corr[valid_idx]
# time_lags = time_lags[valid_idx]

# # Save CREPE times BEFORE shifting
# filtered_time_before = filtered_time.copy()

# # apply time shift (update crepe timestamps)
# best_shift = float(time_lags[np.argmax(corr)])
# filtered_time = filtered_time + best_shift

# print(f"Applied time shift: {best_shift:.3f} seconds")

# # Visualize alignment
# plot_alignment(filtered_time_before, filtered_freq, filtered_time, gt_times_no_rests, gt_freqs_no_rests)


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

# Shifts the predicted frequency by octaves until it's closest to the target frequency.
def correct_octave(pred_freq, target_freq):
    # If CREPE outputs 0 (silence), just return 0
    # if pred_freq <= 0:
    #     return pred_freq

    # while pred_freq > target_freq * 4:
    #     pred_freq /= 2
    # while pred_freq < target_freq / 4:
    #     pred_freq *= 2
    return pred_freq


def process_note_frequencies(pred_freqs, target_freq):
    # # apply correction to each CREPE prediction individually
    # corrected = [correct_octave(f, target_freq) for f in pred_freqs if f > 0]

    # if not corrected:
    #     return None
    # # should be (corrected) if want to work
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
    time_slice = adjusted_time[start_idx:end_idx]

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
You are a music teacher analyzing a student’s recording.

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
   - Note, timestamp, measure, sharp/flat
   - Suggest what to practice
3. End with an overall summary of their intonation.
"""


response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])

print(response.choices[0].message.content)

def generate_feedback(measures, notes_list, cents_list, times_list, tolerance=25):
    # Dictionary (key is measure number, value is a list of mistakes for that measure)
    mistakes_by_measure = {}

    # Loop through every note
    for measure, note, cents, time in zip(measures, notes_list, cents_list, times_list):
        # Skip missing predictions
        if np.isnan(cents):
            continue

        # Classify in tune, sharp, flat
        if abs(cents) <= tolerance:
            continue
        elif cents > 0:
            tip_options = [
                f"{note} was too sharp ({cents:.1f} cents) at {time:.2f}s - try lowering your finger slightly to bring the pitch down.",
                f"{note} was sharp ({cents:.1f} cents) at {time:.2f}s - practice slowly with a tuner to center the pitch.",
            ]
        else:
            tip_options = [
                f"{note} was too flat ({abs(cents):.1f} cents) at {time:.2f}s - try placing your finger slightly higher to raise the pitch.",
                f"{note} was flat ({abs(cents):.1f} cents) at {time:.2f}s - practice sliding up to the correct pitch using a tuner.",
            ]

        # Randomly choose one of the two tips
        chosen_tip = random.choice(tip_options)

        # Group mistakes by measure
        if measure not in mistakes_by_measure:
            mistakes_by_measure[measure] = []
        mistakes_by_measure[measure].append(chosen_tip)

     # If there are no mistakes, give positive feedback
    if not mistakes_by_measure:
        return "Great job! All of your notes were in tune. Keep up the good work!"
        
    # Build the final feedback string
    feedback = "Feedback:\n\n"
    for measure, mistakes in mistakes_by_measure.items():
        feedback += f"Measure {measure}:\n"
        for mistake in mistakes:
            feedback += f"  - {mistake}\n"
        feedback += "\n"

    return feedback

feedback = generate_feedback(measure_numbers, note_names, all_differences, all_times)
print("\n" + feedback + "\n\n")

# print reasoning
print("Reasoning - Model output for each note:")
print("=" * 40)
print("\n".join(reasoning_lines))
print("=" * 40)

def plot_pitch_comparison(crepe_times, crepe_freqs, note_times, note_freqs, corrected_times, corrected_freqs):
    plt.figure(figsize=(14, 6))

    # Plot CREPE predictions
    plt.scatter(crepe_times, crepe_freqs, color="blue", s=8, alpha=0.6, label="CREPE Predicted Frequency")

    # Plot target (sheet music) frequencies
    plt.scatter(note_times, note_freqs, color="red", s=50, marker="x", label="Target Note Frequencies")

    # Plot corrected crepe predictons
    plt.scatter(corrected_times, corrected_freqs, color="green", s=50, marker="x", label="Corrected CREPE predictions")


    # Make it pretty
    plt.title("Pitch Tracking: CREPE Predictions vs. Target Notes")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)

    plt.show()

plot_pitch_comparison(adjusted_time, adjusted_freq, all_times, note_freqs, all_times, predicted_freqs)


for n in notes_and_rests:
    if n.isRest:
        print("rest")
    else:
        print(n.measureNumber, n.nameWithOctave, n.offset, n.quarterLength)




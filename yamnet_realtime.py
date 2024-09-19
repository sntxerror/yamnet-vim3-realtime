import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import csv
import threading
from scipy.signal import resample
import soundfile as sf
import tempfile
import os
import sys

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Audio settings
SAMPLE_RATE = 48000  # Capture at 48kHz
RESAMPLED_RATE = 16000  # YAMNet expects 16kHz
CHANNELS = 1  # Your microphone is likely mono; change if it's stereo
SECONDS_PER_CHUNK = 2  # Capture 3 seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * SECONDS_PER_CHUNK)  # Number of samples per chunk

# Load class names
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path) as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        class_names.append(row[2])

def process_audio_file(filename):
    """Run YAMNet inference on the given audio file and display results sorted by probability."""
    
    # Load audio file
    audio_data, sr = sf.read(filename)
    if sr != RESAMPLED_RATE:
        # Resample the audio to 16kHz if necessary
        audio_data = resample(audio_data, int(len(audio_data) * RESAMPLED_RATE / sr))

    # Run inference
    scores, embeddings, mel_spectrogram = model(audio_data)

    # Sort classes by score in descending order, and remove classes with near-zero probability
    threshold = 0.01  # Set a threshold to filter out near-zero probabilities
    sorted_classes = sorted(
        [(class_names[i], score) for i, score in enumerate(scores.numpy().mean(axis=0)) if score > threshold],
        key=lambda x: x[1],
        reverse=True
    )

    # Display all classes sorted by probability
    display_text = "\rDetected classes (sorted by probability):\n"
    for class_name, score in sorted_classes:
        display_text += f"{class_name}: {score:.3f}\n"

    # Display in the same line without scrolling
    sys.stdout.write("\033c")  # Clear the screen
    sys.stdout.write(display_text)
    sys.stdout.flush()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # Create a temporary file for the current chunk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, indata, SAMPLE_RATE)

        # Process the saved audio file
        process_audio_file(tmp_file.name)

        # Clean up the temp file
        os.unlink(tmp_file.name)

def main():
    print(f"Starting real-time audio stream with {SECONDS_PER_CHUNK}-second chunks. Press Ctrl+C to stop.")

    try:
        # Start the stream and process audio chunks
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE):
            while True:
                sd.sleep(1000)  # Keep the stream alive
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()

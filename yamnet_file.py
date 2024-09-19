import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import csv
import queue
import threading
from scipy.signal import resample
import soundfile as sf

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 48000  # 1 second of audio at 48kHz before resampling

# Load class names
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path) as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        class_names.append(row[2])

def generate_test_file(filename='test_audio.wav', duration=10):
    """Generate a 10-second test audio file."""
    print(f"Generating {duration}-second test audio file...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()  # Wait until the recording is finished
    sf.write(filename, recording, SAMPLE_RATE)
    print(f"Test audio file saved as {filename}")
    return filename

def run_inference_on_file(filename):
    """Run YAMNet inference on the given audio file."""
    print(f"Running inference on {filename}...")

    # Load audio file
    audio_data, sr = sf.read(filename)
    if sr != SAMPLE_RATE:
        # Resample the audio to 16kHz if necessary
        audio_data = resample(audio_data, int(len(audio_data) * SAMPLE_RATE / sr))
    
    # Ensure it's exactly 16000 samples (1 second of audio at 16kHz)
    if len(audio_data) != SAMPLE_RATE * 10:  # for 10 seconds file
        print(f"Warning: Expected {SAMPLE_RATE * 10} samples but got {len(audio_data)}")
        return
    
    # Run inference
    scores, embeddings, mel_spectrogram = model(audio_data[:16000])  # Test with 1 second (16k samples)
    
    print(f"Scores shape: {scores.shape}")  # Debugging line to check the shape of scores
    
    # Get predictions with probability > 5%
    high_prob_indices = np.where(scores.numpy().mean(axis=0) > 0.05)[0]
    high_prob_classes = [(class_names[i], scores.numpy().mean(axis=0)[i]) for i in high_prob_indices]
    
    if high_prob_classes:
        print("\nHigh probability classes:")
        for class_name, score in high_prob_classes:
            print(f"{class_name}: {score:.3f}")
    else:
        print("\nNo classes with probability > 5%")

if __name__ == "__main__":
    # Generate 10-second test audio file
    test_filename = generate_test_file()

    # Run inference on the generated audio file
    run_inference_on_file(test_filename)

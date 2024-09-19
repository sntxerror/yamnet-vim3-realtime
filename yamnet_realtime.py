import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import csv
import threading
from scipy.signal import resample
import tempfile
import os
import soundfile as sf
from flask import Flask, render_template
from flask_socketio import SocketIO

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Audio settings
SAMPLE_RATE = 48000  # Capture at 48kHz
RESAMPLED_RATE = 16000  # YAMNet expects 16kHz
CHANNELS = 1
SECONDS_PER_CHUNK = 2
CHUNK_SIZE = int(SAMPLE_RATE * SECONDS_PER_CHUNK)

# Load class names
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path) as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        class_names.append(row[2])

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Global variable to store accumulated data
accumulated_data = {class_name: {'count': 0, 'total_score': 0} for class_name in class_names}

def process_audio(indata):
    global accumulated_data

    # Create a temporary file for the current chunk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, indata, SAMPLE_RATE)

        # Load and resample audio file
        audio_data, sr = sf.read(tmp_file.name)
        if sr != RESAMPLED_RATE:
            audio_data = resample(audio_data, int(len(audio_data) * RESAMPLED_RATE / sr))

        # Run inference
        scores, embeddings, mel_spectrogram = model(audio_data)

        # Update accumulated data
        for i, score in enumerate(scores.numpy().mean(axis=0)):
            class_name = class_names[i]
            accumulated_data[class_name]['count'] += 1
            accumulated_data[class_name]['total_score'] += score

        # Get top 5 predictions
        top_5_indices = np.argsort(scores.numpy().mean(axis=0))[-5:][::-1]
        top_5_classes = [(class_names[i], float(scores.numpy().mean(axis=0)[i])) for i in top_5_indices]

        # Emit predictions to web clients
        socketio.emit('update_predictions', {'predictions': top_5_classes})

        # Clean up the temp file
        os.unlink(tmp_file.name)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    threading.Thread(target=process_audio, args=(indata,)).start()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('request_accumulated_data')
def send_accumulated_data():
    global accumulated_data
    sorted_data = sorted(
        [(class_name, data['total_score'] / data['count'] if data['count'] > 0 else 0)
         for class_name, data in accumulated_data.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Top 10 classes
    socketio.emit('accumulated_data', {'data': sorted_data})

def run_flask():
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)

def main():
    print(f"Starting real-time audio stream with {SECONDS_PER_CHUNK}-second chunks. Press Ctrl+C to stop.")

    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    try:
        # Start the stream and process audio chunks
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE):
            print("Audio stream started. Open a web browser and navigate to http://localhost:8080")
            while True:
                sd.sleep(1000)  # Keep the stream alive
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Shutting down Flask server...")
        socketio.stop()
        flask_thread.join()

if __name__ == "__main__":
    main()
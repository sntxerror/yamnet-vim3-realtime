import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import csv
import threading
from scipy.signal import resample
from flask import Flask, render_template
from flask_socketio import SocketIO

# Load YAMNet model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Audio settings
SAMPLE_RATE = 48000  # Input capture rate
RESAMPLED_RATE = 16000  # YAMNet expects 16kHz
CHANNELS = 1
SECONDS_PER_CHUNK = 2  # Each audio chunk duration
CHUNK_SIZE = int(SAMPLE_RATE * SECONDS_PER_CHUNK)  # Calculate chunk size

# Load class names
class_names = []
class_map_path = model.class_map_path().numpy().decode('utf-8')
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

    # Resample the input audio to YAMNet's expected sample rate (16kHz)
    audio_data = resample(indata, int(len(indata) * RESAMPLED_RATE / SAMPLE_RATE))

    # Convert to tensor and normalize
    audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    audio_data = tf.reshape(audio_data, [-1])  # Flatten the audio tensor

    # Run YAMNet inference
    scores, embeddings, mel_spectrogram = model(audio_data)

    # Update accumulated data
    for i, score in enumerate(scores.numpy().mean(axis=0)):
        class_name = class_names[i]
        accumulated_data[class_name]['count'] += 1
        accumulated_data[class_name]['total_score'] += score

    # Get top 10 predictions and emit to web clients
    top_10_indices = np.argsort(scores.numpy().mean(axis=0))[-10:][::-1]
    top_10_classes = [(class_names[i], float(scores.numpy().mean(axis=0)[i])) for i in top_10_indices]
    socketio.emit('update_predictions', {'predictions': top_10_classes})

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Start a thread to process the audio data using tensors
    threading.Thread(target=process_audio, args=(indata[:, 0],)).start()  # Only pass the first channel

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
    )[:10]  # Send top 10 accumulated classes
    socketio.emit('accumulated_data', {'data': sorted_data})

def run_flask():
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)

def main():
    print(f"Starting real-time audio stream with {SECONDS_PER_CHUNK}-second chunks. Press Ctrl+C to stop.")

    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    try:
        # Start audio stream with the defined callback
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

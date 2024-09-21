import numpy as np
import sounddevice as sd
import tensorflow as tf
import csv
import threading
from scipy.signal import resample
from flask import Flask, render_template
from flask_socketio import SocketIO

# Load YAMNet model from the local directory (preloaded)
model = tf.saved_model.load('models')  # Point this to the directory where the local model is stored

# Audio settings
SAMPLE_RATE = 48000  # Input capture rate
RESAMPLED_RATE = 16000  # YAMNet expects 16kHz
CHANNELS = 1
SECONDS_PER_CHUNK = 2  # Each audio chunk duration
CHUNK_SIZE = int(SAMPLE_RATE * SECONDS_PER_CHUNK)  # Calculate chunk size

# Load class names from local yamnet_class_map.csv file
class_names = []
class_groups_map = {}
with open('models/assets/yamnet_class_map_grouped.csv') as f:  # Load local CSV with class names and groups
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])
        class_groups_map[row['display_name']] = [group.strip() for group in row['groups'].split(',')]

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Global variable to store accumulated data for classes and groups
accumulated_data = {class_name: {'count': 0, 'total_score': 0} for class_name in class_names}
group_accumulated_data = {group: {'count': 0, 'total_score': 0} for group_list in class_groups_map.values() for group in group_list}

def process_audio(indata):
    global accumulated_data, group_accumulated_data

    # Resample the input audio to YAMNet's expected sample rate (16kHz)
    audio_data = resample(indata, int(len(indata) * RESAMPLED_RATE / SAMPLE_RATE))

    # Convert to tensor and normalize
    audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    audio_data = tf.reshape(audio_data, [-1])  # Flatten the audio tensor

    # Run YAMNet inference
    scores, embeddings, mel_spectrogram = model(audio_data)

    # Update accumulated data for both classes and groups
    for i, score in enumerate(scores.numpy().mean(axis=0)):
        class_name = class_names[i]
        accumulated_data[class_name]['count'] += 1
        accumulated_data[class_name]['total_score'] += score
        
        # Update group-level data based on the groups assigned to the class
        groups = class_groups_map.get(class_name, [])
        for group in groups:
            group_accumulated_data[group]['count'] += 1
            group_accumulated_data[group]['total_score'] += score

    # Get top 10 classes and groups
    top_10_class_indices = np.argsort(scores.numpy().mean(axis=0))[-10:][::-1]
    top_10_classes = [(class_names[i], float(scores.numpy().mean(axis=0)[i])) for i in top_10_class_indices]

    # Select only the top 10 groups, ensuring consistency
    top_10_groups = sorted(
        [(group, data['total_score'] / data['count'] if data['count'] > 0 else 0)
        for group, data in group_accumulated_data.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Emit predictions for both classes and groups to web clients
    socketio.emit('update_predictions', {'predictions': top_10_classes, 'groups': top_10_groups})

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
    global accumulated_data, group_accumulated_data

    # Send accumulated data for both classes and groups
    sorted_classes = sorted(
        [(class_name, data['total_score'] / data['count'] if data['count'] > 0 else 0)
         for class_name, data in accumulated_data.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Limit to top 10 classes

    sorted_groups = sorted(
        [(group, data['total_score'] / data['count'] if data['count'] > 0 else 0)
         for group, data in group_accumulated_data.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Limit to top 10 groups

    socketio.emit('accumulated_data', {'classes': sorted_classes, 'groups': sorted_groups})

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

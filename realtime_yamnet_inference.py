import numpy as np
import librosa
import pyaudio
import threading
import time
from ksnn.api import KSNN
from ksnn.types import *
from flask import Flask, render_template
from flask_socketio import SocketIO
import csv

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize KSNN
ksnn = KSNN('yamnet_npu')

# Load class names
class_names = []
with open('yamnet_class_map.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

# Audio settings
CHUNK = 16000  # 1 second of audio at 16kHz
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

# Global variables for sharing data between threads
audio_data = np.zeros(CHUNK, dtype=np.float32)
predictions = []

def audio_callback(in_data, frame_count, time_info, status):
    global audio_data
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    return (in_data, pyaudio.paContinue)

def process_audio():
    global audio_data, predictions
    while True:
        # Prepare input for NPU
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_mels=64, fmax=8000)
        log_mel_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=(0, -1))
        
        # Run inference
        outputs = ksnn.inference(log_mel_spectrogram)
        scores = outputs[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(scores)[-5:][::-1]
        predictions = [(class_names[i], float(scores[i])) for i in top_indices]
        
        # Emit predictions to web clients
        socketio.emit('update_predictions', {'predictions': predictions})
        
        time.sleep(0.1)  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Start audio processing thread
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Start audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)
    
    stream.start_stream()
    
    # Start web server
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
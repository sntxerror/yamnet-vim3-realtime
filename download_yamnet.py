import tensorflow as tf
import tensorflow_hub as hub
import os

# Directory to save the updated model
SAVE_DIR = 'models'

# Load the YAMNet model from TensorFlow Hub
print("Downloading the YAMNet model from TensorFlow Hub...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Create the local directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Save the model locally
tf.saved_model.save(model, SAVE_DIR)

# Extract and save the class map (CSV) if needed
print("Extracting class map CSV...")
class_map_path = model.class_map_path().numpy().decode('utf-8')
csv_save_path = os.path.join(SAVE_DIR, 'assets', 'yamnet_class_map.csv')

# Create assets directory if it doesn't exist
assets_dir = os.path.join(SAVE_DIR, 'assets')
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Download and save the class map CSV
with open(class_map_path, 'r') as src, open(csv_save_path, 'w') as dst:
    dst.write(src.read())

print(f"YAMNet model and class map CSV saved at {SAVE_DIR}.")
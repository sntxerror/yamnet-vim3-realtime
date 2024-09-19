#!/bin/bash

# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-dev libatlas-base-dev libhdf5-dev libhdf5-serial-dev libjpeg-dev libpng-dev libfreetype6-dev libopenmpi-dev libblas-dev gfortran liblapack-dev libportaudio2 ffmpeg libsndfile1 portaudio19-dev

# Set up virtual environment
python3 -m venv yamnet_env
source yamnet_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download YAMNet class map
rm -f yamnet_class_map.csv && wget https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv

echo "Setup complete. You can now run the YAMNet real-time inference script."
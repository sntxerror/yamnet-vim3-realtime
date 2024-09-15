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

# Clone and install KSNN
git clone https://github.com/khadas/ksnn
cd ksnn
sudo python3 setup.py install
cd ..

# Download YAMNet model
wget https://storage.googleapis.com/tensorflow/keras-applications/yamnet/yamnet.h5

# Download YAMNet class map
wget https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv

echo "Setup complete. Please convert the YAMNet model using KSNN conversion tool."
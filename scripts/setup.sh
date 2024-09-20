#!/bin/bash

set -e

echo "Starting YAMNet setup for Khadas VIM3..."

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-dev libatlas-base-dev libhdf5-dev \
    libhdf5-serial-dev libjpeg-dev libpng-dev libfreetype6-dev libopenmpi-dev \
    libblas-dev gfortran liblapack-dev libportaudio2 ffmpeg libsndfile1 \
    build-essential cmake

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv yamnet_env
source yamnet_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installations
echo "Verifying PIP installations..."
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
python3 -c "import numpy; numpy.show_config()"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import sounddevice as sd; print('sounddevice version:', sd.__version__)"

# Download YAMNet model
echo "Downloading YAMNet model..."
mkdir -p models
wget https://storage.googleapis.com/tfhub-modules/google/yamnet/1.tar.gz -O models/yamnet.tar.gz
tar -xzvf models/yamnet.tar.gz -C models
rm models/yamnet.tar.gz

echo "Setup complete. You can now run the YAMNet real-time inference script."
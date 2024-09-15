# Real-time YAMNet Audio Classification on Khadas VIM3 with Web UI

This guide provides comprehensive instructions for setting up and running a real-time audio classification system using the YAMNet model on a Khadas VIM3 board. The system processes audio from a USB microphone, performs inference using the VIM3's NPU, and displays results through a web interface.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Repository Setup](#repository-setup)
3. [VIM3 Setup](#vim3-setup)
4. [YAMNet Model Conversion](#yamnet-model-conversion)
5. [Project Implementation](#project-implementation)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)
8. [Future Improvements](#future-improvements)

## Prerequisites

- Khadas VIM3 board
- Ubuntu 20.04 LTS installed on the VIM3
- USB microphone
- Internet connection on the VIM3
- SSH access to the VIM3 or direct access via keyboard and monitor
- GitHub account (for repository setup)

## Repository Setup

1. Create a new GitHub repository named "yamnet-vim3-realtime".
2. Clone the repository to your VIM3:
   ```bash
   git clone https://github.com/your-username/yamnet-vim3-realtime.git
   cd yamnet-vim3-realtime
   ```
3. Create the following file structure:
   ```
   yamnet-vim3-realtime/
   ├── README.md
   ├── requirements.txt
   ├── realtime_yamnet_inference.py
   ├── templates/
   │   └── index.html
   └── scripts/
       └── setup.sh
   ```

4. Add the following content to `requirements.txt`:
   ```
   numpy==1.19.5
   tensorflow==2.4.1
   tensorflow-hub
   librosa
   matplotlib
   pyaudio
   flask
   flask-socketio
   ```

5. Create `scripts/setup.sh` with the following content:
   ```bash
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

   echo "Setup complete. Please proceed with YAMNet model conversion."
   ```

6. Commit and push these files to your GitHub repository.

## VIM3 Setup

1. On your Khadas VIM3, if you haven't already, clone the repository:
   ```bash
   git clone https://github.com/your-username/yamnet-vim3-realtime.git
   cd yamnet-vim3-realtime
   ```

2. Run the setup script:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source yamnet_env/bin/activate
   ```

## YAMNet Model Conversion

Perform these steps on your Khadas VIM3:

1. Ensure you're in the project directory and the virtual environment is activated:
   ```bash
   cd yamnet-vim3-realtime
   source yamnet_env/bin/activate
   ```

2. Create a sample input file for the KSNN conversion tool:
   ```python
   python3 -c "import numpy as np; sample_input = np.random.rand(1, 96, 64, 1).astype(np.float32); np.save('sample_input.npy', sample_input)"
   ```

3. Convert the YAMNet model:
   ```bash
   python3 -m ksnn.convert \
       --model_path yamnet.h5 \
       --model_type tensorflow \
       --optimize VIPNANOQI_PID0X88 \
       --input_shapes 1,96,64,1 \
       --input_files sample_input.npy \
       --output_dir yamnet_npu
   ```
   Note: Use VIPNANOQI_PID0X99 instead if you're using VIM3L.

4. Verify the conversion was successful:
   ```bash
   ls yamnet_npu
   ```
   You should see files including a `.nb` file, which is the converted model for the NPU.

Important: The conversion process may take some time and requires significant computational resources. Ensure your VIM3 has adequate cooling during this process.

## Running the Application

1. Ensure your USB microphone is connected to the Khadas VIM3.

2. Run the script:
   ```bash
   python realtime_yamnet_inference.py
   ```

3. Access the web UI by opening a web browser and navigating to `http://<VIM3_IP_ADDRESS>:8080`

## Troubleshooting

1. USB Microphone Issues:
   - Check if the microphone is recognized: `arecord -l`
   - If you encounter permission issues, add your user to the audio group:
     ```bash
     sudo usermod -a -G audio $USER
     ```
     Then log out and log back in.

2. NPU Utilization:
   - Monitor CPU and NPU usage: `top`
   - Check VIM3 temperature: `sudo cat /sys/class/thermal/thermal_zone0/temp`

3. Memory Issues:
   - Monitor memory usage: `free -m`
   - If you're running out of memory, consider reducing the audio chunk size or optimizing the inference process.

4. Network Issues:
   - Ensure the VIM3 is connected to the network: `ip addr show`
   - Check if the Flask server is running and listening on all interfaces.

5. Model Conversion Issues:
   - Double-check the input shapes and optimization flags in the KSNN conversion command.
   - Ensure the sample input file matches the expected input shape of the YAMNet model.

## Future Improvements

1. Implement authentication for the web interface.
2. Add HTTPS support for secure connections.
3. Create a configuration file for easily adjusting parameters (e.g., chunk size, update frequency).
4. Implement a feature to switch between different audio sources from the web UI.
5. Add visualizations for audio waveforms and spectrograms.
6. Implement error logging and automated error reporting.
7. Optimize the inference process for better real-time performance.
8. Add a feature to record and save audio samples along with their classifications.
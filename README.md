# YAMNet Real-time Audio Classification on Khadas VIM3

This project implements real-time audio classification using the YAMNet model on a Khadas VIM3 board. It processes audio from a USB microphone, performs inference using the VIM3's CPU, and displays results through a web interface.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Troubleshooting](#troubleshooting)
6. [Future Improvements](#future-improvements)

## Prerequisites

- Khadas VIM3 board
- Ubuntu 20.04 LTS installed on the VIM3
- USB microphone
- Internet connection on the VIM3
- SSH access to the VIM3 or direct access via keyboard and monitor

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/yamnet-vim3-realtime.git
   cd yamnet-vim3-realtime
   ```

2. Run the setup script:
   ```
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Activate the virtual environment:
   ```
   source yamnet_env/bin/activate
   ```

## Usage

1. Run the main script:
   ```
   python yamnet_realtime.py
   ```

2. Open a web browser and navigate to `http://<VIM3_IP_ADDRESS>:8080` to view the real-time classification results.

## Project Structure

```
yamnet-vim3-realtime/
├── README.md
├── requirements.txt
├── yamnet_realtime.py
├── scripts/
│   └── setup.sh
├── templates/
│   └── index.html
└── yamnet_class_map.csv
```

- `yamnet_realtime.py`: Main script for audio capture, inference, and web server.
- `scripts/setup.sh`: Setup script for installing dependencies.
- `templates/index.html`: HTML template for the web interface.
- `yamnet_class_map.csv`: Mapping of class IDs to human-readable labels.

## Troubleshooting

1. USB Microphone Issues:
   - Check if the microphone is recognized: `arecord -l`
   - If you encounter permission issues, add your user to the audio group:
     ```
     sudo usermod -a -G audio $USER
     ```
     Then log out and log back in.

2. Performance Issues:
   - Monitor CPU usage: `top`
   - Check VIM3 temperature: `sudo cat /sys/class/thermal/thermal_zone0/temp`

3. Network Issues:
   - Ensure the VIM3 is connected to the network: `ip addr show`
   - Check if the Flask server is running and listening on all interfaces.

## Future Improvements

1. Implement authentication for the web interface.
2. Add HTTPS support for secure connections.
3. Create a configuration file for easily adjusting parameters (e.g., chunk size, update frequency).
4. Implement a feature to switch between different audio sources from the web UI.
5. Add more detailed visualizations for audio waveforms and spectrograms.
6. Implement error logging and automated error reporting.
7. Optimize the inference process for better real-time performance.
8. Add a feature to record and save audio samples along with their classifications.
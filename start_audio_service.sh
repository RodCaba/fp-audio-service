#!/bin/bash

# Startup script for audio service with ALSA error suppression

# Set environment variables to suppress ALSA warnings
export ALSA_PCM_CARD=3
export ALSA_PCM_DEVICE=0
export SDL_AUDIODRIVER=pulse

# Redirect ALSA error messages to /dev/null
exec 2> >(grep -v "ALSA lib" >&2)

# Activate virtual environment
source .venv/bin/activate

echo "Starting Audio Service with suppressed ALSA warnings..."
echo "Using USB Audio device: card 3, device 0"

# Start the audio service
python app.py

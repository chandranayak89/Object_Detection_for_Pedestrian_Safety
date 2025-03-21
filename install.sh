#!/bin/bash

# Pedestrian Safety Detection System Installer
# This script sets up the required environment for the pedestrian safety detection system

echo "========================================================"
echo "  Installing Pedestrian Safety Detection Environment"
echo "========================================================"

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Error: Python is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
echo "Using Python version: $PY_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Check if CUDA is available for GPU acceleration
if $PYTHON -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available. GPU acceleration will be used."
else
    echo "CUDA is not available. Using CPU mode."
fi

# Create data directory
mkdir -p data

echo "========================================================"
echo "Installation completed!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Example usage:"
echo "python pedestrian_detection_haar.py --webcam"
echo "========================================================" 
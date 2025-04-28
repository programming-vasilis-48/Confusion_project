#!/bin/bash
# Setup script for QTrobot Confusion Detection System

# Exit on error
set -e

# Print header
echo "=========================================================="
echo "QTrobot Confusion Detection System - Setup Script"
echo "=========================================================="
echo ""

# Check if running on Ubuntu 20.04
if [[ "$(lsb_release -rs)" != "20.04" ]]; then
    echo "WARNING: This system is designed for Ubuntu 20.04."
    echo "Current OS: $(lsb_release -ds)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if ROS Noetic is installed
if ! command -v rosversion &> /dev/null || [[ "$(rosversion -d)" != "noetic" ]]; then
    echo "ROS Noetic not found. Installing..."
    
    # Setup sources.list
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    
    # Setup keys
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    
    # Update package index
    sudo apt update
    
    # Install ROS Noetic
    sudo apt install -y ros-noetic-desktop-full
    
    # Initialize rosdep
    sudo apt install -y python3-rosdep
    sudo rosdep init
    rosdep update
    
    # Environment setup
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    
    echo "ROS Noetic installed successfully."
else
    echo "ROS Noetic already installed: $(rosversion -d)"
fi

# Install required ROS packages
echo "Installing required ROS packages..."
sudo apt install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-rqt-image-view \
    ros-noetic-rviz \
    ros-noetic-rosbash

# Install Python dependencies
echo "Installing Python dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    python3-yaml

# Install Python packages
echo "Installing Python packages..."
pip3 install --user \
    torch==1.10.0 \
    torchvision==0.11.1 \
    onnx==1.10.2 \
    onnxruntime==1.9.0 \
    scikit-learn==1.0.1

# Create catkin workspace if it doesn't exist
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Setting up catkin workspace at: $WORKSPACE_DIR"

if [ ! -f "$WORKSPACE_DIR/src/CMakeLists.txt" ]; then
    echo "Initializing catkin workspace..."
    cd "$WORKSPACE_DIR"
    mkdir -p src
    catkin_make
    source devel/setup.bash
    echo "source $WORKSPACE_DIR/devel/setup.bash" >> ~/.bashrc
else
    echo "Catkin workspace already initialized."
fi

# Create model directories
echo "Creating model directories..."
mkdir -p "$WORKSPACE_DIR/models/face_detection"
mkdir -p "$WORKSPACE_DIR/models/confusion_detection"

# Download pre-trained models
echo "Downloading pre-trained models..."
# Note: In a real implementation, this would download actual models
# For now, we'll just create placeholder files

# Face detection model placeholder
cat > "$WORKSPACE_DIR/models/face_detection/README.md" << EOF
# Face Detection Models

This directory contains pre-trained models for face detection and facial feature extraction.

## Models:
- face_detection_model.onnx: Face detection model
- au_detection_model.onnx: Facial action unit detection model

## Usage:
These models are automatically loaded by the face_detector_node and feature_extractor_node.
EOF

# Confusion detection model placeholder
cat > "$WORKSPACE_DIR/models/confusion_detection/README.md" << EOF
# Confusion Detection Models

This directory contains pre-trained models for confusion detection.

## Models:
- confusion_classifier_model.onnx: LSTM-based confusion classifier model

## Usage:
This model is automatically loaded by the confusion_classifier_node.
EOF

# Create log directory
echo "Creating log directory..."
mkdir -p "$WORKSPACE_DIR/logs"

# Build the packages
echo "Building packages..."
cd "$WORKSPACE_DIR"
catkin_make

# Source the workspace
source devel/setup.bash

echo ""
echo "=========================================================="
echo "Setup completed successfully!"
echo "=========================================================="
echo ""
echo "To use the confusion detection system:"
echo "1. Source the workspace: source $WORKSPACE_DIR/devel/setup.bash"
echo "2. Launch the system: roslaunch confusion_system.launch"
echo ""
echo "For more information, see the README.md file."
echo ""

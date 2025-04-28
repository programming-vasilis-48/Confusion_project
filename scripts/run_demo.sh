#!/bin/bash
# Demo script for QTrobot Confusion Detection System

# Exit on error
set -e

# Print header
echo "=========================================================="
echo "QTrobot Confusion Detection System - Demo Script"
echo "=========================================================="
echo ""

# Get workspace directory
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source ROS workspace
source "$WORKSPACE_DIR/devel/setup.bash"

# Check if camera is connected
echo "Checking camera connection..."
if ! ls /dev/video* &>/dev/null; then
    echo "ERROR: No camera detected. Please connect a camera and try again."
    echo "If you're using a RealSense camera, make sure it's properly connected."
    echo ""
    echo "To run with a simulated camera instead, use: --simulate"
    
    if [[ "$1" == "--simulate" ]]; then
        echo "Running with simulated camera..."
        SIMULATE=true
    else
        exit 1
    fi
fi

# Create log directory for this session
SESSION_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$WORKSPACE_DIR/logs/demo_$SESSION_ID"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"

# Start ROS master if not already running
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "Starting ROS master..."
    roscore &
    sleep 2  # Wait for roscore to start
fi

# If simulating, start camera simulator
if [[ "$SIMULATE" == true ]]; then
    echo "Starting camera simulator..."
    rosrun image_publisher image_publisher "$WORKSPACE_DIR/assets/test_images/face_test.jpg" __name:=camera_simulator &
    CAMERA_PID=$!
    
    # Remap topics
    echo "Remapping camera topics..."
    rosrun topic_tools relay /camera_simulator/image /camera/rgb/image_raw &
    RELAY_PID=$!
    
    # Wait for simulator to start
    sleep 2
fi

# Start confusion detection system
echo "Starting confusion detection system..."
roslaunch confusion_system.launch enable_logging:=true &
SYSTEM_PID=$!

# Wait for system to initialize
sleep 5

echo ""
echo "=========================================================="
echo "Confusion detection system is now running!"
echo "=========================================================="
echo ""
echo "Press Ctrl+C to stop the demo."
echo ""

# Function to clean up processes on exit
function cleanup {
    echo ""
    echo "Stopping demo..."
    
    # Kill processes
    if [[ -n "$SYSTEM_PID" ]]; then
        kill -SIGINT $SYSTEM_PID 2>/dev/null || true
    fi
    
    if [[ "$SIMULATE" == true ]]; then
        if [[ -n "$CAMERA_PID" ]]; then
            kill -SIGINT $CAMERA_PID 2>/dev/null || true
        fi
        if [[ -n "$RELAY_PID" ]]; then
            kill -SIGINT $RELAY_PID 2>/dev/null || true
        fi
    fi
    
    # Wait for processes to terminate
    sleep 2
    
    echo "Demo stopped. Logs saved to: $LOG_DIR"
    echo ""
}

# Register cleanup function
trap cleanup EXIT

# Keep script running until user presses Ctrl+C
while true; do
    sleep 1
done

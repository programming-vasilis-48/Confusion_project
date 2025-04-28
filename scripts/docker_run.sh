#!/bin/bash
# Script to build and run the QTrobot Confusion Detection System Docker container

# Exit on error
set -e

# Print header
echo "=========================================================="
echo "QTrobot Confusion Detection System - Docker Runner"
echo "=========================================================="
echo ""

# Get workspace directory
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"

# Parse command line arguments
BUILD=false
DETACHED=false
RVIZ=false
SIMULATE=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --build)
            BUILD=true
            shift
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        --rviz)
            RVIZ=true
            shift
            ;;
        --simulate)
            SIMULATE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build      Build the Docker image before running"
            echo "  --detached   Run the container in detached mode"
            echo "  --rviz       Run RViz for visualization"
            echo "  --simulate   Run with simulated camera"
            echo "  --help       Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Build the Docker image if requested
if [ "$BUILD" = true ]; then
    echo "Building Docker image..."
    docker-compose -f docker/docker-compose.yml build
fi

# Create log directory
mkdir -p "$WORKSPACE_DIR/logs"

# Set up command
COMPOSE_CMD="docker-compose -f docker/docker-compose.yml"

# Run the container
echo "Running Docker container..."

# Add options
if [ "$DETACHED" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD -d"
fi

if [ "$RVIZ" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD up confusion_detection rviz"
else
    COMPOSE_CMD="$COMPOSE_CMD up confusion_detection"
fi

# Add simulation if requested
if [ "$SIMULATE" = true ]; then
    echo "Running with simulated camera..."
    COMPOSE_CMD="$COMPOSE_CMD --env SIMULATE=true"
fi

# Execute the command
echo "Executing: $COMPOSE_CMD"
eval "$COMPOSE_CMD"

echo ""
echo "Container is running. Press Ctrl+C to stop."
echo ""

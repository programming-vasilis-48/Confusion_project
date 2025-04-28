# QTrobot Confusion Detection and Conversational Repair System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time confusion-detection and conversational-repair system designed for the QTrobot RD-V2 i7 humanoid platform.

## 📋 Overview

This system detects confusion in human-robot interactions using computer vision techniques to analyze facial expressions and implements appropriate repair strategies to improve communication effectiveness. It leverages pretrained models for real-time confusion detection and employs a rule-based repair policy engine.

**System Architecture:**

See the [System Architecture Diagram](docs/images/system_architecture.txt) for a visual representation of the system components and data flow.

For a more detailed view, check out the [System Architecture Documentation](docs/system_architecture.md).

## ✨ Key Features

- **Real-time Facial Expression Analysis**: Detects confusion through facial action units and expressions
- **Adaptive Repair Strategy Selection**: Employs various strategies to address detected confusion
- **ROS-based Integration**: Seamlessly integrates with the QTrobot platform
- **Comprehensive Evaluation Framework**: Includes tools for measuring system performance
- **Optimized for On-device Inference**: Designed for <1 GB model size and ≤15 ms/frame latency

## 🤖 Platform Specifications

### QTrobot RD-V2 i7 Platform

- **Vision**: Intel RealSense™ D455 RGB-D camera with Nuitrack skeleton tracking
- **Audio input**: ReSpeaker Mic Array v2.0 (4 mics, VAD/DOA/beamforming at up to 16 kHz)
- **Compute**: 
  - NUC i7: Ubuntu 20.04 LTS + ROS Noetic (Python 3)
  - Headboard Pi 4B: Raspberry Pi OS (Debian Buster), bridged via ROS
- **APIs/SDKs**:
  - ROS interface: Python/C++/JavaScript topics & services
  - JavaScript SDK: qtrobot.js (roslibjs wrapper)
  - Studio: drag-and-drop GUI with high-level behavior blocks

## 🚀 Quick Start

### Prerequisites

- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/confusion_project.git
cd confusion_project

# Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Running the System

```bash
# Native deployment
source devel/setup.bash
roslaunch confusion_system.launch

# Docker deployment
./scripts/docker_run.sh
```

### Running the Demo

```bash
# Run the demo script
./scripts/run_demo.sh

# Run with simulated camera
./scripts/run_demo.sh --simulate
```

## 📊 Evaluation

The system includes a comprehensive evaluation framework for assessing performance:

```bash
# Evaluate with recorded data
python3 scripts/evaluation.py --mode recorded --bag confusion_session.bag --ground-truth annotations.yaml --visualize

# Live evaluation
python3 scripts/evaluation.py --mode live --duration 60 --visualize
```

## 📚 Documentation

- [System Architecture](docs/system_architecture.md): Detailed overview of system components and data flow
- [User Guide](docs/user_guide.md): Instructions for installation, configuration, and usage
- [API Reference](docs/api_reference.md): Documentation of ROS topics, messages, and services
- [Development Guide](docs/development_guide.md): Guidelines for extending and contributing to the system

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- QTrobot team for providing the hardware platform and support
- Contributors to the open-source libraries used in this project

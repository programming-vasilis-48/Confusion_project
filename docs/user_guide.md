# QTrobot Confusion Detection System - User Guide

This user guide provides instructions for installing, configuring, and using the QTrobot Confusion Detection System.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Running the System](#4-running-the-system)
5. [Evaluation](#5-evaluation)
6. [Troubleshooting](#6-troubleshooting)
7. [FAQ](#7-faq)

## 1. Introduction

The QTrobot Confusion Detection System is designed to detect confusion in human-robot interactions and implement appropriate repair strategies to improve communication effectiveness. The system leverages computer vision techniques to analyze facial expressions and detect signs of confusion, and employs a rule-based policy engine to select and execute appropriate repair strategies.

### 1.1 System Requirements

- **Hardware**:
  - QTrobot RD-V2 i7 humanoid platform
  - Intel RealSense D455 RGB-D camera
  - ReSpeaker Mic Array v2.0 (optional for audio)

- **Software**:
  - Ubuntu 20.04 LTS
  - ROS Noetic
  - Python 3.8+
  - Docker (optional for containerized deployment)

## 2. Installation

### 2.1 Native Installation

1. Clone the repository to the QTrobot's NUC computer:
   ```bash
   git clone https://github.com/yourusername/confusion_project.git
   cd confusion_project
   ```

2. Run the setup script to install dependencies:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Build the ROS packages:
   ```bash
   cd src
   catkin_make
   ```

4. Source the workspace:
   ```bash
   source devel/setup.bash
   ```

### 2.2 Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/confusion_project.git
   cd confusion_project
   ```

2. Build the Docker image:
   ```bash
   chmod +x scripts/docker_run.sh
   ./scripts/docker_run.sh --build
   ```

## 3. Configuration

The system is configured using YAML files located in the `config` directory.

### 3.1 Detection Parameters

Edit `config/detection_params.yaml` to configure the confusion detection components:

```yaml
# Face detection parameters
face_detection:
  detection_frequency: 10  # Hz
  face_detection_threshold: 0.7
  model_path: "models/face_detection/face_detection_model.onnx"
  min_face_size: [30, 30]  # Minimum face size in pixels [width, height]
  scale_factor: 1.1  # Scale factor for face detection

# Confusion classification parameters
confusion_classification:
  classification_frequency: 5  # Hz
  confusion_threshold: 0.6  # Threshold for confusion detection
  min_confidence: 0.7  # Minimum confidence for detection
  # ... other parameters ...
```

### 3.2 Repair Policies

Edit `config/repair_policies.yaml` to configure the repair policy engine:

```yaml
# Policy engine parameters
policy_engine:
  min_confusion_score: 0.6
  min_confidence: 0.7
  max_repair_attempts: 3
  repair_cooldown: 5.0  # seconds
  # ... other parameters ...

# Repair strategy parameters
repair_strategies:
  # Clarification strategy
  clarification:
    priority: 0.8
    initial_success_rate: 0.7
    templates:
      - "Let me clarify what I meant. {}"
      - "To be more clear, {}"
      # ... other templates ...
  # ... other strategies ...
```

## 4. Running the System

### 4.1 Native Deployment

1. Source the workspace:
   ```bash
   source devel/setup.bash
   ```

2. Launch the system:
   ```bash
   roslaunch confusion_system.launch
   ```

3. To run with visualization:
   ```bash
   roslaunch confusion_system.launch enable_visualization:=true
   ```

### 4.2 Docker Deployment

1. Run the container:
   ```bash
   ./scripts/docker_run.sh
   ```

2. To run with visualization:
   ```bash
   ./scripts/docker_run.sh --rviz
   ```

3. To run with a simulated camera:
   ```bash
   ./scripts/docker_run.sh --simulate
   ```

### 4.3 Running the Demo

The system includes a demo script that demonstrates the confusion detection and repair capabilities:

```bash
./scripts/run_demo.sh
```

To run the demo with a simulated camera:

```bash
./scripts/run_demo.sh --simulate
```

## 5. Evaluation

### 5.1 Evaluating with Recorded Data

1. Record a session:
   ```bash
   rosbag record -o confusion_session /camera/rgb/image_raw /vision/face_features /vision/processed_features /human/confusion_state /robot/speech/say
   ```

2. Create a ground truth annotation file:
   ```yaml
   - timestamp: 1619712345.67
     is_confused: true
   - timestamp: 1619712350.12
     is_confused: false
   # ... more annotations ...
   ```

3. Run the evaluation script:
   ```bash
   python3 scripts/evaluation.py --mode recorded --bag confusion_session.bag --ground-truth annotations.yaml --visualize
   ```

### 5.2 Live Evaluation

To evaluate the system in real-time:

```bash
python3 scripts/evaluation.py --mode live --duration 60 --visualize
```

## 6. Troubleshooting

### 6.1 Camera Issues

If the camera is not detected:

1. Check if the camera is properly connected:
   ```bash
   ls /dev/video*
   ```

2. Check if the camera is recognized by ROS:
   ```bash
   rosrun usb_cam usb_cam_node
   ```

3. Try running with a simulated camera:
   ```bash
   ./scripts/run_demo.sh --simulate
   ```

### 6.2 ROS Issues

If you encounter ROS-related issues:

1. Check if ROS is properly sourced:
   ```bash
   echo $ROS_DISTRO
   ```

2. Check if ROS master is running:
   ```bash
   rostopic list
   ```

3. Restart ROS master:
   ```bash
   killall -9 rosmaster
   roscore
   ```

### 6.3 Docker Issues

If you encounter Docker-related issues:

1. Check if Docker is running:
   ```bash
   docker ps
   ```

2. Check Docker logs:
   ```bash
   docker logs confusion_detection
   ```

3. Rebuild the Docker image:
   ```bash
   ./scripts/docker_run.sh --build
   ```

## 7. FAQ

### 7.1 How accurate is the confusion detection?

The accuracy of the confusion detection depends on various factors, including lighting conditions, camera quality, and individual differences in facial expressions. In optimal conditions, the system achieves precision and recall rates of approximately 80-85%.

### 7.2 Can the system detect confusion in multiple people simultaneously?

The current implementation focuses on detecting confusion in a single person at a time, typically the person directly interacting with the robot. Future versions may support multi-person confusion detection.

### 7.3 How can I customize the repair strategies?

You can customize the repair strategies by editing the `config/repair_policies.yaml` file. You can add new templates, adjust priorities, and modify other parameters to suit your specific use case.

### 7.4 Can I use a different camera?

Yes, you can use a different camera by modifying the camera topic in the `config/detection_params.yaml` file. However, the system is optimized for the Intel RealSense D455 camera, and using a different camera may require additional configuration.

### 7.5 How can I contribute to the project?

Contributions to the project are welcome! Please see the `CONTRIBUTING.md` file for guidelines on how to contribute.

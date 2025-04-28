# QTrobot Confusion Detection System

A ROS-based system for detecting human confusion during interactions with QTrobot and executing appropriate repair strategies.

## Overview

This system uses computer vision and machine learning techniques to detect confusion in humans based on facial expressions, and then executes appropriate conversational repair strategies to address the confusion.

The system consists of the following components:

1. **Face Detector Node**: Detects faces in the camera feed and extracts basic facial features
2. **Feature Extractor Node**: Processes facial features to extract higher-level features for confusion detection
3. **Confusion Classifier Node**: Classifies whether a human is confused based on processed facial features
4. **Policy Engine Node**: Selects and executes appropriate repair strategies in response to detected confusion

## Installation

### Prerequisites

- ROS Noetic
- Python 3
- OpenCV
- NumPy
- PyYAML

### Installation Steps

1. Clone this repository into your catkin workspace:

```bash
cd ~/catkin_ws/src
git clone https://github.com/yourusername/qtrobot_confusion_detection.git
```

2. Build the packages:

```bash
cd ~/catkin_ws
catkin_make
```

3. Source the workspace:

```bash
source ~/catkin_ws/devel/setup.bash
```

## Usage

### Running the System

To start the confusion detection system:

```bash
roslaunch qtrobot_confusion_detection confusion_detection.launch
```

### Configuration

You can configure the system by passing arguments to the launch file:

```bash
roslaunch qtrobot_confusion_detection confusion_detection.launch camera_topic:=/camera/color/image_raw min_confusion_score:=0.7 min_confidence:=0.8 max_repair_attempts:=4 repair_cooldown:=3.0
```

### Topics

The system publishes and subscribes to the following topics:

- **Input**:
  - `/camera/color/image_raw` (sensor_msgs/Image): Camera feed from QTrobot
  
- **Output**:
  - `/robot/speech/say` (std_msgs/String): Text for the robot to speak
  - `/robot/behavior/trigger` (std_msgs/String): Behavior triggers for the robot

- **Internal**:
  - `/vision/face_features` (confusion_detection/FaceFeatures): Detected facial features
  - `/vision/processed_features` (confusion_detection/FaceFeatures): Processed facial features
  - `/human/confusion_state` (confusion_detection/ConfusionState): Detected confusion state

### Services

- `/repair_policy/get_strategy` (repair_policy/RepairStrategy): Service for getting a repair strategy based on confusion state

## License

MIT License

## Acknowledgments

- QTrobot by LuxAI S.A.
- OpenCV community
- ROS community

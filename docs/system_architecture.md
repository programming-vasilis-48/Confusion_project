# QTrobot Confusion Detection System - System Architecture

This document provides a comprehensive overview of the QTrobot Confusion Detection System architecture, including its components, data flow, and integration with the QTrobot platform.

## 1. System Overview

The QTrobot Confusion Detection System is designed to detect confusion in human-robot interactions and implement appropriate repair strategies to improve communication effectiveness. The system leverages computer vision techniques to analyze facial expressions and detect signs of confusion, and employs a rule-based policy engine to select and execute appropriate repair strategies.

### 1.1 High-Level Architecture

The system follows a modular architecture based on ROS (Robot Operating System), with the following main components:

```
                                 ┌─────────────────┐
                                 │                 │
                                 │  QTrobot        │
                                 │  Hardware       │
                                 │                 │
                                 └───────┬─────────┘
                                         │
                                         ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│                 │            │                 │            │                 │
│  Sensor Input   │───────────▶│  Confusion      │───────────▶│  Repair Policy  │
│  Processing     │            │  Detection      │            │  Engine         │
│                 │            │                 │            │                 │
└─────────────────┘            └─────────────────┘            └─────────────────┘
                                                                       │
                                                                       │
                                                                       ▼
                                                              ┌─────────────────┐
                                                              │                 │
                                                              │  Robot          │
                                                              │  Behavior       │
                                                              │                 │
                                                              └─────────────────┘
```

### 1.2 Key Features

- Real-time facial expression analysis for confusion detection
- Rule-based confusion classification using facial action units
- Adaptive repair strategy selection
- ROS-based integration with QTrobot platform
- Comprehensive evaluation framework
- Containerized deployment using Docker

## 2. Component Architecture

### 2.1 Sensor Input Processing

This component interfaces with the QTrobot's sensors (primarily the Intel RealSense D455 camera) to capture and preprocess visual data for confusion detection.

**Key Nodes:**
- `face_detector_node`: Detects faces in the camera feed and extracts facial regions
- `feature_extractor_node`: Extracts facial features relevant to confusion detection

**Key Topics:**
- `/camera/rgb/image_raw`: Raw RGB image from the camera
- `/vision/face_features`: Extracted facial features

### 2.2 Confusion Detection

This component analyzes the extracted facial features to detect signs of confusion and publishes the confusion state.

**Key Nodes:**
- `confusion_classifier_node`: Classifies confusion based on facial features

**Key Topics:**
- `/vision/processed_features`: Processed facial features
- `/human/confusion_state`: Detected confusion state

### 2.3 Repair Policy Engine

This component selects and executes appropriate repair strategies based on the detected confusion state.

**Key Nodes:**
- `policy_engine_node`: Selects and executes repair strategies

**Key Topics:**
- `/human/confusion_state`: Detected confusion state
- `/robot/speech/say`: Robot speech output
- `/robot/behavior/trigger`: Robot behavior triggers

**Key Services:**
- `/repair_policy/get_strategy`: Service to get a repair strategy

### 2.4 Robot Behavior

This component interfaces with the QTrobot's behavior system to execute repair strategies.

**Key Topics:**
- `/robot/speech/say`: Robot speech output
- `/robot/behavior/trigger`: Robot behavior triggers

## 3. Data Flow

The data flow through the system follows these steps:

1. The `face_detector_node` subscribes to the camera feed (`/camera/rgb/image_raw`) and detects faces in the image.
2. Detected faces are processed to extract facial features, which are published on the `/vision/face_features` topic.
3. The `feature_extractor_node` subscribes to the `/vision/face_features` topic and extracts facial action units and other features relevant to confusion detection.
4. Processed features are published on the `/vision/processed_features` topic.
5. The `confusion_classifier_node` subscribes to the `/vision/processed_features` topic and classifies confusion based on the extracted features.
6. The confusion state is published on the `/human/confusion_state` topic.
7. The `policy_engine_node` subscribes to the `/human/confusion_state` topic and selects an appropriate repair strategy when confusion is detected.
8. The selected repair strategy is executed by publishing messages on the `/robot/speech/say` and `/robot/behavior/trigger` topics.

## 4. Message Definitions

### 4.1 FaceFeatures Message

```
# Header with timestamp and frame_id
Header header

# Face bounding box in image coordinates [x, y, width, height]
float32[] bbox

# Facial landmarks (x,y coordinates of key points)
float32[] landmarks

# Facial action units (AUs) with intensity values
float32[] action_units

# Head pose [pitch, yaw, roll] in radians
float32[] head_pose

# Eye gaze direction [x, y, z] vectors for left and right eyes
float32[] left_eye_gaze
float32[] right_eye_gaze

# Confidence score for the face detection (0.0 to 1.0)
float32 detection_confidence

# Additional metadata or features
string[] metadata
```

### 4.2 ConfusionState Message

```
# Header with timestamp and frame_id
Header header

# Confusion confidence score (0.0 to 1.0)
float32 confusion_score

# Boolean flag indicating if confusion is detected
bool is_confused

# Confidence level in the detection (0.0 to 1.0)
float32 confidence

# Additional metadata about the detection
string[] detected_cues

# Duration of the current confusion state in seconds
float32 duration
```

### 4.3 RepairStrategy Service

```
# Request: Confusion state and context information
confusion_detection/ConfusionState confusion_state
string current_topic
string[] conversation_history
float32 confusion_duration
int32 previous_repair_attempts
string[] previous_strategies_used

---

# Response: Selected repair strategy and parameters
string strategy_name
string[] strategy_parameters
string repair_message
float32 confidence
bool escalate_to_human
```

## 5. Configuration

The system is configured using YAML files located in the `config` directory:

### 5.1 Detection Parameters (`detection_params.yaml`)

This file configures the confusion detection components, including:
- Face detection parameters
- Feature extraction parameters
- Confusion classification parameters
- Camera parameters
- Logging parameters

### 5.2 Repair Policies (`repair_policies.yaml`)

This file configures the repair policy engine, including:
- Policy engine parameters
- Repair strategy parameters
- Robot behavior parameters
- Logging parameters

## 6. Deployment

The system can be deployed in several ways:

### 6.1 Native Deployment

For native deployment on the QTrobot platform:
1. Clone the repository to the QTrobot's NUC computer
2. Run the setup script: `./scripts/setup.sh`
3. Build the ROS packages: `cd src && catkin_make`
4. Launch the system: `roslaunch confusion_system.launch`

### 6.2 Docker Deployment

For containerized deployment:
1. Clone the repository
2. Build the Docker image: `./scripts/docker_run.sh --build`
3. Run the container: `./scripts/docker_run.sh`

## 7. Evaluation

The system includes an evaluation framework for assessing the performance of the confusion detection and repair components:

### 7.1 Metrics

- **Detection Metrics**: Precision, recall, F1 score
- **Repair Metrics**: Success rate, mean reaction time
- **User Study Metrics**: SUS scores, task completion time, subjective satisfaction

### 7.2 Evaluation Tools

- `scripts/evaluation.py`: Script for evaluating the system using recorded data or live testing
- Visualization tools for confusion timeline, confusion matrix, and repair strategy performance

## 8. Future Enhancements

Potential future enhancements to the system include:

- Integration of audio-based confusion detection
- Learning-based repair

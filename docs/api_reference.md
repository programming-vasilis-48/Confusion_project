# QTrobot Confusion Detection System - API Reference

This document provides a reference for the ROS topics, messages, and services used in the QTrobot Confusion Detection System.

## Table of Contents

1. [ROS Topics](#ros-topics)
2. [ROS Messages](#ros-messages)
3. [ROS Services](#ros-services)
4. [Parameters](#parameters)

## ROS Topics

### Input Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/rgb/image_raw` | `sensor_msgs/Image` | Raw RGB image from the camera |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Raw depth image from the camera (optional) |
| `/audio/raw` | `audio_common_msgs/AudioData` | Raw audio data from the microphone (optional) |

### Internal Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/face_features` | `confusion_detection/FaceFeatures` | Extracted facial features |
| `/vision/processed_features` | `confusion_detection/FaceFeatures` | Processed facial features with action units |
| `/human/confusion_state` | `confusion_detection/ConfusionState` | Detected confusion state |

### Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/robot/speech/say` | `std_msgs/String` | Text to be spoken by the robot |
| `/robot/behavior/trigger` | `std_msgs/String` | Behavior to be triggered on the robot |

## ROS Messages

### FaceFeatures

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

### ConfusionState

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

## ROS Services

### RepairStrategy

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

## Parameters

### Face Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face_detection/detection_frequency` | `int` | `10` | Frequency of face detection in Hz |
| `face_detection/face_detection_threshold` | `float` | `0.7` | Threshold for face detection confidence |
| `face_detection/model_path` | `string` | `models/face_detection/face_detection_model.onnx` | Path to face detection model |
| `face_detection/min_face_size` | `int[]` | `[30, 30]` | Minimum face size in pixels [width, height] |
| `face_detection/scale_factor` | `float` | `1.1` | Scale factor for face detection |

### Feature Extraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_extraction/processing_frequency` | `int` | `10` | Frequency of feature extraction in Hz |
| `feature_extraction/model_path` | `string` | `models/face_detection/au_detection_model.onnx` | Path to action unit detection model |
| `feature_extraction/feature_types` | `string[]` | `["action_units", "head_pose", "eye_gaze"]` | Types of features to extract |

### Confusion Classification Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confusion_classification/classification_frequency` | `int` | `5` | Frequency of confusion classification in Hz |
| `confusion_classification/confusion_threshold` | `float` | `0.6` | Threshold for confusion detection |
| `confusion_classification/min_confidence` | `float` | `0.7` | Minimum confidence for detection |
| `confusion_classification/model_path` | `string` | `models/confusion_detection/confusion_classifier_model.onnx` | Path to confusion classifier model |
| `confusion_classification/temporal_window` | `float` | `3.0` | Temporal window for confusion detection in seconds |
| `confusion_classification/feature_weights` | `dict` | See config file | Weights for different features in rule-based classification |

### Policy Engine Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_engine/min_confusion_score` | `float` | `0.6` | Minimum confusion score for repair |
| `policy_engine/min_confidence` | `float` | `0.7` | Minimum confidence for repair |
| `policy_engine/max_repair_attempts` | `int` | `3` | Maximum number of repair attempts |
| `policy_engine/repair_cooldown` | `float` | `5.0` | Cooldown period between repairs in seconds |
| `policy_engine/escalation_threshold` | `float` | `30.0` | Threshold for escalation to human in seconds |

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Functions for QTrobot Confusion Detection System

This module provides common utility functions used across the confusion
detection and repair policy packages.
"""

import rospy
import numpy as np
import cv2
import os
import yaml
import time
from datetime import datetime

def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            rospy.loginfo(f"Loaded configuration from {config_path}")
            return config
        else:
            rospy.logwarn(f"Configuration file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        rospy.logerr(f"Failed to load configuration: {e}")
        return {}

def save_config(config, config_path):
    """Save configuration to a YAML file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        rospy.loginfo(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        rospy.logerr(f"Failed to save configuration: {e}")
        return False

def get_timestamp():
    """Get a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        rospy.logerr(f"Failed to create directory {directory_path}: {e}")
        return False

def save_image(image, file_path):
    """Save an OpenCV image to a file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the image
        cv2.imwrite(file_path, image)
        
        rospy.loginfo(f"Saved image to {file_path}")
        return True
    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")
        return False

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on an image."""
    try:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        return image
    except Exception as e:
        rospy.logerr(f"Failed to draw bounding box: {e}")
        return image

def draw_landmarks(image, landmarks, color=(0, 0, 255), radius=2):
    """Draw facial landmarks on an image."""
    try:
        for i in range(0, len(landmarks), 2):
            x, y = int(landmarks[i]), int(landmarks[i + 1])
            cv2.circle(image, (x, y), radius, color, -1)
        return image
    except Exception as e:
        rospy.logerr(f"Failed to draw landmarks: {e}")
        return image

def draw_head_pose(image, head_pose, face_center, scale=50):
    """Draw head pose axes on an image."""
    try:
        pitch, yaw, roll = head_pose
        
        # Convert to rotation matrix
        # This is a simplified version, a real implementation would use proper 3D rotation
        
        # Draw axes
        center = (int(face_center[0]), int(face_center[1]))
        
        # X-axis (pitch)
        x_end = (
            int(center[0] + scale * np.cos(yaw) * np.cos(roll)),
            int(center[1] + scale * np.sin(roll))
        )
        cv2.line(image, center, x_end, (0, 0, 255), 2)
        
        # Y-axis (yaw)
        y_end = (
            int(center[0] + scale * np.sin(yaw)),
            int(center[1] + scale * np.cos(pitch) * np.cos(roll))
        )
        cv2.line(image, center, y_end, (0, 255, 0), 2)
        
        # Z-axis (roll)
        z_end = (
            int(center[0] + scale * np.sin(roll)),
            int(center[1] + scale * np.cos(roll))
        )
        cv2.line(image, center, z_end, (255, 0, 0), 2)
        
        return image
    except Exception as e:
        rospy.logerr(f"Failed to draw head pose: {e}")
        return image

def draw_gaze(image, gaze, eye_center, scale=50):
    """Draw eye gaze vector on an image."""
    try:
        gaze_x, gaze_y, gaze_z = gaze
        
        # Normalize gaze vector
        norm = np.sqrt(gaze_x**2 + gaze_y**2 + gaze_z**2)
        gaze_x, gaze_y, gaze_z = gaze_x / norm, gaze_y / norm, gaze_z / norm
        
        # Project 3D gaze vector onto 2D image plane
        center = (int(eye_center[0]), int(eye_center[1]))
        end = (
            int(center[0] + scale * gaze_x),
            int(center[1] + scale * gaze_y)
        )
        
        # Draw gaze vector
        cv2.line(image, center, end, (255, 0, 255), 2)
        
        return image
    except Exception as e:
        rospy.logerr(f"Failed to draw gaze: {e}")
        return image

def calculate_confusion_metrics(predictions, ground_truth):
    """Calculate confusion detection metrics (precision, recall, F1)."""
    try:
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate true positives, false positives, false negatives
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        rospy.logerr(f"Failed to calculate confusion metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

def calculate_repair_success_rate(repair_attempts, successful_repairs):
    """Calculate the success rate of repair strategies."""
    try:
        if repair_attempts == 0:
            return 0.0
        return successful_repairs / repair_attempts
    except Exception as e:
        rospy.logerr(f"Failed to calculate repair success rate: {e}")
        return 0.0

def calculate_reaction_time(confusion_timestamps, repair_timestamps):
    """Calculate the mean reaction time from confusion detection to repair trigger."""
    try:
        if len(confusion_timestamps) != len(repair_timestamps):
            rospy.logwarn("Mismatch in number of confusion and repair timestamps")
            return 0.0
        
        if len(confusion_timestamps) == 0:
            return 0.0
        
        reaction_times = [repair - confusion for confusion, repair in zip(confusion_timestamps, repair_timestamps)]
        return np.mean(reaction_times)
    except Exception as e:
        rospy.logerr(f"Failed to calculate reaction time: {e}")
        return 0.0

def log_event(event_type, event_data, log_file):
    """Log an event to a file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create log entry
        timestamp = get_timestamp()
        log_entry = {
            'timestamp': timestamp,
            'type': event_type,
            'data': event_data
        }
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(yaml.dump([log_entry], default_flow_style=False))
            f.write('---\n')
        
        return True
    except Exception as e:
        rospy.logerr(f"Failed to log event: {e}")
        return False

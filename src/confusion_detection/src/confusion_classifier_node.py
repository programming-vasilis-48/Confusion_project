#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Confusion Classifier Node for QTrobot Confusion Detection System

This node analyzes facial features to detect signs of confusion and
publishes the confusion state.
"""

import rospy
import numpy as np
import os
import time
from confusion_detection.msg import FaceFeatures, ConfusionState

class ConfusionClassifierNode:
    """ROS node for classifying confusion based on facial features."""
    
    def __init__(self):
        """Initialize the confusion classifier node."""
        rospy.init_node('confusion_classifier_node', anonymous=True)
        
        # Parameters
        self.classification_frequency = rospy.get_param('~classification_frequency', 5)  # Hz
        self.confusion_threshold = rospy.get_param('~confusion_threshold', 0.6)
        self.model_path = rospy.get_param('~model_path', 'models/confusion_detection/confusion_classifier_model.onnx')
        self.temporal_window = rospy.get_param('~temporal_window', 3.0)  # seconds
        
        # Load confusion classification model
        self.load_confusion_model()
        
        # State variables
        self.confusion_history = []
        self.confusion_start_time = None
        self.current_confusion_duration = 0.0
        
        # Publishers and subscribers
        self.processed_features_sub = rospy.Subscriber('/vision/processed_features', FaceFeatures, self.features_callback)
        self.confusion_state_pub = rospy.Publisher('/human/confusion_state', ConfusionState, queue_size=10)
        
        rospy.loginfo("Confusion classifier node initialized")
    
    def load_confusion_model(self):
        """Load the confusion classification model."""
        try:
            # This is a placeholder for the actual model loading code
            # In a real implementation, this would load a model for classifying
            # confusion based on facial features, such as a pretrained LSTM model
            rospy.loginfo(f"Loading confusion classification model from {self.model_path}")
            
            # Placeholder for model loading
            # self.confusion_classifier = onnx.load(self.model_path)
            # self.classifier_session = onnxruntime.InferenceSession(self.model_path)
            
            rospy.loginfo("Confusion classification model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load confusion classification model: {e}")
    
    def features_callback(self, data):
        """Process incoming processed facial features."""
        try:
            # Extract relevant features for confusion detection
            features = self.extract_confusion_features(data)
            
            # Classify confusion
            confusion_score, is_confused, confidence, cues = self.classify_confusion(features)
            
            # Update confusion history and duration
            self.update_confusion_state(is_confused)
            
            # Create and publish confusion state message
            self.publish_confusion_state(confusion_score, is_confused, confidence, cues, data.header)
            
        except Exception as e:
            rospy.logerr(f"Error classifying confusion: {e}")
    
    def extract_confusion_features(self, face_features):
        """Extract features relevant to confusion detection from facial features."""
        # In a real implementation, this would extract and normalize
        # the features needed for the confusion classification model
        
        # For now, we'll focus on action units that are relevant to confusion
        # Based on research, these AUs are often associated with confusion:
        # - AU04 (Brow Lowerer)
        # - AU07 (Lid Tightener)
        # - AU24 (Lip Pressor)
        # - AU14 (Dimpler)
        
        # Extract relevant AUs if they exist
        confusion_features = {}
        if len(face_features.action_units) >= 18:
            confusion_features['AU01'] = face_features.action_units[0]  # Inner Brow Raiser
            confusion_features['AU02'] = face_features.action_units[1]  # Outer Brow Raiser
            confusion_features['AU04'] = face_features.action_units[2]  # Brow Lowerer
            confusion_features['AU07'] = face_features.action_units[5]  # Lid Tightener
            confusion_features['AU14'] = face_features.action_units[9]  # Dimpler
            confusion_features['AU24'] = face_features.action_units[14]  # Lip Pressor
        
        # Add head pose if available
        if len(face_features.head_pose) == 3:
            confusion_features['head_pitch'] = face_features.head_pose[0]
            confusion_features['head_yaw'] = face_features.head_pose[1]
            confusion_features['head_roll'] = face_features.head_pose[2]
        
        # Add eye gaze if available
        if len(face_features.left_eye_gaze) == 3 and len(face_features.right_eye_gaze) == 3:
            # Average gaze direction
            confusion_features['gaze_x'] = (face_features.left_eye_gaze[0] + face_features.right_eye_gaze[0]) / 2
            confusion_features['gaze_y'] = (face_features.left_eye_gaze[1] + face_features.right_eye_gaze[1]) / 2
            confusion_features['gaze_z'] = (face_features.left_eye_gaze[2] + face_features.right_eye_gaze[2]) / 2
        
        return confusion_features
    
    def classify_confusion(self, features):
        """Classify confusion based on extracted features."""
        # In a real implementation, this would use the loaded model to classify
        # confusion based on the extracted features
        
        # For now, we'll use a simple rule-based approach
        # In a real implementation, this would be replaced with model inference
        
        # Calculate confusion score based on relevant action units
        confusion_score = 0.0
        detected_cues = []
        
        # Check for furrowed brow (AU04)
        if 'AU04' in features and features['AU04'] > 0.5:
            confusion_score += features['AU04'] * 0.4  # Weight: 40%
            detected_cues.append("furrowed_brow")
        
        # Check for tightened eyelids (AU07)
        if 'AU07' in features and features['AU07'] > 0.3:
            confusion_score += features['AU07'] * 0.2  # Weight: 20%
            detected_cues.append("tightened_eyelids")
        
        # Check for pressed lips (AU24)
        if 'AU24' in features and features['AU24'] > 0.4:
            confusion_score += features['AU24'] * 0.2  # Weight: 20%
            detected_cues.append("pressed_lips")
        
        # Check for dimpler (AU14)
        if 'AU14' in features and features['AU14'] > 0.3:
            confusion_score += features['AU14'] * 0.1  # Weight: 10%
            detected_cues.append("dimpler")
        
        # Check for raised inner brows (AU01) - can indicate questioning
        if 'AU01' in features and features['AU01'] > 0.4:
            confusion_score += features['AU01'] * 0.1  # Weight: 10%
            detected_cues.append("raised_inner_brows")
        
        # Normalize confusion score to [0, 1]
        confusion_score = min(max(confusion_score, 0.0), 1.0)
        
        # Determine if confused based on threshold
        is_confused = confusion_score >= self.confusion_threshold
        
        # Set confidence level (in a real implementation, this would come from the model)
        confidence = 0.8 if confusion_score > 0.8 or confusion_score < 0.3 else 0.6
        
        return confusion_score, is_confused, confidence, detected_cues
    
    def update_confusion_state(self, is_confused):
        """Update the confusion state history and duration."""
        current_time = rospy.get_time()
        
        # Update confusion history
        self.confusion_history.append((current_time, is_confused))
        
        # Remove entries older than the temporal window
        while self.confusion_history and current_time - self.confusion_history[0][0] > self.temporal_window:
            self.confusion_history.pop(0)
        
        # Update confusion duration
        if is_confused:
            if self.confusion_start_time is None:
                self.confusion_start_time = current_time
            self.current_confusion_duration = current_time - self.confusion_start_time
        else:
            self.confusion_start_time = None
            self.current_confusion_duration = 0.0
    
    def publish_confusion_state(self, confusion_score, is_confused, confidence, cues, header):
        """Publish the confusion state."""
        # Create confusion state message
        confusion_state = ConfusionState()
        
        # Set header
        confusion_state.header = header
        
        # Set confusion state
        confusion_state.confusion_score = confusion_score
        confusion_state.is_confused = is_confused
        confusion_state.confidence = confidence
        confusion_state.detected_cues = cues
        confusion_state.duration = self.current_confusion_duration
        
        # Publish the message
        self.confusion_state_pub.publish(confusion_state)
    
    def run(self):
        """Run the confusion classifier node."""
        rate = rospy.Rate(self.classification_frequency)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        classifier = ConfusionClassifierNode()
        classifier.run()
    except rospy.ROSInterruptException:
        pass

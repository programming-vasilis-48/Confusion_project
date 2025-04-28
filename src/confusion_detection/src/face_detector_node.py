#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face Detector Node for QTrobot Confusion Detection System

This node interfaces with the Intel RealSense D455 camera to detect faces
and extract facial features for confusion detection.
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from confusion_detection.msg import FaceFeatures

class FaceDetectorNode:
    """ROS node for face detection and feature extraction."""
    
    def __init__(self):
        """Initialize the face detector node."""
        rospy.init_node('face_detector_node', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.detection_frequency = rospy.get_param('~detection_frequency', 10)  # Hz
        self.face_detection_threshold = rospy.get_param('~face_detection_threshold', 0.7)
        self.model_path = rospy.get_param('~model_path', 'models/face_detection/face_detection_model.onnx')
        
        # Load face detection model
        self.load_face_detection_model()
        
        # Publishers and subscribers
        self.face_features_pub = rospy.Publisher('/vision/face_features', FaceFeatures, queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Face detector node initialized")
    
    def load_face_detection_model(self):
        """Load the face detection model."""
        try:
            # This is a placeholder for the actual model loading code
            # In a real implementation, this would load a face detection model
            # such as RetinaFace, MTCNN, or a custom ONNX model
            rospy.loginfo(f"Loading face detection model from {self.model_path}")
            
            # Placeholder for model loading
            # self.face_detector = cv2.dnn.readNetFromONNX(self.model_path)
            
            # For now, we'll use OpenCV's built-in face detector
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            rospy.loginfo("Face detection model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load face detection model: {e}")
    
    def image_callback(self, data):
        """Process incoming image data."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Detect faces
            faces = self.detect_faces(cv_image)
            
            # Process each detected face
            for face in faces:
                # Extract facial features
                features = self.extract_facial_features(cv_image, face)
                
                # Publish face features
                self.publish_face_features(features, data.header)
                
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def detect_faces(self, image):
        """Detect faces in the input image."""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_facial_features(self, image, face):
        """Extract facial features from the detected face."""
        x, y, w, h = face
        
        # Create a FaceFeatures message
        features = FaceFeatures()
        
        # Set bounding box
        features.bbox = [float(x), float(y), float(w), float(h)]
        
        # Set detection confidence (placeholder)
        features.detection_confidence = 0.9
        
        # In a real implementation, we would extract:
        # - Facial landmarks
        # - Action units
        # - Head pose
        # - Eye gaze
        
        # Placeholder for facial landmarks (would be extracted using a dedicated model)
        features.landmarks = []
        
        # Placeholder for action units (would be extracted using a dedicated model)
        features.action_units = []
        
        # Placeholder for head pose (would be extracted using a dedicated model)
        features.head_pose = [0.0, 0.0, 0.0]  # pitch, yaw, roll
        
        # Placeholder for eye gaze (would be extracted using a dedicated model)
        features.left_eye_gaze = [0.0, 0.0, 1.0]  # x, y, z
        features.right_eye_gaze = [0.0, 0.0, 1.0]  # x, y, z
        
        return features
    
    def publish_face_features(self, features, header):
        """Publish the extracted face features."""
        # Set the header
        features.header = header
        
        # Publish the message
        self.face_features_pub.publish(features)
    
    def run(self):
        """Run the face detector node."""
        rate = rospy.Rate(self.detection_frequency)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        detector = FaceDetectorNode()
        detector.run()
    except rospy.ROSInterruptException:
        pass

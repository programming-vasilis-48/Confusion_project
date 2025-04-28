# Message definition for facial features used in confusion detection

# Header with timestamp and frame_id
Header header

# Face bounding box in image coordinates [x, y, width, height]
float32[] bbox

# Facial landmarks (x,y coordinates of key points)
float32[] landmarks

# Facial action units (AUs) with intensity values
# Format: [AU01_intensity, AU02_intensity, ..., AU45_intensity]
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

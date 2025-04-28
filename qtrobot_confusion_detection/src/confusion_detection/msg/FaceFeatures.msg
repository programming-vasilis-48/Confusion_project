# Message for representing facial features extracted from a detected face

# Header with timestamp and frame_id
Header header

# Bounding box of the face [x, y, width, height]
float32[] bbox

# Confidence of the face detection
float32 detection_confidence

# Facial landmarks (x, y coordinates)
float32[] landmarks

# Facial action units (intensities)
float32[] action_units

# Head pose (pitch, yaw, roll)
float32[] head_pose

# Eye gaze direction (left eye)
float32[] left_eye_gaze

# Eye gaze direction (right eye)
float32[] right_eye_gaze

# Message for representing the confusion state of a human

# Header with timestamp and frame_id
Header header

# Whether the human is confused
bool is_confused

# Confidence in the confusion detection (0.0 to 1.0)
float32 confidence

# Confusion score (0.0 to 1.0)
float32 confusion_score

# Duration of the confusion state in seconds
float32 duration

# Facial action units associated with confusion
float32[] action_units

# Head pose (pitch, yaw, roll)
float32[] head_pose

# Eye gaze direction
float32[] eye_gaze

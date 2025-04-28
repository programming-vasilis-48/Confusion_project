# Message definition for confusion state

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

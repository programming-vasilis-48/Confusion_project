cmake_minimum_required(VERSION 3.0.2)
project(confusion_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  message_generation
)

add_message_files(
  FILES
  ConfusionState.msg
  FaceFeatures.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge image_transport message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/face_detector_node.py
  src/feature_extractor_node.py
  src/confusion_classifier_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

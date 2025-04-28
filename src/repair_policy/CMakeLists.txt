cmake_minimum_required(VERSION 3.0.2)
project(repair_policy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  confusion_detection
  message_generation
)

add_service_files(
  FILES
  RepairStrategy.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  confusion_detection
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs confusion_detection message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/policy_engine_node.py
  src/repair_strategies.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

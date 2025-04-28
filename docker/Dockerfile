# Dockerfile for QTrobot Confusion Detection System
FROM ros:noetic-robot

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV PYTHONIOENCODING=utf-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    python3-yaml \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-rqt-image-view \
    ros-${ROS_DISTRO}-rviz \
    ros-${ROS_DISTRO}-rosbash \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==1.10.0 \
    torchvision==0.11.1 \
    onnx==1.10.2 \
    onnxruntime==1.9.0 \
    scikit-learn==1.0.1

# Create catkin workspace
WORKDIR /catkin_ws
RUN mkdir -p /catkin_ws/src

# Copy the project files
COPY . /catkin_ws/src/confusion_project/

# Build the ROS packages
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /catkin_ws && \
    catkin_make && \
    source devel/setup.bash"

# Create model directories and log directory
RUN mkdir -p /catkin_ws/src/confusion_project/models/face_detection && \
    mkdir -p /catkin_ws/src/confusion_project/models/confusion_detection && \
    mkdir -p /catkin_ws/src/confusion_project/logs

# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["roslaunch", "confusion_system.launch"]

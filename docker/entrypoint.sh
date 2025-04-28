#!/bin/bash
# Entrypoint script for QTrobot Confusion Detection System Docker container

# Source ROS setup
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# Set ROS master URI if provided
if [ ! -z "$ROS_MASTER_URI" ]; then
    export ROS_MASTER_URI=$ROS_MASTER_URI
fi

# Set ROS IP if provided
if [ ! -z "$ROS_IP" ]; then
    export ROS_IP=$ROS_IP
elif [ ! -z "$ROS_HOSTNAME" ]; then
    export ROS_HOSTNAME=$ROS_HOSTNAME
fi

# Print environment information
echo "=========================================================="
echo "QTrobot Confusion Detection System - Docker Container"
echo "=========================================================="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
if [ ! -z "$ROS_IP" ]; then
    echo "ROS_IP: $ROS_IP"
elif [ ! -z "$ROS_HOSTNAME" ]; then
    echo "ROS_HOSTNAME: $ROS_HOSTNAME"
fi
echo "=========================================================="

# Execute the command passed to the container
exec "$@"

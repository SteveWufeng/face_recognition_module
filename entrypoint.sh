#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"

echo "[face_recognition] Starting ROS2 node"
exec python3 -m face_recognition.ros_node

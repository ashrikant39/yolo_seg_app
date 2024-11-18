# Base Image: from https://hub.docker.com/r/dustynv/ros/tags
FROM dustynv/ros:humble-ros-base-l4t-r35.2.1

# Set working directory
WORKDIR /yoloSegApp

# Copy your project files into the container
COPY . /yoloSegApp


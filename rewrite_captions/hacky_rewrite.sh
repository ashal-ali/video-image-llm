#!/bin/bash

# Function to run the Python script with a timeout of 1 hour (3600 seconds)
run_python_with_timeout() {
  timeout 2m python rewrite_captions.py
}

# Function to gracefully exit the script
cleanup() {
  echo "Stopping script..."
  exit 0
}

# Trap the SIGINT signal (Ctrl + C) to call the cleanup function
trap cleanup SIGINT

# Continuously run the Python script
while true; do
  echo "Starting Python script..."
  run_python_with_timeout

  # Sleep for a few seconds before running the script again
  # This helps to avoid starting the script immediately after it was killed
  sleep 5
done

#!/usr/bin/python3

# Configuration for Pi zero 2 W with IMX708 camera module. 
# This script captures a high-resolution image with auto focus and saves it as a JPEG file and a NumPy array.


from picamera2 import Picamera2
from libcamera import controls
import numpy as np
import time


# Initialize the camera
picam2 = Picamera2()

# Set the JPEG quality (0 - 95). Higher = larger file, better quality.
# Defaults to a standard compression level if not set.
picam2.options["quality"] = 75  

# Generate a high-resolution still capture configuration (Center Cut)
# Example: Setting a custom resolution of 1024x768 (or use your sensor's max resolution)
capture_config = picam2.create_still_configuration(main={"size": (4608, 2592)})

# Apply the configuration block to the camera
picam2.configure(capture_config)

# Start camera and wait briefly for exposure and white balance to stabilize
picam2.start()
time.sleep(5)

# Set auto focus to auto mode and trigger auto focus
picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})  # Setauto focus mode to auto
picam2.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})  # Trigger auto focus

# Wait for a moment to allow auto focus to complete
time.sleep(5)

# Define custom prefix and generate timestamp
custom_name = "/home/agcam/images/ID1234"
timestamp = time.strftime("%Y%m%dT%H%M%SZ") # Format: YYYYMMDD'T'HHMMSS'Z'

# Combine to create the filename and array name
file_name = f"{custom_name}_{timestamp}.jpg"  # Save as a JPEG filename
array_name = f"{custom_name}_{timestamp}.npy"  # Save as a NumPy array file


# Capture and save the image at the designated resolution
picam2.capture_file(file_name)   # Capture and save the image as a JPEG file

time.sleep(1)  # Optional: brief pause before capturing the NumPy array

image_array = picam2.capture_array()
np.save(array_name, image_array)  # Save the captured image as a NumPy array


picam2.stop()

#Releases the camera bus
picam2.close() 

from datetime import datetime
import time

import sys
import cv2
import numpy as np

import importlib

sys.path.append("/mnt/db/agcam")

processor = importlib.import_module("processing.processor")

def main():
   while(True):
       print("Agcam Processing Activated")
       processor.process_files()
       print("Standby")
       # wait 1 day to process new images again.
       time.sleep(86400)

main()
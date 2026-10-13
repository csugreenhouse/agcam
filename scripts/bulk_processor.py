from datetime import datetime
import time

import sys
import cv2
import numpy as np

import importlib

sys.path.append("/mnt/db/agcam")

processor = importlib.import_module("processing.processor")

def blk_process():
    print("Bulk Processing Activated")
    processor.process_files("/mnt/image/image-inbox", 5)
    print("Bulk Processing Complete")

blk_process()
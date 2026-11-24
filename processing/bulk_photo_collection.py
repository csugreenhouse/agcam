import importlib
import time
import sys

sys.path.append("/mnt/db/agcam")

cameraControl = importlib.import_module("utils.cameraControl_util")

camera_num = 3
cam_num2 = 4
time_delay_s = 45
number_photos = 100

for i in range(number_photos):
    cameraControl.run_pic_pipeline(camera_num)
    time.sleep(time_delay_s)
    cameraControl.run_pic_pipeline(cam_num2)
    time.sleep(time_delay_s)
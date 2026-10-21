import os
import cv2
import numpy as np
import sys

from pathlib import Path
import time
import importlib

import sys

sys.path.append("/mnt/db/agcam")

gu = importlib.import_module("utils.graph_util")
scanner_util = importlib.import_module("utils.reference_tag_util")
db = importlib.import_module("utils.database_util")

def process_and_make_copies_height_visuals(file_path,cam_name,bulk_delay_s, conn):
    print (f"Processing height visuals for file: {file_path} and camera: {cam_name}")
    #section 1: Prepare directories for processed and archived images
    processed_folder_path = f"/mnt/image/images-processed/{cam_name}/"
    Path(processed_folder_path).mkdir(parents=True, exist_ok=True)

    archived_folder_path = f"/mnt/image/image-archive/{cam_name}/"
    Path(archived_folder_path).mkdir(parents=True, exist_ok=True)

    image_name = file_path.rsplit("/",1)[-1]
    out_path = processed_folder_path + image_name

    time.sleep(bulk_delay_s)

    #step 2: process & check for successful processing

    response = gu.plot_height(file_path, out_path)

    time.sleep(bulk_delay_s*2) #extra delay for graphing

    cam_number = int(cam_name[0])

    # step 3: log the estimated height to the database if available
    #GET int timestamp from file path, of the form .../ID1234_20260817T160013Z.jpg
    section = file_path.rsplit("_",1)[-1].split("Z")[0] #outputs 20260817T160013
    timestamp = f"{section[0:4]}-{section[4:6]}-{section[6:8]} {section[9:11]}:{section[11:13]}:{section[13:]}"
    
    #return from response is : return (reference_tag_number, float(estimated_height), out_path)
    db.log_height(conn, cam_number, timestamp, response[0], response[1], response[2])

    time.sleep(bulk_delay_s)
   
    archived_file_path = archived_folder_path + image_name
    os.rename(file_path, archived_file_path)

    print("\033[32m" + f"Processed and saved height visuals to {out_path}" + "\033[0m")

    time.sleep(bulk_delay_s)

def make_height_visuals_for_all_imgs_in_folder(folder_path, bulk_delay_s):
    conn = db.open_connection_to_database()
    time.sleep(bulk_delay_s)
    for folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            for file_name in os.listdir(folder_full_path):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    file_path = os.path.join(folder_full_path, file_name)
                    cam_name=str(folder)
                    process_and_make_copies_height_visuals(file_path, cam_name, bulk_delay_s, conn)
                    

#if you are processing lots of files, increase bulk delay to prevent choking.
def process_files(folder_path="/mnt/image/image-inbox", bulk_delay_s=1):
    try:
        make_height_visuals_for_all_imgs_in_folder(folder_path, bulk_delay_s) #this is the current runtime script that will process and copy photos for now.
    except Exception as e:
        print(e)
        print(f"\033[91mError\033[0m")
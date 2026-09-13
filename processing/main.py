from datetime import datetime
import time

import sys
import cv2
import numpy as np

def main():
   while(True):
       print("Agcam Processing Activated")


        image_paths = []

        #here put the processing code for the images; find them in the folder, then run open cv masking, measuring, save copies with the visuals, then saving everything to the database.
         

               print(f"Agcam Processing Successful for files: {image_paths}")


           except Exception as e:
               print(e)
               print(f"\033[91mError\033[0m")
          
       print("Standby")

       # wait 1 day to process new images again.
       time.sleep(86400)

main()
















































'''
while True:
   print("starting again")
   for camera_parameters in camera_parameter_list:
       image = None
       print(f"Getting and saving image from https:{camera_parameters['ip_address']}")
       try:
           image = image_util.get_image_from_camera_parameter(camera_parameters)
           camera_id = camera_parameters["camera_id"]
           save_path = image_util.save_image_to_server_directory(camera_id,image)
           cv2.imwrite(str(f"plant_requests/requestor/{camera_id}.jpg"), image)
           print(f"image successfully saved to : {save_path}")
       except Exception as e:
          
           print(e)
           print(f"failed to get image from camera {camera_parameters['camera_id']}")
       try:
           if image is not None:
               reference_tags = reference_util.scan_reference_tags(image,camera_parameters)
               for reference_tag in reference_tags:
                   response = hr.height_request(image, [reference_tag], camera_parameters)
              
               graph_path = f"plant_requests/requestor/{camera_id}_graph.jpg"
               graph_util.plot_height_request_response(image, graph_path,response)
               print(f"graph successfully saved to : {graph_path}")
       except Exception as e:
           print(e)
           print(f"failed to process height request for camera {camera_parameters['camera_id']}")


       time.sleep(5)
   print("taking a break")
   time.sleep(1800)


'''

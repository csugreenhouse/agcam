import cv2
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

def get_image_from_camera_parameter(camera_parameters):
    camera_ip = camera_parameters["ip_address"]
    camera_id = camera_parameters["camera_id"]
    camera_url = f"http://{camera_ip}/capture"
    try:
        response = requests.get(camera_url,timeout=10)
        response.raise_for_status()
        image = response.content
        return cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Camera {camera_id}'s ip {camera_url} failed")
        return None
    
def get_images_from_list_of_camera_parameters(camera_parameter_list):
    images = []
    for camera_parameter in camera_parameter_list:
        images.append(get_image_from_camera_parameter(camera_parameter))
    return images

def time_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_image_to_server_directory(camera_id, image):
    save_path = f'/srv/samba/plants/image/by_camera/cam{camera_id}/{time_stamp()}.jpg'
    cv2.imwrite(save_path, image)
    return save_path

def save_image_to_server_directory_by_plant(plant_id, image):
    save_path = f'/srv/samba/plants/image/by_plant/plant{plant_id}/{time_stamp()}.jpg'
    cv2.imwrite(save_path, image)
    return save_path

    


        
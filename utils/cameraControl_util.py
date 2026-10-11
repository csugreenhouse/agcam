import paramiko
import json
import time

import sys 

sys.path.append('/mnt/db/agcam')

def ssh_connect_to_camera(cam_number):
    with open("/mnt/db/agcam/utils/secrets_util.json") as f:
        secrets = json.load(f)["ssh"]
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(secrets["SSH_IP_BASE"]+str(cam_number))
        print("Successfully connected to camera via SSH")
        return ssh
    except Exception as e:
        print(f"Failed to connect to camera via SSH: {e}")
        return None 

def close_ssh_connection(ssh):
    if ssh:
        ssh.close()
        print("SSH connection closed")

def take_picture(ssh):
    command = "python3 image_capture_store_af_jpg_npy.py"
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.set_combine_stderr(True)
        output = stdout.readlines()
        print("Picture taken successfully")
    except Exception as e:
        print(f"Failed to take picture: {e}")

def send_pictures_to_server(ssh):
    command = "./Rpi-rsync.sh"
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.set_combine_stderr(True)
        output = stdout.readlines()
        print("Pictures sent to server successfully")
    except Exception as e:
        print(f"Failed to send pictures to server: {e}")

def run_pic_pipeline(cam_number):
    ssh = ssh_connect_to_camera(cam_number)
    take_picture(ssh)
    send_pictures_to_server(ssh)
    close_ssh_connection(ssh)
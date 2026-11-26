import paramiko
import json
import time
import subprocess

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

#method outline from https://stackoverflow.com/a/73315109 CC By SA 4.0
def check_ping(cam_number):
    import platform
    with open("/mnt/db/agcam/utils/secrets_util.json") as f:
        secrets = json.load(f)["ssh"]
    try:
        host = secrets["SSH_IP_BASE"]+str(cam_number)
        param = '-n' if platform.system().lower()=='windows' else '-c'
        command = ['ping', param, '1', host]

        result = subprocess.run(command, stdout=subprocess.PIPE)
        output = result.stdout.decode('utf8')
        if "Request timed out." in output or "100% packet loss" in output:
            print(f"Camera {cam_number} not connected")
            return "NOT CONNECTED"
        print(f"Camera {cam_number} connected")
        return "CONNECTED"
    except Exception as e:
        print(f"Camera {cam_number} not connected")
        return "NOT CONNECTED"
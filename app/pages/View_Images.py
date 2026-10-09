import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sys
import importlib
from pathlib import Path

sys.path.append("/mnt/db/agcam")

db = importlib.import_module("database.database")

try:
    conn = db.open_connection_to_database() 
    print("Successfully opened connection to database")
except Exception as e:
    st.error(f"Failed to open connection to database: {e}")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("View Images")

cam_number = st.number_input("Enter Camera Number", min_value=0, step=1)

def latest_file(path: Path, pattern: str = "*"):
    files = list(path.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_ctime)

def get_image(cam_number):
    folder_path = f"/mnt/image/images-processed/{cam_number}CAM"
    image_path = latest_file(Path(folder_path))
    return Image.open(image_path)

try:
    img = get_image(cam_number)
except Exception as e:
    st.write(f"Failed to load image: {e}")
    img = None

if img is not None:
    st.image(img, caption=f"Camera {cam_number} Latest Processed Image")
else:
    st.write(f"No image available for Camera {cam_number}.")
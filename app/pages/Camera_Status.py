import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib
import time
sys.path.append("/mnt/db/agcam")

cameraControl = importlib.import_module("utils.cameraControl_util")

def take_photo_button(cam_number):
    if st.button(f"Request Camera {cam_number} Photo"):
        with st.spinner("Taking Photo..."):
            cameraControl.run_pic_pipeline(cam_number)
            success = st.empty()
            success.success("Photo saved.")
            time.sleep(2)
            success.empty()

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Camera Status")

st.divider()

st.write(f"**_Camera 0:_**")
status_0 = cameraControl.check_ping(0)
if status_0 == "CONNECTED":
    st.write(f"**:green[{status_0}]**")
else:
    st.write(f"**:red[{status_0}]**")
take_photo_button(0)
st.divider()

st.write(f"**_Camera 1:_**")
status_1 = cameraControl.check_ping(1)
if status_1 == "CONNECTED":
    st.write(f"**:green[{status_1}]**")
else:
    st.write(f"**:red[{status_1}]**")
take_photo_button(1)

st.divider()

st.write(f"**_Camera 2:_**")
status_2 = cameraControl.check_ping(2)
if status_2 == "CONNECTED":
    st.write(f"**:green[{status_2}]**")
else:
    st.write(f"**:red[{status_2}]**")
take_photo_button(2)
st.divider()
st.write(f"**_Camera 3:_**")
status_3 = cameraControl.check_ping(3)
if status_3 == "CONNECTED":
    st.write(f"**:green[{status_3}]**")
else:
    st.write(f"**:red[{status_3}]**")
take_photo_button(3)
st.divider()

st.write(f"**_Camera 4:_**")
status_4 = cameraControl.check_ping(4)
if status_4 == "CONNECTED":
    st.write(f"**:green[{status_4}]**")
else:
    st.write(f"**:red[{status_4}]**")
take_photo_button(4)
st.divider()
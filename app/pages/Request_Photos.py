import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib
import time
sys.path.append("/mnt/db/agcam")

cameraControl = importlib.import_module("utils.cameraControl_util")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Request Photos")

cam_number = st.number_input("Enter Camera Number", min_value=0, step=1)
if st.button("Request Photo"):
    with st.spinner("Taking Photo..."):
        cameraControl.run_pic_pipeline(cam_number)
        success = st.empty()
        success.success("Photo saved.")
        time.sleep(2)
        success.empty()
    
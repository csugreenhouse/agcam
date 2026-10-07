import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib

sys.path.append("/mnt/db/agcam")

db = importlib.import_module("database.database")
cameraControl = importlib.import_module("database.cameraControl")

try:
    conn = db.open_connection_to_database() 
    print("Successfully opened connection to database")
except Exception as e:
    st.error(f"Failed to open connection to database: {e}")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Request Photos")

cam_number = st.number_input("Enter Camera Number", min_value=0, step=1)
if st.button("Request Photo"):
    cameraControl.run_pic_pipeline(cam_number)
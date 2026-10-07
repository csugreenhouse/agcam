import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib
import time

sys.path.append("/mnt/db/agcam")

db = importlib.import_module("database.database")
processor = importlib.import_module("processing.processor")

try:
    conn = db.open_connection_to_database() 
    print("Successfully opened connection to database")
except Exception as e:
    st.error(f"Failed to open connection to database: {e}")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Run Processing")

if st.button("Run Processing"):
    with st.spinner("Processing in progress."):
        processor.make_blobs_for_all_imgs_in_folder("/mnt/image/image-inbox")
        success = st.empty()
        success.success("Processing completed.")
        time.sleep(2)
        success.empty()
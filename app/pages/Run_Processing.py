import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib
import time

sys.path.append("/mnt/db/agcam")
st.title("Run Processing")

processor = importlib.import_module("processing.processor")

st.set_page_config(layout="wide")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if st.button("Run"):
    with st.spinner("Processing in progress."):
        processor.make_height_visuals_for_all_imgs_in_folder("/mnt/image/image-inbox")
        success = st.empty()
        success.success("Processing completed.")
        time.sleep(2)
        success.empty()
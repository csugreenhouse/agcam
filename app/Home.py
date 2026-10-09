#run the app by doing streamlit run app/Home.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path

sys.path.append("/mnt/db/agcam")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Welcome to Agcam")

st.html(open(Path("app/home.html")).read())
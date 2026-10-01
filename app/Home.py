#run the app by doing streamlit run app/Home.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib

sys.path.append("/mnt/db/agcam")

db = importlib.import_module("database.database")

try:
    conn = db.open_connection_to_database() 
    print("Successfully opened connection to database")
except Exception as e:
    st.error(f"Failed to open connection to database: {e}")

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

print("Loading app")

st.set_page_config(layout="wide")

st.title("Welcome to Agcam")
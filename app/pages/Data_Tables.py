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

st.set_page_config(layout="wide")

st.title("Data Tables")

#this section of code allows the user to specify a cam id and get database entries for it.
x = st.selectbox('Select a cam id', options=['cam0', 'cam1', 'cam2'])
dataframe = pd.read_sql(f"SELECT * FROM imgIndex WHERE cam_id = '{x[-1]}'", conn)
st.table(dataframe)
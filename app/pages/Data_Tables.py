import streamlit as st
import pandas as pd
import numpy as np
import sys
import importlib

sys.path.append("/mnt/db/agcam")

db = importlib.import_module("utils.database_util")

conn = db.open_connection_to_database()

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Data Tables")

#this section of code allows the user to specify a cam id and get database entries for it.
x = st.selectbox('Select a table', options=['heights', 'imgIndex'])

table_name = x
dataframe = db.return_table_as_dataframe(conn, table_name)

st.table(dataframe)
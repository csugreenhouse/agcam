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

st.title("Data to CSV")

#this section of code allows the user to specify a cam id and get database entries for it.
x = st.selectbox('Select a table', options=['heights', 'imgIndex'])
if st.button("Request CSV"):
    with st.spinner("Generating CSV..."):
        CSVfile_path = f"/mnt/db/agcam/app/pages/csv_out.csv"
        tablename = x
        db.table_to_CSV(tablename, CSVfile_path, conn)
        st.success(f"CSV generated at {CSVfile_path}")
        with open(CSVfile_path) as f:
            st.download_button('Download CSV', f)
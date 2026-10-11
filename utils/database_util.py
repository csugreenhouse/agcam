import cv2
import numpy as np
import sys 
from datetime import datetime
import psycopg2
import os
import json
import pandas as pd
import csv
import tabulate

sys.path.append('/mnt/db/agcam')

#####################################################
#these are the most general database control methods:
######################################################

def open_connection_to_database():
    with open("/mnt/db/agcam/utils/secrets_util.json") as f:
        secrets = json.load(f)["database"]
    try:
        conn = psycopg2.connect(
            host=secrets["PG_HOST"],
            dbname=secrets["PG_DATABASE"],
            user=secrets["PG_USER"],
            password=secrets["PG_PASSWORD"]
        )
        print("Successfully opened connection to database")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None
        conn.close() 
    
def close_connection_to_database(conn):
    if conn:
        conn.close() 

def execute_query(conn,query,params=None):
    try:
        cur = conn.cursor()
        cur.execute(query,params)
        if cur.description is not None:
            results = cur.fetchall()
        else:
            results = []
        conn.commit()
        cur.close()
        return results
    except Exception as e:
        conn.rollback()
        raise LookupError(f"Database error: {e}")
    finally:
        pass

################################################################
# These methods are used to interact with tables in the database.
################################################################

def append_photo_to_imgIndex(conn, cam_id, file_path): #need to generalize to be append to any table, but for now it is hardcoded for imgIndex ***
    query = (
        "INSERT INTO imgIndex (cam_id, file_path) "
        "VALUES (%s, %s);"
    )
    try:
        execute_query(conn, query, (cam_id, file_path))
    except Exception as e:
        print(f"Error appending photo to imgIndex: {e}")

def log_height(conn, cam_id, time_stamp, tag, height, file_path):
    query = (
        "INSERT INTO heights (cam_id, time_stamp, tag, height, file_path) "
        "VALUES (%s, %s, %s, %s, %s);"
    )
    try:
        execute_query(conn, query, (cam_id, time_stamp, tag, height, file_path))
    except Exception as e:
        print(f"Error logging height: {e}")

####################################################
# Functions that are used to output database tables.
####################################################

def table_to_CSV(tablename, CSVfile_path, conn): #very useful for exporting tables to CSV, final outputs
    query = f"COPY {tablename} TO STDIN CSV HEADER;"
    try:
        with conn.cursor() as cursor:
            cursor.copy_expert(query, open(CSVfile_path, "w"))
        conn.commit()
        return f"CSV written at {CSVfile_path}"
    except Exception as e:
        conn.rollback()
        raise LookupError(f"Database error: {e}")
    finally:
        pass

def results_to_CSV(results, CSVfile_path): #not highly useful
    with open(CSVfile_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results[0].keys())
        for entry in results:
            writer.writerow(entry.values())
        print(CSVfile_path)

def print_table_in_terminal(conn, table_name): #useful for debug
    try:
        df = return_table_as_dataframe(conn, table_name)
        print(df.to_markdown())
    except Exception as e:
        print(f"Error printing table in terminal: {e}")

def return_table_as_dataframe(conn, table_name, where_option=None): #useful for printing in streamlit app
    query = f"SELECT * FROM {table_name}"
    if where_option:
        query += f" WHERE {where_option}"
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        raise LookupError(f"Database error: {e}")